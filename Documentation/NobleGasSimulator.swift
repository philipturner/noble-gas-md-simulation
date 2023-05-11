//
//  NobleGasSimulator.swift
//  MolecularRenderer
//
//  Created by Philip Turner on 4/23/23.
//

import Foundation
import PythonKit
import simd
import QuartzCore
import Metal

PythonLibrary.useLibrary(at: "/Users/philipturner/miniforge3/bin/python")
let plt = Python.import("matplotlib.pyplot")

// MARK: - Setup

// Run the molecular dynamics simulation in FP32 because energy drift is not a
// major issue yet.
typealias Real = Float32

// Lennard-Jones parameters in nm and zJ. These will be down-casted to FP32
// when running the molecular dynamics simulation.
struct LJParameters {
  var sigma: Double
  var epsilon: Double
}

// Map of atomic numbers to LJ parameters.
let ljParametersMap: [Int: LJParameters] = [
  2: LJParameters(sigma: 0.2411, epsilon: 0.732),
  10: LJParameters(sigma: 0.2687, epsilon: 1.371),
  18: LJParameters(sigma: 0.3425, epsilon: 2.189),
  36: LJParameters(sigma: 0.3698, epsilon: 2.916),
]

struct AtomPair: Hashable {
  // The smaller element in the pair.
  var lower: Int
  
  // The larger element in the pair.
  var upper: Int
  
  // 'i' and 'j' can be in any order.
  init(i: Int, j: Int) {
    self.lower = min(i, j)
    self.upper = max(i, j)
  }
}

// Map of atom pairs to LJ parameters.
var ljParametersPairMap: [AtomPair: LJParameters] = [:]
for (i, iParameters) in ljParametersMap {
  for (j, jParameters) in ljParametersMap {
    let pair = AtomPair(i: i, j: j)
    var parameters: LJParameters
    
    if i == j {
      parameters = iParameters
    } else {
      let sigma_ii = iParameters.sigma
      let sigma_jj = jParameters.sigma
      
#if true
      let sigma_ii_3 = sigma_ii * sigma_ii * sigma_ii
      let sigma_jj_3 = sigma_jj * sigma_jj * sigma_jj
      let sigma_6_avg = (sigma_ii_3 * sigma_ii_3 + sigma_jj_3 * sigma_jj_3) / 2
      
      let sigma_ij = pow(sigma_6_avg, Double(1.0) / 6)
      var epsilon_ij = sigma_ii_3 * sigma_jj_3 / sigma_6_avg
      epsilon_ij *= sqrt(iParameters.epsilon * jParameters.epsilon)
#else
      let sigma_ij = (sigma_ii + sigma_jj) / 2
      let epsilon_ij = sqrt(iParameters.epsilon * jParameters.epsilon)
#endif
      
      parameters = LJParameters(sigma: sigma_ij, epsilon: epsilon_ij)
    }
    
    ljParametersPairMap[pair] = parameters
  }
}

print("LJ Parameters:")
for i in ljParametersMap.keys.sorted() {
  for j in ljParametersMap.keys.sorted() {
    let atomPair = AtomPair(i: i, j: j)
    let parameters = ljParametersPairMap[atomPair]!
    print(atomPair,
          "sigma", String(format: "%.3f", parameters.sigma),
          "epsilon", String(format: "%.3f", parameters.epsilon))
  }
}

struct LJPotentialParameters {
  // 48 * epsilon * sigma^12
  // Units: aJ * nm^12
  var c12: Real
  
  // -24 * epsilon * sigma^6
  // Units: aJ * nm^6
  var c6: Real
  
  // Input sigma in nm, epsilon in zJ.
  init(sigma: Double, epsilon: Double) {
    let sigma2 = sigma * sigma
    let sigma6 = sigma2 * sigma2 * sigma2
    let epsilon_aJ = epsilon * 0.001
    self.c12 = Real(48 * epsilon_aJ * sigma6 * sigma6)
    self.c6 = Real(-24 * epsilon_aJ * sigma6)
  }
}

// Eventually, this should be optimized to use a raw pointer, to avoid bounds
// checking during the inner loop. Another optimization could be to elide the
// matrix entirely, if you can guarantee every atom is of the same element.
class LJPotentialParametersMatrix {
  var width: Int
  var height: Int
  var elements: UnsafeMutableBufferPointer<LJPotentialParameters>
  var gpuBuffer: MTLBuffer?
  var gpuOffset: Int = 0
  
  init(width: Int, height: Int) {
    self.width = width
    self.height = height
    
    let size = width * height
    let initial = LJPotentialParameters(sigma: 0, epsilon: 0)
    self.elements = .allocate(capacity: size)
    self.elements.initialize(repeating: initial)
  }
  
  func initializeGPU(device: MTLDevice) {
    let numBytes = width * height * MemoryLayout<LJPotentialParameters>.stride
    let addressBase = UInt(bitPattern: elements.baseAddress!)
    let addressAligned = ~(4096 - 1) & addressBase
    let pointerAligned = UnsafeMutableRawPointer(bitPattern: addressAligned)!
    let numBytesAligned = ~(4096 - 1) & (numBytes + 4096 - 1)
    
    self.gpuBuffer = device.makeBuffer(
      bytesNoCopy: pointerAligned, length: numBytesAligned)!
    self.gpuOffset = Int(addressBase - addressAligned)
  }
  
  func index(row: Int, column: Int) -> Int {
    self.width * row + column
  }
  
  func setElement(_ value: LJPotentialParameters, index: Int) {
    self.elements[index] = value
  }
  
  // A highly performant index into the array.
  @inline(__always)
  func getElement(index: Int, row: Int, column: Int) -> LJPotentialParameters {
    #if DEBUG
    assert(index == row * self.width + column)
    assert(index / self.width == row)
    assert(index % self.width == column)
    assert(row >= 0 && row < self.height)
    assert(column >= 0 && column < self.width)
    let baseAddress = self.elements.baseAddress!
    #else
    let baseAddress = self.elements.baseAddress.unsafelyUnwrapped
    #endif
    
    return baseAddress[index]
  }
}

// There are 119 elements (including neutronium). This will allocate 200 KB,
// most of which is unused.
var parametersMatrix = LJPotentialParametersMatrix(width: 119, height: 119)
for i in ljParametersMap.keys {
  for j in ljParametersMap.keys {
    let atomPair = AtomPair(i: i, j: j)
    let parameters = ljParametersPairMap[atomPair]!
    let potentialParameters = LJPotentialParameters(
      sigma: parameters.sigma, epsilon: parameters.epsilon)
    
    let index = parametersMatrix.index(row: i, column: j)
    parametersMatrix.setElement(potentialParameters, index: index)
  }
}

print()
print("LJ Parameters Matrix:")
for i in ljParametersMap.keys.sorted() {
  for j in ljParametersMap.keys.sorted() {
    let index = parametersMatrix.index(row: i, column: j)
    let parameters = parametersMatrix
      .getElement(index: index, row: i, column: j)
    print(i, j,
          "c12", String(/*format: "%.3f",*/ parameters.c12),
          "c6", String(/*format: "%.3f",*/ parameters.c6))
  }
}

// MARK: - Boilerplate code for VectorBlock implementation

protocol VectorBlock {
  associatedtype Scalar
  
  static var scalarCount: Int { get }
  
  static var zero: Self { get }
  
  func forEachElement(_ closure: (Int, Scalar) -> Void)
}

extension UInt8: VectorBlock {
  static var scalarCount: Int { 1 }
  
  @inline(__always)
  func forEachElement(_ closure: (Int, Self) -> Void) { closure(0, self) }
}

extension Float: VectorBlock {
  static var scalarCount: Int { 1 }
  
  @inline(__always)
  func forEachElement(_ closure: (Int, Self) -> Void) { closure(0, self) }
}

extension Double: VectorBlock {
  static var scalarCount: Int { 1 }
  
  @inline(__always)
  func forEachElement(_ closure: (Int, Self) -> Void) { closure(0, self) }
}

extension SIMD2: VectorBlock where Scalar: VectorBlock {
  static var zero: Self { .init(repeating: .zero) }
  
  @inline(__always)
  func forEachElement(_ closure: (Int, Scalar) -> Void) {
    for i in 0..<scalarCount {
      closure(i, self[i])
    }
  }
}

extension SIMD4: VectorBlock where Scalar: VectorBlock {
  static var zero: Self { .init(repeating: .zero) }
  
  @inline(__always)
  func forEachElement(_ closure: (Int, Scalar) -> Void) {
    for i in 0..<scalarCount {
      closure(i, self[i])
    }
  }
}

extension SIMD8: VectorBlock where Scalar: VectorBlock {
  static var zero: Self { .init(repeating: .zero) }
  
  @inline(__always)
  func forEachElement(_ closure: (Int, Scalar) -> Void) {
    for i in 0..<scalarCount {
      closure(i, self[i])
    }
  }
}

extension SIMD16: VectorBlock where Scalar: VectorBlock {
  static var zero: Self { .init(repeating: .zero) }
  
  @inline(__always)
  func forEachElement(_ closure: (Int, Scalar) -> Void) {
    for i in 0..<scalarCount {
      closure(i, self[i])
    }
  }
}

extension SIMD32: VectorBlock where Scalar: VectorBlock {
  static var zero: Self { .init(repeating: .zero) }
  
  @inline(__always)
  func forEachElement(_ closure: (Int, Scalar) -> Void) {
    for i in 0..<scalarCount {
      closure(i, self[i])
    }
  }
}

extension SIMD64: VectorBlock where Scalar: VectorBlock {
  static var zero: Self { .init(repeating: .zero) }
  
  @inline(__always)
  func forEachElement(_ closure: (Int, Scalar) -> Void) {
    for i in 0..<scalarCount {
      closure(i, self[i])
    }
  }
}

// WARNING: Make sure to avoid copy-on-write semantics!
struct Vector<T> {
  var count: Int
  var alignedCount: Int
  private(set) var elements: [T]
  var bufferPointer: UnsafeMutableBufferPointer<T>
  var gpuBuffer: MTLBuffer?
  var gpuOffset: Int = 0
  
  init(repeating repeatedValue: T, count: Int, alignment: Int) {
    self.count = count
    self.alignedCount = ~(alignment - 1) & (count + alignment - 1)
    self.elements = Array(repeating: repeatedValue, count: alignedCount)
    self.bufferPointer = elements.withUnsafeMutableBufferPointer { $0 }
  }
  
  mutating func initializeGPU(device: MTLDevice) {
    let numBytes = alignedCount * MemoryLayout<T>.stride
    let addressBase = UInt(bitPattern: bufferPointer.baseAddress!)
    let addressAligned = ~(4096 - 1) & addressBase
    let pointerAligned = UnsafeMutableRawPointer(bitPattern: addressAligned)!
    let numBytesAligned = ~(4096 - 1) & (numBytes + 4096 - 1)
    
    self.gpuBuffer = device.makeBuffer(
      bytesNoCopy: pointerAligned, length: numBytesAligned)!
    self.gpuOffset = Int(addressBase - addressAligned)
  }
  
  // A highly performant index into the array.
  @inline(__always)
  mutating func setElement(_ value: T, index: Int) {
    #if DEBUG
    elements.withUnsafeMutableBufferPointer { bufferPointer in
      assert(index >= 0 && index < self.count)
      assert(self.bufferPointer.baseAddress == bufferPointer.baseAddress)
      assert(self.bufferPointer.count == bufferPointer.count)
    }
    let baseAddress = self.bufferPointer.baseAddress!
    #else
    let baseAddress = self.bufferPointer.baseAddress.unsafelyUnwrapped
    #endif

    baseAddress[index] = value
  }
  
  // A highly performant index into the array.
  @inline(__always)
  mutating func setElement<
    U: VectorBlock
  >(_ value: U, type: U.Type, actualIndex: Int) {
    #if DEBUG
    elements.withUnsafeMutableBufferPointer { bufferPointer in
      assert(actualIndex >= 0 && actualIndex < self.count)
      assert(self.bufferPointer.baseAddress == bufferPointer.baseAddress)
      assert(self.bufferPointer.count == bufferPointer.count)
      assert(actualIndex % U.scalarCount == 0)
    }
    let baseAddress = self.bufferPointer.baseAddress!
    #else
    let baseAddress = self.bufferPointer.baseAddress.unsafelyUnwrapped
    #endif

    let castedAddress = unsafeBitCast(
      baseAddress + actualIndex, to: UnsafeMutablePointer<U>.self)
    castedAddress.pointee = value
  }
  
  // A highly performant index into the array.
  @inline(__always)
  mutating func getElement(index: Int) -> T {
    #if DEBUG
    elements.withUnsafeMutableBufferPointer { bufferPointer in
      assert(index >= 0 && index < self.count)
      assert(self.bufferPointer.baseAddress == bufferPointer.baseAddress)
      assert(self.bufferPointer.count == bufferPointer.count)
    }
    let baseAddress = self.bufferPointer.baseAddress!
    #else
    let baseAddress = self.bufferPointer.baseAddress.unsafelyUnwrapped
    #endif
    
    return baseAddress[index]
  }
  
  // A highly performant index into the array.
  @inline(__always)
  mutating func getElement<
    U: VectorBlock
  >(type: U.Type, actualIndex: Int) -> U {
    #if DEBUG
    elements.withUnsafeMutableBufferPointer { bufferPointer in
      assert(actualIndex >= 0 && actualIndex < self.count)
      assert(self.bufferPointer.baseAddress == bufferPointer.baseAddress)
      assert(self.bufferPointer.count == bufferPointer.count)
      assert(actualIndex % U.scalarCount == 0)
    }
    let baseAddress = self.bufferPointer.baseAddress!
    #else
    let baseAddress = self.bufferPointer.baseAddress.unsafelyUnwrapped
    #endif

    let castedAddress = unsafeBitCast(
      baseAddress + actualIndex, to: UnsafeMutablePointer<U>.self)
    return castedAddress.pointee
  }
}

class AtomicMasses {
  // Mass of each element on the periodic table, in kg.
  private var elements: Vector<Double>
  
  // Pretend-masked as 'Double' in Swift, but actually 'DoubleSingle'
  private(set) var elementsGPU: Vector<Double>?
  
  init() {
    self.elements = .init(repeating: 0, count: 119, alignment: 1)
    
    let amuToGrams = 1 / Double(6.022e23)
    let amuToKg = amuToGrams / 1000
    elements.setElement(4.002602 * amuToKg, index: 2)
    elements.setElement(20.1797 * amuToKg, index: 10)
    elements.setElement(39.948 * amuToKg, index: 18)
    elements.setElement(83.798 * amuToKg, index: 36)
  }
  
  func initializeGPU(device: MTLDevice) {
    self.elementsGPU = .init(repeating: 0, count: 119, alignment: 1)
    self.elementsGPU?.initializeGPU(device: device)
    
    for index in 0..<elementsGPU!.count {
      let fp64 = simd_precise_recip(self.elements.getElement(index: index))
      let fp32 = unsafeBitCast(DoubleSingle(fp64), to: Double.self)
      self.elementsGPU?.setElement(fp32, index: index)
    }
  }
  
  @inline(__always)
  func getMass(atomicNumber: UInt8) -> Double {
    self.elements.getElement(index: Int(atomicNumber))
  }
}

// MARK: - Simulation

// A very small MD simulation that will not benefit from cutoffs or modern
// optimization algorithms. Computational complexity is O(n^2).
//
// Stores the intermediate data in range-reduced single precision for nonbonded
// force calculations, but expands to double precision (SI units) elsewhere.
//
// WARNING: Make sure to avoid copy-on-write semantics!
struct System {
  // The amount of data to store in registers while traversing down the matrix
  // of atom pairs.
  typealias Block = SIMD8<Real>
  typealias WideBlock = SIMD8<Double>
  
  // The type used for the self-interaction mask during the inner loop. Use the
  // smallest value possible that fits the required number of atoms.
  typealias Index = UInt16
  typealias IndexBlock = SIMD8<Index>
  typealias AtomicNumberBlock = SIMD8<UInt8>
  
  // Simulation parameters
  var atoms: Int
  var ljParameters: LJPotentialParametersMatrix
  var masses: AtomicMasses
  var accelerationsInitialized: Bool = false
  
  // Resources for GPU acceleration
  var useGPU: Bool = false
  var device: MTLDevice
  var commandQueue: MTLCommandQueue
  var commandBuffer: MTLCommandBuffer?
  var encoder: MTLComputeCommandEncoder?
  
  // Arguments for the GPU kernel
  struct Arguments {
    var h: DoubleSingle
    var billionth: DoubleSingle
    var thousandth: DoubleSingle
  }
  
  // If this value changes, stall the pipeline to avoid corrupting the other
  // command buffers.
  var lastH: Double = -1
  var argumentsBuffer: MTLBuffer
  var updatePositionsPipeline: MTLComputePipelineState
  var updateVelocitiesPipeline: MTLComputePipelineState
  // ICB, which can be optionally enabled or disabled
  
  // Atomic number
  private var element: Vector<UInt8>
  
  // Position - nm
  private var x: Vector<Real>
  private var y: Vector<Real>
  private var z: Vector<Real>
  
  // Velocity - nm/ps
  private var v_x: Vector<Real>
  private var v_y: Vector<Real>
  private var v_z: Vector<Real>
  
  // Acceleration - nm/ps/s
  private var a_x: Vector<Real>
  private var a_y: Vector<Real>
  private var a_z: Vector<Real>
  
  // Initialize all the underlying memory. The caller fills in atom positions
  // outside of the loop.
  init(atoms: Int, ljParameters: LJPotentialParametersMatrix) {
    self.atoms = atoms
    self.ljParameters = ljParameters
    self.masses = AtomicMasses()
    
    let align = Block.scalarCount
    
    self.element = .init(repeating: 0, count: atoms, alignment: align)
    
    self.x = .init(repeating: 0, count: atoms, alignment: align)
    self.y = .init(repeating: 0, count: atoms, alignment: align)
    self.z = .init(repeating: 0, count: atoms, alignment: align)
    
    self.v_x = .init(repeating: 0, count: atoms, alignment: align)
    self.v_y = .init(repeating: 0, count: atoms, alignment: align)
    self.v_z = .init(repeating: 0, count: atoms, alignment: align)
    
    self.a_x = .init(repeating: 0, count: atoms, alignment: align)
    self.a_y = .init(repeating: 0, count: atoms, alignment: align)
    self.a_z = .init(repeating: 0, count: atoms, alignment: align)
    
    self.device = MTLCopyAllDevices().first!
    self.commandQueue = device.makeCommandQueue(maxCommandBufferCount: 8)!
    
    let source = createGPUEvolveSrc()
    let options = MTLCompileOptions()
    options.fastMathEnabled = false
    let library = try! device.makeLibrary(source: source, options: options)
    
    self.argumentsBuffer = device.makeBuffer(
      length: MemoryLayout<Arguments>.stride)!
    self.updatePositionsPipeline = try! device
      .makeComputePipelineState(function: library
        .makeFunction(name: "updatePositions")!)
    self.updateVelocitiesPipeline = try! device
      .makeComputePipelineState(function: library
        .makeFunction(name: "updateVelocities")!)
  }
  
  mutating func initializeAccelerations() {
    // Set the accelerations to something nonzero, so the Verlet integrator can
    // produce something correct on the first iteration.
    self._evolve(timeStep: 0, onlyUpdatingAcceleration: true)
    self.accelerationsInitialized = true
  }
  
  // Tell the system to use the GPU for the bulk of the computations. This is
  // a partial port from CPU -> GPU, offloading the computations but not fully
  // optimizing them with ICBs and interleaved memory formats.
  mutating func initializeGPU() {
    self.useGPU = true
    self.ljParameters.initializeGPU(device: device)
    self.masses.initializeGPU(device: device)
    
    self.element.initializeGPU(device: device)
    
    self.x.initializeGPU(device: device)
    self.y.initializeGPU(device: device)
    self.z.initializeGPU(device: device)
    
    self.v_x.initializeGPU(device: device)
    self.v_y.initializeGPU(device: device)
    self.v_z.initializeGPU(device: device)
    
    self.a_x.initializeGPU(device: device)
    self.a_y.initializeGPU(device: device)
    self.a_z.initializeGPU(device: device)
  }
  
  // Call this *before* creating the next command buffer and encoder.
  mutating func updateArguments(timeStep h: Double) {
    guard lastH != h else {
      return
    }
    synchronizeGPU()
    self.lastH = h
    
    let dst = argumentsBuffer.contents().assumingMemoryBound(to: Arguments.self)
    dst.pointee = Arguments(
      h: .init(h), billionth: .init(1e-9), thousandth: .init(1e-3))
  }
  
  mutating func synchronizeGPU() {
    if let commandBuffer {
      if let encoder {
        encoder.endEncoding()
        self.encoder = nil
      }
      if commandBuffer.status == .notEnqueued {
        commandBuffer.commit()
      }
      commandBuffer.waitUntilCompleted()
      self.commandBuffer = nil
    }
  }
  
  mutating func setType(_ value: UInt8, index: Int) {
    precondition(commandBuffer == nil)
    element.setElement(value, index: index)
  }
  
  mutating func getType(index: Int) -> UInt8 {
    precondition(commandBuffer == nil)
    return element.getElement(index: index)
  }
  
  mutating func setPosition(_ value: SIMD3<Real>, index: Int) {
    precondition(commandBuffer == nil)
    x.setElement(value.x, index: index)
    y.setElement(value.y, index: index)
    z.setElement(value.z, index: index)
  }
  
  mutating func getPosition(index: Int) -> SIMD3<Real> {
    precondition(commandBuffer == nil)
    return SIMD3(
      x.getElement(index: index),
      y.getElement(index: index),
      z.getElement(index: index)
    )
  }
  
  mutating func setVelocity(_ value: SIMD3<Real>, index: Int) {
    precondition(commandBuffer == nil)
    v_x.setElement(value.x, index: index)
    v_y.setElement(value.y, index: index)
    v_z.setElement(value.z, index: index)
  }
  
  mutating func getVelocity(index: Int) -> SIMD3<Real> {
    precondition(commandBuffer == nil)
    return SIMD3(
      v_x.getElement(index: index),
      v_y.getElement(index: index),
      v_z.getElement(index: index)
    )
  }
  
  mutating func getAcceleration(index: Int) -> SIMD3<Real> {
    precondition(commandBuffer == nil)
    return SIMD3(
      a_x.getElement(index: index),
      a_y.getElement(index: index),
      a_z.getElement(index: index)
    )
  }
}

extension System {
  mutating func description(atomRange: Range<Int>) -> String {
    func fmt(_ value: Real) -> String {
      String(format: "%.3f", value)
    }
    func fmt(_ value: SIMD3<Real>) -> String {
      "[\(fmt(value.x)), \(fmt(value.y)), \(fmt(value.z))]"
    }
    
    var output: String = ""
    for i in atomRange {
      let type = getType(index: i)
      let position = getPosition(index: i)
      let velocity = getVelocity(index: i)
      let acceleration = getAcceleration(index: i)
      output.append("Z=\(type)\t ")
      output.append("p = \(fmt(position)) nm \t")
      output.append("v = \(fmt(velocity)) nm/ps \t")
      output.append("a = \(fmt(acceleration)) nm/ps/s")
      output.append("\n")
    }
    if output.count > 0 {
      // Remove the last newline.
      output.removeLast(1)
    }
    return output
  }
}

extension System {
  // This operation has perfect elementwise parallelism. An advanced
  // implementation could try parallelizing across multiple CPU cores.
  //
  // 'timesStep' is in seconds.
  mutating func evolve(timeStep h: Double, steps: Int = 1) {
    precondition(accelerationsInitialized)
    if !useGPU {
      for _ in 0..<steps {
        self._evolve(timeStep: h)
      }
    } else {
      var i = 0
      let chunkSize = 512 // Number of timesteps per command buffer.
      while true {
        var shouldExit = false
        var i_next = i + chunkSize
        if i_next >= steps {
          shouldExit = true
          i_next = steps
        }
        
        self._evolveGPU(timeStep: h, steps: i_next - i)
        
        if shouldExit {
          break
        } else {
          i = i_next
        }
      }
      self.synchronizeGPU()
    }
  }
  
  // Returns energy in joules.
  mutating func getEnergy() -> (kinetic: Double, potential: Double) {
    precondition(accelerationsInitialized)
    var kinetic: Double = 0
    var potential: Double = 0
    self._evolve(
      timeStep: 0, queryEnergy: true, kineticEnergy: &kinetic,
      potentialEnergy: &potential)
    return (kinetic, potential)
  }
  
  // Returns the energy after the time step has finished. If querying energy, do
  // not update any state variables.
  @inline(__always)
  private mutating func _evolve(
    timeStep h: Double,
    onlyUpdatingAcceleration: Bool = false,
    queryEnergy: Bool = false,
    kineticEnergy: UnsafeMutablePointer<Double>? = nil,
    potentialEnergy: UnsafeMutablePointer<Double>? = nil
  ) {
    let loopWidth = Block.scalarCount
    var i1 = 0 // initial i; do not modify this later
    while i1 < atoms {
      defer { i1 += loopWidth }
      
      // Some numbers are stored in SI units, while others values are
      // range-reduced to fit single precision. Any range-reduced numbers should
      // be explicitly labeled.
      
      // nm: 1e-9
      let x_curr = x.getElement(type: Block.self, actualIndex: i1)
      let y_curr = y.getElement(type: Block.self, actualIndex: i1)
      let z_curr = z.getElement(type: Block.self, actualIndex: i1)
      
      // nm/ps: 1e3
      let v_x_curr = v_x.getElement(type: Block.self, actualIndex: i1)
      let v_y_curr = v_y.getElement(type: Block.self, actualIndex: i1)
      let v_z_curr = v_z.getElement(type: Block.self, actualIndex: i1)
      
      // nm/ps/s: 1e3
      // NOTE: nm/ps/s could be very off. I may need a larger/smaller unit.
      let a_x_curr = a_x.getElement(type: Block.self, actualIndex: i1)
      let a_y_curr = a_y.getElement(type: Block.self, actualIndex: i1)
      let a_z_curr = a_z.getElement(type: Block.self, actualIndex: i1)
      
      // Apply velocity Verlet integrator.
      
      // m: 1e0
      let nm___m: Double = 1e-9
      var x_next = nm___m * WideBlock(x_curr)
      var y_next = nm___m * WideBlock(y_curr)
      var z_next = nm___m * WideBlock(z_curr)
      
      let nm_ps___m_s: Double = 1e3
      x_next += nm_ps___m_s * h * WideBlock(v_x_curr)
      y_next += nm_ps___m_s * h * WideBlock(v_y_curr)
      z_next += nm_ps___m_s * h * WideBlock(v_z_curr)
      
      let nm_ps_s___m_s2: Double = 1e3
      let multiplier: Double = 0.5 * h * h
      x_next += nm_ps_s___m_s2 * multiplier * WideBlock(a_x_curr)
      y_next += nm_ps_s___m_s2 * multiplier * WideBlock(a_y_curr)
      z_next += nm_ps_s___m_s2 * multiplier * WideBlock(a_z_curr)
      
      // nm: 1e-9
      let m___nm: Double = 1e9
      let x_i = Block(x_next * m___nm)
      let y_i = Block(y_next * m___nm)
      let z_i = Block(z_next * m___nm)
      
      // Update the positions *before* calculating forces.
      if !queryEnergy && !onlyUpdatingAcceleration {
        self.x.setElement(x_i, type: Block.self, actualIndex: i1)
        self.y.setElement(y_i, type: Block.self, actualIndex: i1)
        self.z.setElement(z_i, type: Block.self, actualIndex: i1)
      }
    }
    
    // TODO: Add autotuning functionality so it chooses the optimal number of
    // cores automatically, but without sacrificing performance for extremely
    // small simulations.
    let numCores = 1
    
    func execute(coreID: Int) {
      var i2 = loopWidth * coreID
      while i2 < atoms {
        defer { i2 += loopWidth * numCores }
        
        // nm: 1e-9
        let x_i = x.getElement(type: Block.self, actualIndex: i2)
        let y_i = y.getElement(type: Block.self, actualIndex: i2)
        let z_i = z.getElement(type: Block.self, actualIndex: i2)
        
        // nm/ps: 1e3
        let v_x_curr = v_x.getElement(type: Block.self, actualIndex: i2)
        let v_y_curr = v_y.getElement(type: Block.self, actualIndex: i2)
        let v_z_curr = v_z.getElement(type: Block.self, actualIndex: i2)
        
        // nm/ps/s: 1e3
        let a_x_curr = a_x.getElement(type: Block.self, actualIndex: i2)
        let a_y_curr = a_y.getElement(type: Block.self, actualIndex: i2)
        let a_z_curr = a_z.getElement(type: Block.self, actualIndex: i2)
        
        // Generate a coordinate vector of indices.
        var indexInList: IndexBlock = .zero
        for k in 0..<Block.scalarCount {
          indexInList[k] = Index(i2 + k)
        }
        
        // Atomic number
        let id = element.getElement(
          type: AtomicNumberBlock.self, actualIndex: i2)
        
        // nN: 1e-9
        var f_x_next: Block = .zero
        var f_y_next: Block = .zero
        var f_z_next: Block = .zero
        
        // aJ: 1e-18, but needs to be scaled by 1/12
        var u_next: Block = .zero
        
        // Find new accelerations and update velocities.
        defer {
          // kg: 1e0
          var mass: WideBlock = .zero
          id.forEachElement { i, element in
            mass[i] = self.masses.getMass(atomicNumber: element)
          }
          
          // kg^-1: 1e0
          let inverseMass = simd_precise_recip(mass)
          
          // m/s^2: 1e0
          let nm_s2___m_s2: Double = 1e-9
          let a_x_next = nm_s2___m_s2 * inverseMass * WideBlock(f_x_next)
          let a_y_next = nm_s2___m_s2 * inverseMass * WideBlock(f_y_next)
          let a_z_next = nm_s2___m_s2 * inverseMass * WideBlock(f_z_next)
          
          // Store the next accelerations.
          if !queryEnergy {
            // nm/ps/s: 1e3
            let m_s2___nm_ps_s: Double = 1e-3
            let _a_x_next = Block(a_x_next * m_s2___nm_ps_s)
            let _a_y_next = Block(a_y_next * m_s2___nm_ps_s)
            let _a_z_next = Block(a_z_next * m_s2___nm_ps_s)
            self.a_x.setElement(_a_x_next, type: Block.self, actualIndex: i2)
            self.a_y.setElement(_a_y_next, type: Block.self, actualIndex: i2)
            self.a_z.setElement(_a_z_next, type: Block.self, actualIndex: i2)
          }
          
          // Average the accelerations.
          // The current value is twice the average acceleration. A later line of
          // code will correct for the factor of 2.
          let nm_ps_s___m_s2: Double = 1e3
          let a_x_sum = a_x_next + nm_ps_s___m_s2 * WideBlock(a_x_curr)
          let a_y_sum = a_y_next + nm_ps_s___m_s2 * WideBlock(a_y_curr)
          let a_z_sum = a_z_next + nm_ps_s___m_s2 * WideBlock(a_z_curr)
          
          // nm/ps: 1e3
          let nm_ps___m_s: Double = 1e3
          var v_x_next = nm_ps___m_s * WideBlock(v_x_curr)
          var v_y_next = nm_ps___m_s * WideBlock(v_y_curr)
          var v_z_next = nm_ps___m_s * WideBlock(v_z_curr)
          
          let multiplier: Double = 0.5 * h
          v_x_next += multiplier * a_x_sum
          v_y_next += multiplier * a_y_sum
          v_z_next += multiplier * a_z_sum
          
          if queryEnergy {
            var v2 = v_x_next * v_x_next
            v2.addProduct(v_y_next, v_y_next)
            v2.addProduct(v_z_next, v_z_next)
            
            let kineticProduct = 0.5 * mass * v2
            var kineticSum: Double = 0
            for k in 0..<min(Block.scalarCount, atoms - i2) {
              kineticSum += kineticProduct[k]
            }
            kineticEnergy!.pointee += kineticSum
            
            // aJ: 1e-18
            let aJ___J: Double = 1e-18
            
            // This doesn't make any sense - apparently the potential energy is
            // attributed to a pair of particles. But, the force is applied twice
            // - once to particle i, another time to particle j. That means the
            // force (derivative of U) is applied twice:
            //
            // F = -delta U / delta r
            // F = F_i = -F_j
            // new U = U - F_i * delta r - (-F_j) * delta (-r)
            //
            // assume both particles are the same and |F_i| = |F_j|
            // new U = U - 2 * F * delta r
            // new U = U + 2 * delta U
            // new U - U = 2 * delta U
            // delta U = 2 * delta U
            //
            // U = 2U ?????????????????????????
            //
            // For some reason, this formula works, even though the math doesn't
            // seem to work out.
            let multiplier = aJ___J / 12 / 2
            let potentialProduct = multiplier * WideBlock(u_next)
            var potentialSum: Double = 0
            for k in 0..<min(Block.scalarCount, atoms - i2) {
              potentialSum += potentialProduct[k]
            }
            potentialEnergy!.pointee += potentialSum
          }
          
          if !queryEnergy && !onlyUpdatingAcceleration {
            // TODO: Implement a good thermostat. Modify the velocities before
            // the next time step, so they remain at the right temperature.
            
            // nm/ps: 1e3
            let m_s___nm_ps: Double = 1e-3
            let _v_x_next = Block(v_x_next * m_s___nm_ps)
            let _v_y_next = Block(v_y_next * m_s___nm_ps)
            let _v_z_next = Block(v_z_next * m_s___nm_ps)
            self.v_x.setElement(_v_x_next, type: Block.self, actualIndex: i2)
            self.v_y.setElement(_v_y_next, type: Block.self, actualIndex: i2)
            self.v_z.setElement(_v_z_next, type: Block.self, actualIndex: i2)
          }
        }
        
        var parameterIndexBase: IndexBlock = .zero
        id.forEachElement { i, element in
          parameterIndexBase[i] = Index(
            self.ljParameters.index(row: Int(element), column: 0))
        }
        
        // Calculate nonbonded forces.
        for j in 0..<Index(atoms) {
          let id_j = element.getElement(index: Int(j))
          
          // C12 = 48 * epsilon * sigma^12
          // C6 = -24 * epsilon * sigma^6
          
          // aJ * nm^12: 1e-126
          var c12: Block = .zero
          
          // aJ * nm^6: 1e-72
          var c6: Block = .zero
          
          // NOTE: If simulating only one element, you can elide the parameter
          // query, leading to a significant speedup.
          let matrixIndex = parameterIndexBase &+ Index(id_j)
          for k in 0..<Block.scalarCount {
            let index = matrixIndex[k]
            let parameters = self.ljParameters.getElement(
              index: Int(index), row: Int(id[k]), column: Int(id_j))
            c12[k] = parameters.c12
            c6[k] = parameters.c6
          }
          
          // TODO: Maybe store positions in both split and interleaved formats to
          // reduce the number of memory instructions here. All three memory
          // accesses can be issued in a single cycle, so the optimization might
          // complicate things and backfire.
          let x_j = self.x.getElement(index: Int(j))
          let y_j = self.y.getElement(index: Int(j))
          let z_j = self.z.getElement(index: Int(j))
          
          // Order of subtraction: if LJ force is positive, we should push the
          // Particles away from each other. Take particle i's position, and
          // subtract particle j's position. Then, normalize the delta. This is
          // the direction where the force acts. The quantity F/r normalizes for
          // you while making the force calculation cheaper.
          //
          // Also note that by Newton's third law, the force will be duplicated.
          // From particle j's point of view, particle i is pushed with double the
          // acceleration derived from -dU/dr. This can be explained by the
          // following:
          //
          // Each particle has its own potential energy, independent from the
          // other interacting particle. The PES curves happen to look the same,
          // but they're two separate potential energies for two separate
          // particles. -dU/dr for particle i is coming out of U_i. -dU/dr for
          // particle j is coming out of U_j. If they came out of the same
          // potential energy function, then you'd be correct that the potential
          // energy delta is being duplicated.
          //
          // nm: 1e-9
          let x_delta = x_i - x_j
          let y_delta = y_i - y_j
          let z_delta = z_i - z_j
          
          // nm^-2: 1e18
          var r2 = x_delta * x_delta
          r2.addProduct(y_delta, y_delta)
          r2.addProduct(z_delta, z_delta)
          r2 = simd_fast_recip(r2)
          
          // Check whether j == index. If so, skip the iteration.
          let upcasted = Block.MaskStorage(truncatingIfNeeded: indexInList)
          r2.replace(with: .zero, where: upcasted .== Real.SIMDMaskScalar(j))
          
          // nm^-6: 1e54
          let r6 = r2 * r2 * r2
          
          // nm^-12: 1e108
          let r12 = r6 * r6
          
          // aJ: 1e-18
          var temp1 = c6 * r6
          if queryEnergy {
            u_next.addProduct(temp1, 2)
            u_next.addProduct(c12, r12)
          }
          temp1.addProduct(c12, r12)
          
          // aJ * nm^-2: 1e0
          // 10^-18 * (kg * m^2/s^2) * nm^-2
          // 10^-18 * (kg/s^2) * 10^18
          let temp2 = temp1 * r2
          
          // aJ/nm: 1e-9
          // nano-Newtons
          f_x_next.addProduct(temp2, x_delta)
          f_y_next.addProduct(temp2, y_delta)
          f_z_next.addProduct(temp2, z_delta)
        }
      }
    }
    
    if numCores == 1 {
      execute(coreID: 0)
    } else {
      // Profile CPU synchronization latency.
      let loggingLatency = false
      let start = loggingLatency ? CACurrentMediaTime() : 0
      DispatchQueue.concurrentPerform(iterations: numCores, execute: execute)
      let end = loggingLatency ? CACurrentMediaTime() : 0
      if loggingLatency {
        print("\((end - start) / 1e-6) us")
      }
    }
  }
  
  // Creates a command buffer and encodes the specified number of steps.
  private mutating func _evolveGPU(timeStep h: Double, steps: Int) {
    self.updateArguments(timeStep: h)
    
    self.commandBuffer = commandQueue.makeCommandBuffer()!
    self.encoder = commandBuffer!.makeComputeCommandEncoder()!
    
    encoder!.setBuffer(argumentsBuffer, offset: 0, index: 0)
    encoder!.setBuffer(x.gpuBuffer!, offset: x.gpuOffset, index: 1)
    encoder!.setBuffer(y.gpuBuffer!, offset: y.gpuOffset, index: 2)
    encoder!.setBuffer(z.gpuBuffer!, offset: z.gpuOffset, index: 3)
    encoder!.setBuffer(v_x.gpuBuffer!, offset: v_x.gpuOffset, index: 4)
    encoder!.setBuffer(v_y.gpuBuffer!, offset: v_y.gpuOffset, index: 5)
    encoder!.setBuffer(v_z.gpuBuffer!, offset: v_z.gpuOffset, index: 6)
    encoder!.setBuffer(a_x.gpuBuffer!, offset: a_x.gpuOffset, index: 7)
    encoder!.setBuffer(a_y.gpuBuffer!, offset: a_y.gpuOffset, index: 8)
    encoder!.setBuffer(a_z.gpuBuffer!, offset: a_z.gpuOffset, index: 9)
    encoder!.setBuffer(element.gpuBuffer!, offset: element.gpuOffset, index: 10)
    
    let mass = self.masses.elementsGPU!
    encoder!.setBuffer(mass.gpuBuffer!, offset: mass.gpuOffset, index: 11)
    let params = self.ljParameters
    encoder!.setBuffer(params.gpuBuffer!, offset: params.gpuOffset, index: 12)
    
    precondition(atoms < UInt16.max, "Too many atoms.")
    for _ in 0..<steps {
      let atomsSize = MTLSizeMake(atoms, 1, 1)
      let tgSize = MTLSizeMake(256, 1, 1)
      encoder!.setComputePipelineState(updatePositionsPipeline)
      encoder!.dispatchThreads(atomsSize, threadsPerThreadgroup: tgSize)
      
      let quadsSize = MTLSizeMake(4 * atoms, 1, 1)
      encoder!.setComputePipelineState(updateVelocitiesPipeline)
      encoder!.dispatchThreads(quadsSize, threadsPerThreadgroup: tgSize)
    }
    
    encoder!.endEncoding()
    encoder = nil
    commandBuffer!.commit()
  }
}

// When testing speed, do not graph the results.
let testingSpeed: Bool = false

// Whether to use GPU acceleration. This only helps at around 100 atoms.
let useGPU: Bool = false

if testingSpeed {
  // MARK: - Testing Speed
  
  let gridWidth: SIMD3<Int> = .init(4, 4, 4)
  let femtoseconds: Double = 4
  let timeStep: Double = 1e-15 * femtoseconds
  let maxTimeSteps: Int = Int(20_000 / femtoseconds)
  let gridSpacing: Double = 1.0 // nm
  let trials: Int = 3 // run multiple trials to warm the caches
  
  let atoms: Int = gridWidth.x * gridWidth.y * gridWidth.z
  var system = System(atoms: atoms, ljParameters: parametersMatrix)
  
  // 2-3 becomes 1, 4-5 becomes 2, 6-7 becomes 3, etc.
  let axisExtentsMinus = gridWidth / 2
  var axisExtentsPlus = axisExtentsMinus
  
  // Skip an extra spacing for even-sized grids.
  for axisID in 0..<3 {
    if gridWidth[axisID].isMultiple(of: 2) {
      axisExtentsPlus[axisID] -= 1
    }
  }
  
  var atomIndex: Int = 0
  for x in -axisExtentsMinus.x...axisExtentsPlus.x {
    for y in -axisExtentsMinus.y...axisExtentsPlus.y {
      for z in -axisExtentsMinus.z...axisExtentsPlus.z {
        precondition(
          atomIndex >= 0 && atomIndex < atoms, "Unexpected atom index.")
        defer {
          atomIndex += 1
        }
        
        system.setType(10, index: atomIndex)
        
        let xyzDeltaReal = SIMD3<Real>(SIMD3(x, y, z))
        system.setPosition(Real(gridSpacing) * xyzDeltaReal, index: atomIndex)
        system.setVelocity(.zero, index: atomIndex)
      }
    }
  }
  system.initializeAccelerations()
  if useGPU {
    system.initializeGPU()
  }
  
  let debugAtoms = false
  if debugAtoms {
    print()
    print(system.description(atomRange: 0..<system.atoms))
  }
  
  // Run `maxTimeSteps` multiple times in a loop.
  var minIRLTime: Double = .infinity
  for _ in 0..<trials {
    let start = CACurrentMediaTime()
    system.evolve(timeStep: timeStep, steps: maxTimeSteps)
    
    if debugAtoms {
      print()
      print(system.description(atomRange: 0..<system.atoms))
    }
    
    let end = CACurrentMediaTime()
    minIRLTime = min(minIRLTime, end - start)
  }
  
  let simulatedTime_ns = Double(maxTimeSteps) * timeStep / 1e-9
  print()
  print("Atoms: \(system.atoms)")
  print("IRL time: \(Float(minIRLTime)) s")
  print("Simulated time: \(Float(simulatedTime_ns)) ns")
  print("ns/day: \(Float(simulatedTime_ns * 86400 / minIRLTime))")
} else {
  // MARK: - Graphing
  
  enum GraphType: CaseIterable {
    case trajectory1 // 1st half of the trajectory; before the collision
    case trajectory2 // 2nd half of the trajectory; after the collision
    case energy
    
    // Label of the Matplotlib graph.
    var title: String {
      switch self {
      case .trajectory1: return "Trajectory, Before Collision"
      case .trajectory2: return "Trajectory, After Collision"
      case .energy: return "Energy"
      }
    }
    
    // Label of the Matplotlib x-axis.
    var xLabel: String {
      switch self {
      case .trajectory1, .trajectory2: return "x position (nm)"
      case .energy: return "time (ps)"
      }
    }
    
    // Label of the Matplotlib y-axis.
    var yLabel: String {
      switch self {
      case .trajectory1, .trajectory2: return "y position (nm)"
      case .energy: return "energy (zJ)"
      }
    }
  }
  
  let debuggingLJ: Bool = false // for debugging the LJ potential
  
  let (fig, axes) = plt.subplots(3, 3).tuple2
  let dpi = Double(fig.dpi)!
  let fig_width: Double = 2000
  let fig_height: Double = 2000
  fig.set_size_inches(fig_width / dpi, fig_height / dpi)
  fig.subplots_adjust(
    left: 0.09, right: 0.92,
    top: 0.94, bottom: 0.08,
    wspace: 0.15, hspace: 0.25)
  
  var counter: Int = 1
  for simulationID in 0..<3 {
    // Neon or Argon
    let firstAtomsType: UInt8 = simulationID <= 1 ? 10 : 18
    let secondAtomsType: UInt8 = simulationID <= 0 ? 10 : 18
    let systemAtoms = debuggingLJ ? 2 : 4
    var system = System(atoms: systemAtoms, ljParameters: parametersMatrix)
    
    // Set atom types.
    let rangeFirst = debuggingLJ ? 0..<1 : 0..<2
    for i in rangeFirst {
      system.setType(firstAtomsType, index: i)
    }
    let rangeSecond = debuggingLJ ? 1..<2 : 2..<4
    for i in rangeSecond {
      system.setType(secondAtomsType, index: i)
    }
    
    // START - ad-hoc place for simulation hyperparameters
    
    let femtoseconds: Double = 1
    let timeStep: Double = 1e-15 * femtoseconds
    let stepsPerSample: Int = 10
    let maxTimeSteps: Int = Int(20_000 / femtoseconds) // 20_000
    var positions1: [[SIMD3<Real>]] = Array(repeating: [], count: system.atoms)
    var velocities1: [[SIMD3<Real>]] = Array(repeating: [], count: system.atoms)
    var kineticEnergies: [Double] = []
    var potentialEnergies: [Double] = []
    
    let p_magnitude: Real = 0.3
    let v_magnitude: Real = 0.03
    
    // END
    
    if debuggingLJ {
      system.setPosition(SIMD3(-p_magnitude, 0, 0), index: 0)
      system.setPosition(SIMD3( p_magnitude, 0, 0), index: 1)
      system.setVelocity(SIMD3( v_magnitude, 0, 0), index: 0)
      system.setVelocity(SIMD3(-v_magnitude, 0, 0), index: 1)
    } else {
      // Make the atoms move in a spiral.
      let angle = Real.pi * 0.04 // counterclockwise
      let rotationMatrixCol1 = SIMD3<Real>( cos(angle), sin(angle), 0)
      let rotationMatrixCol2 = SIMD3<Real>(-sin(angle), cos(angle), 0)
      let rotationMatrixCol3 = SIMD3<Real>(0, 0, 1)
      
      // Set atom positions and velocities.
      // Avoiding any motion in the Z plane for now.
      for i in 0..<4 {
        let signX: Real = (i % 2 == 0) ? 1 : -1
        let signY: Real = (i / 2 == 0) ? 1 : -1
        
        let distanceScale: Real = p_magnitude // nm
        let position = SIMD3<Real>(
          distanceScale * signX, distanceScale * signY, 0)
        
        let velocityScale: Real = -v_magnitude // nm/ps
        var velocity = SIMD3<Real>(
          velocityScale * signX, velocityScale * signY, 0)
        
        // Rotate the velocity.
        velocity =
        rotationMatrixCol1 * velocity.x +
        rotationMatrixCol2 * velocity.y +
        rotationMatrixCol3 * velocity.z
        
        system.setPosition(position, index: i)
        system.setVelocity(velocity, index: i)
      }
    }
    system.initializeAccelerations()
    if useGPU {
      system.initializeGPU()
    }
    
    // Minimum average distance over all the points. Gathering this metric takes
    // O(n^2) computations.
    var minAverageDistance: Real = .infinity
    var minDistanceSampleID: Int = -1
    var tempPositions: [SIMD3<Real>] = Array(
      repeating: .zero, count: system.atoms)
    let start = CACurrentMediaTime()
    for sampleID in 0..<maxTimeSteps / stepsPerSample {
      system.evolve(timeStep: timeStep, steps: stepsPerSample)
      
      for i in 0..<system.atoms {
        let position = system.getPosition(index: i)
        tempPositions[i] = position
        positions1[i].append(position)
        velocities1[i].append(system.getVelocity(index: i))
      }
      
      var averageDistance: Real = 0
      for i in 0..<system.atoms {
        for j in 0..<system.atoms where i != j {
          averageDistance += distance(tempPositions[i], tempPositions[j])
        }
      }
      averageDistance /= Real(system.atoms * system.atoms - system.atoms)
      if averageDistance < minAverageDistance {
        minAverageDistance = averageDistance
        minDistanceSampleID = sampleID
      }
      
      let (kinetic, potential) = system.getEnergy()
      kineticEnergies.append(kinetic)
      potentialEnergies.append(potential)
    }
    let end = CACurrentMediaTime()
    let irlTime = end - start
    let simulatedTime_ns = Double(maxTimeSteps) * timeStep / 1e-9
    print()
    print("IRL time: \(Float(irlTime)) s")
    print("Simulated time: \(Float(simulatedTime_ns)) ns")
    print("ns/day: \(Float(simulatedTime_ns * 86400 / irlTime))")
    
    
    // Split the trajectory at the collision point.
    var positions2: [[SIMD3<Real>]]
    
    if minDistanceSampleID < 0 || minDistanceSampleID == 0 {
      positions2 = positions1
    } else {
      func splice(array: inout [[SIMD3<Real>]]) -> [[SIMD3<Real>]] {
        var output: [[SIMD3<Real>]] = Array(repeating: [], count: system.atoms)
        let sampleRange = minDistanceSampleID..<maxTimeSteps / stepsPerSample
        for i in 0..<system.atoms {
          output[i] = Array(array[i][sampleRange])
          array[i].removeSubrange(sampleRange)
        }
        return output
      }
      
      positions2 = splice(array: &positions1)
    }
    
    for (subplotIndex, subplotType) in GraphType.allCases.enumerated() {
      let ax = axes[simulationID, subplotIndex]
      
      switch subplotType {
      case .trajectory1, .trajectory2:
        var neonAtomID: Int = 1
        var argonAtomID: Int = 1
        for atomIndexInList in 0..<system.atoms {
          let element = system.getType(index: atomIndexInList)
          let isArgon = element == 18
          precondition(isArgon || element == 10, "Unsupported element.")
          defer {
            if isArgon {
              argonAtomID += 1
            } else {
              neonAtomID += 1
            }
          }
          
          var label: String
          if isArgon {
            label = "Ar \(argonAtomID)"
          } else {
            label = "Ne \(neonAtomID)"
          }
          
          let isTrajectory1 = (subplotType == .trajectory1)
          let positions = isTrajectory1 ? positions1 : positions2
          
          var positionsX: [Real] = []
          var positionsY: [Real] = []
          for sampleID in positions[0].indices {
            let position = positions[atomIndexInList][sampleID]
            positionsX.append(position.x)
            positionsY.append(position.y)
          }
          ax.plot(positionsX, positionsY, label: label)
        }
        
        // There's no other way to force both plots to have the same scale.
        var forcedBound: Real
        if simulationID == 0 {
          forcedBound = 0.6 // 0.6, 2.5
        } else if simulationID == 1 {
          forcedBound = 1.1 // 1.1, 2.5
        } else {
          forcedBound = 0.6 // 0.6, 8.0
        }
        ax.plot(forcedBound, forcedBound)
        ax.plot(-forcedBound, -forcedBound)
        
        if subplotType == .trajectory1 {
          ax.legend(loc: "upper right")
          ax.axis("equal")
        } else {
          ax.axis("equal")
        }
        
      case .energy:
        let kinetic = kineticEnergies.map { $0 / 1e-21 }
        let potential = potentialEnergies.map { $0 / 1e-21 }
        let total = zip(kinetic, potential).map(+)
        
        let timeMultiplier = timeStep * Double(stepsPerSample) / 1e-12
        let times = total.indices.map { Double($0 + 1) * timeMultiplier }
        ax.plot(times, kinetic, label: "kinetic")
        ax.plot(times, potential, label: "potential")
        ax.plot(times, total, label: "total")
        
        // Ensure the origin is graphed.
        ax.plot([Double.zero], [Double.zero])
        
        if simulationID == 0 {
          ax.legend()
        }
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
      }
      
      if simulationID == 0 {
        ax.set_title(subplotType.title)
      }
      if simulationID == 2 {
        ax.set_xlabel(subplotType.xLabel)
      }
      if subplotType == .trajectory2 {
        ax.yaxis.set_tick_params(labelleft: false)
      } else {
        ax.set_ylabel(subplotType.yLabel)
      }
    }
  }
  
  plt.show()
}

// MARK: - GPU Support

struct DoubleSingle {
  var hi: Float
  var lo: Float
  
  init(_ value: Double) {
    self.hi = Float(value)
    self.lo = Float(value - Double(hi))
  }
}

func createGPUEvolveSrc() -> String { """
#include <metal_stdlib>
using namespace metal;

struct DoubleSingle {
  float hi;
  float lo;
};

typedef struct DoubleSingle DS;

DS DS_init(float hi, float lo) {
  DS output;
  output.hi = hi;
  output.lo = lo;
  return output;
}

DS DS_negated(DS input) {
  return DS_init(-input.hi, -input.lo);
}

DS DS_halved(DS input) {
  return DS_init(input.hi / 2, input.lo / 2);
}

DS DS_normalized(DS input) {
  float s = input.hi + input.lo;
  float e = input.lo - (s - input.hi);
  return DS_init(s, e);
}

DS DS_init_adding(float lhs, float rhs) {
  float s = lhs + rhs;
  float v = s - lhs;
  float e = (lhs - (s - v)) + (rhs - v);
  return DS_init(s, e);
}

DS DS_init_multiplying(float lhs, float rhs) {
  float hi = lhs * rhs;
  float lo = fma(lhs, rhs, -hi);
  return DS_init(hi, lo);
}

// Addition and Subtraction

DS DS_add(DS lhs, DS rhs) {
  DS s = DS_init_adding(lhs.hi, rhs.hi);
  s.lo += lhs.lo + rhs.lo;
  return DS_normalized(s);
}

DS DS_add_float_rhs(DS lhs, float rhs) {
  DS s = DS_init_adding(lhs.hi, rhs);
  s.lo += lhs.lo;
  return DS_normalized(s);
}

DS DS_add_float_lhs(float lhs, DS rhs) {
  return DS_add_float_rhs(rhs, lhs);
}

DS DS_sub(DS lhs, DS rhs) {
  return DS_add(lhs, DS_negated(rhs));
}

DS DS_sub_float_rhs(DS lhs, float rhs) {
  return DS_add_float_rhs(lhs, -rhs);
}

DS DS_sub_float_lhs(float lhs, DS rhs) {
  return DS_add_float_lhs(lhs, DS_negated(rhs));
}

// Multiplication and Division

DS DS_mul(DS lhs, DS rhs) {
  DS p = DS_init_multiplying(lhs.hi, rhs.hi);
  p.lo = fma(lhs.hi, rhs.lo, p.lo);
  p.lo = fma(lhs.lo, rhs.hi, p.lo);
  return DS_normalized(p);
}

DS DS_mul_float_rhs(DS lhs, float rhs) {
  DS p = DS_init_multiplying(lhs.hi, rhs);
  p.lo = fma(lhs.lo, rhs, p.lo);
  return DS_normalized(p);
}

DS DS_mul_float_lhs(float lhs, DS rhs) {
  return DS_mul_float_rhs(rhs, lhs);
}

DS DS_div(DS lhs, DS rhs) {
  float xn = fast::divide(1, rhs.hi);
  float yn = lhs.hi * xn;
  DS ayn = DS_mul_float_rhs(rhs, yn);
  float diff = DS_sub(lhs, ayn).hi;
  DS prod = DS_init_multiplying(xn, diff);
  DS q = DS_add_float_lhs(yn, prod);
  
  // Don't handle infinity case because any `INF` will cause undefined
  // behavior in other code anyway.
  return q;
}

DS DS_recip(DS input) {
  float xn = fast::divide(1, input.hi);
  DS ayn = DS_mul_float_rhs(input, xn);
  float diff = DS_sub_float_lhs(1, ayn).hi;
  DS prod = DS_init_multiplying(xn, diff);
  DS q = DS_add_float_lhs(xn, prod);
  
  // Don't handle infinity case because any `INF` will cause undefined
  // behavior in other code anyway.
  return q;
}

// Kernels

// FP64 numbers pre-computed to FP32 on the CPU.
typedef struct {
  DS h;
  DS billionth;
  DS thousandth;
} Arguments;

kernel void updatePositions(
  // Generate by casting the float from fp64 -> fp32 on CPU, then casting back
  // to fp64 and measuring the difference. Do the same when converting masses
  // into a GPU-compatible format.
  constant Arguments &args [[buffer(0)]],

  // State variables.
  device float *x [[buffer(1)]],
  device float *y [[buffer(2)]],
  device float *z [[buffer(3)]],
  device float *v_x [[buffer(4)]],
  device float *v_y [[buffer(5)]],
  device float *v_z [[buffer(6)]],
  device float *a_x [[buffer(7)]],
  device float *a_y [[buffer(8)]],
  device float *a_z [[buffer(9)]],

  // Loop index.
  ushort i [[thread_position_in_grid]]
) {
  // nm: 1e-9
  float x_curr = x[i];
  float y_curr = y[i];
  float z_curr = z[i];

  // nm/ps: 1e3
  float v_x_curr = v_x[i];
  float v_y_curr = v_y[i];
  float v_z_curr = v_z[i];

  // nm/ps/s: 1e3
  float a_x_curr = a_x[i];
  float a_y_curr = a_y[i];
  float a_z_curr = a_z[i];

  // m: 1e0
  DS nm___m = args.billionth;
  DS x_next = DS_mul_float_rhs(nm___m, x_curr);
  DS y_next = DS_mul_float_rhs(nm___m, y_curr);
  DS z_next = DS_mul_float_rhs(nm___m, z_curr);

  float nm_ps___m_s = 1e3;
  DS h = args.h;
  DS v_scale = DS_mul_float_lhs(nm_ps___m_s, h);
  x_next = DS_add(x_next, DS_mul_float_rhs(v_scale, v_x_curr));
  y_next = DS_add(y_next, DS_mul_float_rhs(v_scale, v_y_curr));
  z_next = DS_add(z_next, DS_mul_float_rhs(v_scale, v_z_curr));

  float nm_ps_s___m_s2 = 1e3;
  DS multiplier = DS_mul_float_lhs(0.5, DS_mul(h, h));
  DS a_scale = DS_mul_float_lhs(nm_ps_s___m_s2, multiplier);
  x_next = DS_add(x_next, DS_mul_float_rhs(a_scale, a_x_curr));
  y_next = DS_add(y_next, DS_mul_float_rhs(a_scale, a_y_curr));
  z_next = DS_add(z_next, DS_mul_float_rhs(a_scale, a_z_curr));

  // nm: 1e-9
  float m___nm = 1e9;
  float x_i = DS_mul_float_rhs(x_next, m___nm).hi;
  float y_i = DS_mul_float_rhs(y_next, m___nm).hi;
  float z_i = DS_mul_float_rhs(z_next, m___nm).hi;

  // Update the positions *before* calculating forces.
  // You never query energy or update acceleration on the GPU.
  x[i] = x_i;
  y[i] = y_i;
  z[i] = z_i;
}

struct LJPotentialParameters {
  union {
    float2 _data;
    struct {
      float c12;
      float c6;
    };
  };
};

template <uint width, uint height>
struct LJPotentialParametersMatrix {
  device LJPotentialParameters* elements;

  uint index(uint row, uint column) {
    return width * row + column;
  }

  LJPotentialParameters getElement(uint index) {
    return elements[index];
  }
};

kernel void updateVelocities(
  constant Arguments &args [[buffer(0)]],

  // State variables.
  device float *x [[buffer(1)]],
  device float *y [[buffer(2)]],
  device float *z [[buffer(3)]],
  device float *v_x [[buffer(4)]],
  device float *v_y [[buffer(5)]],
  device float *v_z [[buffer(6)]],
  device float *a_x [[buffer(7)]],
  device float *a_y [[buffer(8)]],
  device float *a_z [[buffer(9)]],
  device uchar *element [[buffer(10)]],

  device DS *massesInv [[buffer(11)]],
  device LJPotentialParameters *paramsRaw [[buffer(12)]],

  // Loop index.
  uint global_id [[thread_position_in_grid]],
  ushort quad_lane [[thread_index_in_quadgroup]],
  uint quad_threads [[threads_per_grid]]
) {
  ushort i = global_id / 4;
  float x_i = x[i];
  float y_i = y[i];
  float z_i = z[i];
  ushort id = ushort(element[i]);
  ushort atoms(uint(quad_threads / 4));

  float f_x_next = 0;
  float f_y_next = 0;
  float f_z_next = 0;

  LJPotentialParametersMatrix<119, 119> params { paramsRaw };
  auto paramsBase = params.elements + params.index(id, 0);

  // Calculate nonbonded forces.
  constexpr ushort group_size = 4;
  ushort j = group_size * quad_lane;
  ushort j_max = ~(4 - 1) & atoms;
  for (; j < j_max; j += group_size * 4) {
    uchar4 id_j = *(device uchar4*)(element + j);
    auto c12c6_0 = paramsBase[id_j[0]];
    auto c12c6_1 = paramsBase[id_j[1]];
    auto c12c6_2 = paramsBase[id_j[2]];
    auto c12c6_3 = paramsBase[id_j[3]];
    float4 c12(c12c6_0.c12, c12c6_1.c12, c12c6_2.c12, c12c6_3.c12);
    float4 c6 (c12c6_0.c6,  c12c6_1.c6,  c12c6_2.c6,  c12c6_3.c6);

    float4 x_j = *(device float4*)(x + j);
    float4 y_j = *(device float4*)(y + j);
    float4 z_j = *(device float4*)(z + j);

    float4 x_delta = x_i - x_j;
    float4 y_delta = y_i - y_j;
    float4 z_delta = z_i - z_j;

    float4 r2 = x_delta * x_delta;
    r2 = fma(y_delta, y_delta, r2);
    r2 = fma(z_delta, z_delta, r2);
    r2 = fast::divide(1, r2);

    ushort4 j_vector(j, j + 1, j + 2, j + 3);
    r2 = select(r2, float4(0), i == j_vector);

    float4 r6 = r2 * r2 * r2;
    float4 r12 = r6 * r6;

    float4 temp1 = c6 * r6;
    temp1 = fma(c12, r12, temp1);
    float4 temp2 = temp1 * r2;

    f_x_next = fma(temp2[0], x_delta[0], f_x_next);
    f_y_next = fma(temp2[0], y_delta[0], f_y_next);
    f_z_next = fma(temp2[0], z_delta[0], f_z_next);
    f_x_next = fma(temp2[1], x_delta[1], f_x_next);
    f_y_next = fma(temp2[1], y_delta[1], f_y_next);
    f_z_next = fma(temp2[1], z_delta[1], f_z_next);
    f_x_next = fma(temp2[2], x_delta[2], f_x_next);
    f_y_next = fma(temp2[2], y_delta[2], f_y_next);
    f_z_next = fma(temp2[2], z_delta[2], f_z_next);
    f_x_next = fma(temp2[3], x_delta[3], f_x_next);
    f_y_next = fma(temp2[3], y_delta[3], f_y_next);
    f_z_next = fma(temp2[3], z_delta[3], f_z_next);
  };

  f_x_next = quad_sum(f_x_next);
  f_y_next = quad_sum(f_y_next);
  f_z_next = quad_sum(f_z_next);
  if (!quad_is_first()) {
    return;
  }

  // Properly handle the border of the matrix.
  for (ushort j = j_max; j < atoms; j += 1) {
    ushort id_j = ushort(element[j]);
    auto c12c6 = paramsBase[id_j];
    float c12 = c12c6.c12;
    float c6 = c12c6.c6;

    float x_j = x[j];
    float y_j = y[j];
    float z_j = z[j];

    // nm: 1e-9
    float x_delta = x_i - x_j;
    float y_delta = y_i - y_j;
    float z_delta = z_i - z_j;

    // nm^-2: 1e18
    float r2 = x_delta * x_delta;
    r2 = fma(y_delta, y_delta, r2);
    r2 = fma(z_delta, z_delta, r2);
    r2 = fast::divide(1, r2);

    // Check whether j == index. If so, skip the iteration.
    r2 = (i == j) ? 0 : r2;

    // nm^-6: 1e54
    float r6 = r2 * r2 * r2;

    // nm^-12: 1e108
    float r12 = r6 * r6;

    // aJ: 1e-18
    float temp1 = c6 * r6;
    temp1 = fma(c12, r12, temp1);

    // aJ * nm^-2: 1e0
    // 10^-18 * (kg * m^2/s^2) * nm^-2
    // 10^-18 * (kg/s^2) * 10^18
    float temp2 = temp1 * r2;

    // aJ/nm: 1e-9
    // nano-Newtons
    f_x_next = fma(temp2, x_delta, f_x_next);
    f_y_next = fma(temp2, y_delta, f_y_next);
    f_z_next = fma(temp2, z_delta, f_z_next);
  }

  // Prefetch the accelerations.
  float a_x_curr = a_x[i];
  float a_y_curr = a_y[i];
  float a_z_curr = a_z[i];

  // kg^-1: 1e0
  DS inverseMass = massesInv[id];

  // m/s^2: 1e0
  float nm_s2___m_s2 = 1e-9;
  DS a_scale = DS_mul_float_lhs(nm_s2___m_s2, inverseMass);
  DS a_x_next = DS_mul_float_rhs(a_scale, f_x_next);
  DS a_y_next = DS_mul_float_rhs(a_scale, f_y_next);
  DS a_z_next = DS_mul_float_rhs(a_scale, f_z_next);

  // Prefetch the velocities.
  float v_x_curr = v_x[i];
  float v_y_curr = v_y[i];
  float v_z_curr = v_z[i];

  DS thousandth = args.thousandth;
  
  // nm/ps/s: 1e3
  DS m_s2___nm_ps_s = thousandth;
  a_x[i] = DS_mul(a_x_next, m_s2___nm_ps_s).hi;
  a_y[i] = DS_mul(a_y_next, m_s2___nm_ps_s).hi;
  a_z[i] = DS_mul(a_z_next, m_s2___nm_ps_s).hi;

  // Average the accelerations.
  // The current value is twice the average acceleration. A later line of
  // code will correct for the factor of 2.
  float nm_ps_s___m_s2 = 1e3;
  DS a_x_sum = DS_add(a_x_next, DS_init_multiplying(nm_ps_s___m_s2, a_x_curr));
  DS a_y_sum = DS_add(a_y_next, DS_init_multiplying(nm_ps_s___m_s2, a_y_curr));
  DS a_z_sum = DS_add(a_z_next, DS_init_multiplying(nm_ps_s___m_s2, a_z_curr));

  // nm/ps: 1e3
  float nm_ps___m_s = 1e3;
  DS v_x_next = DS_init_multiplying(nm_ps___m_s, v_x_curr);
  DS v_y_next = DS_init_multiplying(nm_ps___m_s, v_y_curr);
  DS v_z_next = DS_init_multiplying(nm_ps___m_s, v_z_curr);

  DS multiplier = DS_halved(args.h);
  v_x_next = DS_add(v_x_next, DS_mul(multiplier, a_x_sum));
  v_y_next = DS_add(v_y_next, DS_mul(multiplier, a_y_sum));
  v_z_next = DS_add(v_z_next, DS_mul(multiplier, a_z_sum));

  // nm/ps: 1e3
  DS m_s___nm_ps = thousandth;
  v_x[i] = DS_mul(v_x_next, m_s___nm_ps).hi;
  v_y[i] = DS_mul(v_y_next, m_s___nm_ps).hi;
  v_z[i] = DS_mul(v_z_next, m_s___nm_ps).hi;
}
"""
}
