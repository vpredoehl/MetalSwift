import Foundation

/// Runs a simple C++-backed vector addition via Objective-C++ bridge and prints the result.
func runCppDemo() {
    // Matrix multiplication + addition demo: (2x3) * (3x2) + (2x2)
    let A: [Float] = [
        1, 2, 3,
        4, 5, 6
    ]
    let B: [Float] = [
        7, 8,
        9, 10,
        11, 12
    ]
    let D: [Float] = [
        0.5, 1.5,
        2.5, 3.5
    ]

    let rowsA = 2, colsA = 3
    let rowsB = 3, colsB = 2

    do {
        let C = try MetalMatrixMultiplier.matmulThenAdd(a: A, rowsA: rowsA, colsA: colsA,
                                                         b: B, rowsB: rowsB, colsB: colsB,
                                                         d: D, tileSize: 16)
        for r in 0..<rowsA {
            let row = (0..<colsB).map { String(format: "%g", C[r * colsB + $0]) }.joined(separator: ", ")
            print("C++ (via ObjC++) GPU MatMul+Add C[\(r)]: [\(row)]")
        }
    } catch {
        print("C++ (via ObjC++) GPU MatMul+Add error: \(error.localizedDescription)")
    }
}

import Metal
import Foundation

/// Runs a simple vector addition on the GPU using a C++/Objective-C++ Metal bridge and prints the result.
@_silgen_name("cpp_gpu_vector_add")
func cpp_gpu_vector_add(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>, _ count: Int) -> Int32

func runGpuDemo() {
    let a: [Float] = [1, 2, 3, 4, 5]
    let b: [Float] = [10, 20, 30, 40, 50]
    var out = Array<Float>(repeating: 0, count: a.count)

    let status = a.withUnsafeBufferPointer { aPtr in
        b.withUnsafeBufferPointer { bPtr in
            out.withUnsafeMutableBufferPointer { oPtr in
                cpp_gpu_vector_add(aPtr.baseAddress!, bPtr.baseAddress!, oPtr.baseAddress!, a.count)
            }
        }
    }

    if status == 0 {
        let formatted = out.map { String(format: "%g", $0) }.joined(separator: ", ")
        print("GPU (Metal via C++ bridge) result: [\(formatted)]")
    } else {
        print("GPU compute failed with status: \(status)")
    }
}

