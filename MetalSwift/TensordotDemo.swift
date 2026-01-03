// TensordotDemo.swift
// Exposes a C ABI entry that main.cpp calls: cpp_gpu_tensordot_bwo_demo
// Computes Y = tensordot over last axis of X(B,W,F) and first axis of W(F,O)
// Using float32 Metal compute, with B=7, W=5, F=4, O=4

import Foundation
import Metal


@_cdecl("cpp_gpu_tensordot_bwo_demo")
public func cpp_gpu_tensordot_bwo_demo(_ B: Int32, _ W: Int32, _ F: Int32, _ O: Int32) {
    let b = Int(B), w = Int(W), fdim = Int(F), odim = Int(O)

    guard let device = MTLCreateSystemDefaultDevice() else {
        fputs("[TensorDot] No Metal device available\n", stderr)
        return
    }
    guard let queue = device.makeCommandQueue() else {
        fputs("[TensorDot] Failed to create command queue\n", stderr)
        return
    }

    // Metal kernel source
    let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct Tensor3DArgs {
        uint d0, d1, d2; // shape
        uint s0, s1, s2; // strides (elements)
    };

    struct MatrixArgs {
        uint rows;  // F
        uint cols;  // O
        uint rowStride; // usually cols
        uint colStride; // usually 1
    };

    inline uint idx3(uint i, uint j, uint k, constant Tensor3DArgs& t) {
        return i * t.s0 + j * t.s1 + k * t.s2;
    }

    inline uint idx2(uint r, uint c, constant MatrixArgs& m) {
        return r * m.rowStride + c * m.colStride;
    }

    // Grid: (O, W, B)
    kernel void tensordot_lastaxis_mat(
        device const float* X [[ buffer(0) ]],
        constant Tensor3DArgs& Xargs [[ buffer(1) ]],
        device const float* Wp [[ buffer(2) ]],
        constant MatrixArgs& Wargs [[ buffer(3) ]],
        device float* Y [[ buffer(4) ]],
        constant Tensor3DArgs& Yargs [[ buffer(5) ]],
        uint3 tid [[ thread_position_in_grid ]]
    ) {
        uint o = tid.x; // 0..O-1
        uint t = tid.y; // 0..W-1
        uint b = tid.z; // 0..B-1

        if (b >= Xargs.d0 || t >= Xargs.d1 || o >= Yargs.d2) return;

        float acc = 0.0f;
        for (uint f = 0; f < Xargs.d2; ++f) {
            float x = X[idx3(b, t, f, Xargs)];
            float w = Wp[idx2(f, o, Wargs)];
            acc += x * w;
        }
        Y[idx3(b, t, o, Yargs)] = acc;
    }
    """

    let library: MTLLibrary
    do {
        library = try device.makeLibrary(source: source, options: nil)
    } catch {
        fputs("[TensorDot] Failed to build Metal library: \(error)\n", stderr)
        return
    }
    guard let fn = library.makeFunction(name: "tensordot_lastaxis_mat") else {
        fputs("[TensorDot] Failed to find function 'tensordot_lastaxis_mat'\n", stderr)
        return
    }
    let pipeline: MTLComputePipelineState
    do {
        pipeline = try device.makeComputePipelineState(function: fn)
    } catch {
        fputs("[TensorDot] Failed to create pipeline: \(error)\n", stderr)
        return
    }

    // Allocate buffers (float32)
    let countX = b * w * fdim
    let countW = fdim * odim
    let countY = b * w * odim
    let bytesX = MemoryLayout<Float>.stride * countX
    let bytesW = MemoryLayout<Float>.stride * countW
    let bytesY = MemoryLayout<Float>.stride * countY

    guard let bufX = device.makeBuffer(length: bytesX, options: .storageModeShared),
          let bufW = device.makeBuffer(length: bytesW, options: .storageModeShared),
          let bufY = device.makeBuffer(length: bytesY, options: .storageModeShared) else {
        fputs("[TensorDot] Failed to allocate buffers\n", stderr)
        return
    }

    // Initialize X and W with simple values for demo
    let xPtr = bufX.contents().bindMemory(to: Float.self, capacity: countX)
    for bb in 0..<b {
        for tt in 0..<w {
            for ff in 0..<fdim {
                let idx = bb * (w*fdim) + tt * fdim + ff
                xPtr[idx] = Float(bb + tt + ff) * 0.1 // simple pattern
            }
        }
    }
    let wPtr = bufW.contents().bindMemory(to: Float.self, capacity: countW)
    for ff in 0..<fdim {
        for oo in 0..<odim {
            wPtr[ff*odim + oo] = (ff == oo) ? 1.0 : 0.25 // near-identity for demo
        }
    }

    // Pack args
    struct Tensor3DArgs { var d0, d1, d2: UInt32; var s0, s1, s2: UInt32 }
    struct MatrixArgs { var rows, cols, rowStride, colStride: UInt32 }

    let Xargs = Tensor3DArgs(d0: UInt32(b), d1: UInt32(w), d2: UInt32(fdim),
                             s0: UInt32(w*fdim), s1: UInt32(fdim), s2: 1)
    let Yargs = Tensor3DArgs(d0: UInt32(b), d1: UInt32(w), d2: UInt32(odim),
                             s0: UInt32(w*odim), s1: UInt32(odim), s2: 1)
    let Wargs = MatrixArgs(rows: UInt32(fdim), cols: UInt32(odim), rowStride: UInt32(odim), colStride: 1)

    guard let bufXargs = device.makeBuffer(bytes: [Xargs], length: MemoryLayout<Tensor3DArgs>.stride, options: .storageModeShared),
          let bufYargs = device.makeBuffer(bytes: [Yargs], length: MemoryLayout<Tensor3DArgs>.stride, options: .storageModeShared),
          let bufWargs = device.makeBuffer(bytes: [Wargs], length: MemoryLayout<MatrixArgs>.stride, options: .storageModeShared) else {
        fputs("[TensorDot] Failed to allocate args buffers\n", stderr)
        return
    }

    guard let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else {
        fputs("[TensorDot] Failed to create command buffer/encoder\n", stderr)
        return
    }
    enc.setComputePipelineState(pipeline)

    // Match kernel indices: X(0), Xargs(1), W(2), Wargs(3), Y(4), Yargs(5)
    enc.setBuffer(bufX, offset: 0, index: 0)
    enc.setBuffer(bufXargs, offset: 0, index: 1)
    enc.setBuffer(bufW, offset: 0, index: 2)
    enc.setBuffer(bufWargs, offset: 0, index: 3)
    enc.setBuffer(bufY, offset: 0, index: 4)
    enc.setBuffer(bufYargs, offset: 0, index: 5)

    let grid = MTLSize(width: odim, height: w, depth: b)
    let tg = MTLSize(width: 8, height: 4, depth: 1)
    enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
    enc.endEncoding()

    cmd.commit()
    cmd.waitUntilCompleted()

    if let e = cmd.error {
        fputs("[TensorDot] Command buffer error: \(e.localizedDescription)\n", stderr)
        return
    }

    // Read back and print a small slice
    let yPtr = bufY.contents().bindMemory(to: Float.self, capacity: countY)
    print("TensorDot Y shape (\(b),\(w),\(odim)) sample:")
    for bb in 0..<min(b, 2) { // print first 2 batches
        for tt in 0..<w {
            var row: [String] = []
            for oo in 0..<odim {
                let idx = bb * (w*odim) + tt * odim + oo
                row.append(String(format: "%0.3f", yPtr[idx]))
            }
            print("b=\(bb) t=\(tt): [\(row.joined(separator: ", "))]")
        }
    }
}
