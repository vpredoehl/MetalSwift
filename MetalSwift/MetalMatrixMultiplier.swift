import Foundation
import Metal

enum MetalMatMulError: Error, LocalizedError {
    case deviceUnavailable
    case commandQueueCreationFailed
    case libraryBuildFailed(String)
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case bufferAllocationFailed
    case commandBufferCreationFailed
    case encoderCreationFailed
    case shapeMismatch
    case commandBufferError(String)

    var errorDescription: String? {
        switch self {
        case .deviceUnavailable: return "Failed to create system default Metal device."
        case .commandQueueCreationFailed: return "Failed to create Metal command queue."
        case .libraryBuildFailed(let reason): return "Failed to build Metal library: \(reason)"
        case .functionNotFound(let name): return "Failed to find function '\(name)' in Metal library."
        case .pipelineCreationFailed(let reason): return "Failed to create compute pipeline state: \(reason)"
        case .bufferAllocationFailed: return "Failed to allocate Metal buffers."
        case .commandBufferCreationFailed: return "Failed to create Metal command buffer."
        case .encoderCreationFailed: return "Failed to create Metal compute command encoder."
        case .shapeMismatch: return "Matrix shape mismatch: colsA must equal rowsB."
        case .commandBufferError(let reason): return "Metal command buffer error: \(reason)"
        }
    }
}

struct MetalMatrixMultiplier {
    /// Multiplies matrices A (rowsA x colsA) and B (rowsB x colsB) producing C (rowsA x colsB).
    /// - Parameters:
    ///   - a: Flattened input matrix A.
    ///   - rowsA: Number of rows in matrix A.
    ///   - colsA: Number of columns in matrix A.
    ///   - b: Flattened input matrix B.
    ///   - rowsB: Number of rows in matrix B.
    ///   - colsB: Number of columns in matrix B.
    /// - Returns: Flattened output matrix C.
    /// - Throws: MetalMatMulError on failure.
    static func matmul(a: [Float], rowsA: Int, colsA: Int,
                       b: [Float], rowsB: Int, colsB: Int) throws -> [Float] {
        precondition(rowsA >= 0 && colsA >= 0 && rowsB >= 0 && colsB >= 0, "Matrix dimensions must be non-negative")
        guard colsA == rowsB else { throw MetalMatMulError.shapeMismatch }
        guard a.count == rowsA * colsA, b.count == rowsB * colsB else {
            throw MetalMatMulError.shapeMismatch
        }
        let rowsC = rowsA
        let colsC = colsB
        let countC = rowsC * colsC
        if rowsC == 0 || colsC == 0 { return [] }

        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalMatMulError.deviceUnavailable }
        guard let commandQueue = device.makeCommandQueue() else { throw MetalMatMulError.commandQueueCreationFailed }

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        struct MatDims {
            uint rowsA;
            uint colsA;
            uint rowsB;
            uint colsB;
        };

        kernel void matmul(
            device const float* A [[ buffer(0) ]],
            device const float* B [[ buffer(1) ]],
            device float* C [[ buffer(2) ]],
            constant MatDims& dims [[ buffer(3) ]],
            uint2 gid [[ thread_position_in_grid ]]) {
            uint row = gid.y;
            uint col = gid.x;
            if (row >= dims.rowsA || col >= dims.colsB) return;

            float sum = 0.0f;
            for (uint k = 0; k < dims.colsA; ++k) {
                float a = A[row * dims.colsA + k];
                float b = B[k * dims.colsB + col];
                sum += a * b;
            }
            C[row * dims.colsB + col] = sum;
        }
        """

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw MetalMatMulError.libraryBuildFailed(String(describing: error))
        }

        guard let function = library.makeFunction(name: "matmul") else {
            throw MetalMatMulError.functionNotFound("matmul")
        }

        let pipeline: MTLComputePipelineState
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalMatMulError.pipelineCreationFailed(String(describing: error))
        }

        // Buffers
        let bytesA = MemoryLayout<Float>.stride * a.count
        let bytesB = MemoryLayout<Float>.stride * b.count
        let bytesC = MemoryLayout<Float>.stride * countC
        guard let bufA = device.makeBuffer(length: bytesA, options: .storageModeShared),
              let bufB = device.makeBuffer(length: bytesB, options: .storageModeShared),
              let bufC = device.makeBuffer(length: bytesC, options: .storageModeShared),
              let dimsBuf = device.makeBuffer(length: MemoryLayout<UInt32>.stride * 4, options: .storageModeShared) else {
            throw MetalMatMulError.bufferAllocationFailed
        }

        bufA.contents().copyMemory(from: a, byteCount: bytesA)
        bufB.contents().copyMemory(from: b, byteCount: bytesB)
        var dims = [UInt32(rowsA), UInt32(colsA), UInt32(rowsB), UInt32(colsB)]
        dimsBuf.contents().copyMemory(from: dims, byteCount: MemoryLayout<UInt32>.stride * 4)

        guard let cmd = commandQueue.makeCommandBuffer() else { throw MetalMatMulError.commandBufferCreationFailed }
        guard let enc = cmd.makeComputeCommandEncoder() else { throw MetalMatMulError.encoderCreationFailed }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufC, offset: 0, index: 2)
        enc.setBuffer(dimsBuf, offset: 0, index: 3)

        // One thread per output element
        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        let threadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)
        let threadsPerGrid = MTLSize(width: colsC, height: rowsC, depth: 1)

        enc.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        if let e = cmd.error { throw MetalMatMulError.commandBufferError(e.localizedDescription) }

        var c = Array<Float>(repeating: 0, count: countC)
        bufC.contents().copyMemory(to: &c, byteCount: bytesC)
        return c
    }
}

private extension UnsafeMutableRawPointer {
    func copyMemory<T>(from array: [T], byteCount: Int) {
        array.withUnsafeBytes { src in
            memcpy(self, src.baseAddress!, byteCount)
        }
    }
    func copyMemory<T>(to array: inout [T], byteCount: Int) {
        array.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, self, byteCount)
        }
    }
}
