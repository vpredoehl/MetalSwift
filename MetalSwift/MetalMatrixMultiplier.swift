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

        constant uint TILE_SIZE = 16;

        struct MatDims {
            uint rowsA;
            uint colsA;
            uint rowsB;
            uint colsB;
        };

        kernel void matmul_tiled(
            device const float* A [[ buffer(0) ]],
            device const float* B [[ buffer(1) ]],
            device float* C [[ buffer(2) ]],
            constant MatDims& dims [[ buffer(3) ]],
            uint2 tgp_id [[ threadgroup_position_in_grid ]],
            uint2 tid [[ thread_position_in_threadgroup ]]) {

            // Compute the row and column of the C element this thread will produce
            const uint row = tgp_id.y * TILE_SIZE + tid.y;
            const uint col = tgp_id.x * TILE_SIZE + tid.x;

            threadgroup float As[TILE_SIZE][TILE_SIZE];
            threadgroup float Bs[TILE_SIZE][TILE_SIZE];

            float sum = 0.0f;

            // Number of tiles to iterate over along the K dimension
            const uint numTiles = (dims.colsA + TILE_SIZE - 1) / TILE_SIZE;

            for (uint t = 0; t < numTiles; ++t) {
                // Global indices for the elements to load into shared tiles
                const uint aCol = t * TILE_SIZE + tid.x;
                const uint bRow = t * TILE_SIZE + tid.y;

                // Load A tile element if in-bounds, else 0
                if (row < dims.rowsA && aCol < dims.colsA) {
                    As[tid.y][tid.x] = A[row * dims.colsA + aCol];
                } else {
                    As[tid.y][tid.x] = 0.0f;
                }

                // Load B tile element if in-bounds, else 0
                if (bRow < dims.rowsB && col < dims.colsB) {
                    Bs[tid.y][tid.x] = B[bRow * dims.colsB + col];
                } else {
                    Bs[tid.y][tid.x] = 0.0f;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Compute partial sum for this tile
                for (uint k = 0; k < TILE_SIZE; ++k) {
                    sum += As[tid.y][k] * Bs[k][tid.x];
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Write result if the output index is in-bounds
            if (row < dims.rowsA && col < dims.colsB) {
                C[row * dims.colsB + col] = sum;
            }
        }
        """

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw MetalMatMulError.libraryBuildFailed(String(describing: error))
        }

        guard let function = library.makeFunction(name: "matmul_tiled") else {
            throw MetalMatMulError.functionNotFound("matmul_tiled")
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

        let tile = 16
        let threadsPerThreadgroup = MTLSize(width: tile, height: tile, depth: 1)
        let tgWidth = (colsC + tile - 1) / tile
        let tgHeight = (rowsC + tile - 1) / tile
        let threadsPerGrid = MTLSize(width: tgWidth * tile, height: tgHeight * tile, depth: 1)

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

