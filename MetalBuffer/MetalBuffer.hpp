//
//  MetalBuffer.hpp
//  MetalSwift
//
//  Created by Vincent Predoehl on 3/7/26.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace MetalSwift
{
    enum class StorageMode {    Shared, Private };

class MetalBuffer
{
public:
    MetalBuffer();
    MetalBuffer(std::size_t size, StorageMode mode = StorageMode::Shared);
    ~MetalBuffer();

    MetalBuffer(const MetalBuffer&) = delete;
    MetalBuffer& operator=(const MetalBuffer&) = delete;

    MetalBuffer(MetalBuffer&&) noexcept;
    MetalBuffer& operator=(MetalBuffer&&) noexcept;

    bool Allocate(std::size_t size, StorageMode mode = StorageMode::Shared);
    void Release();

    void* CPUContents();                  // nullptr for Private buffers
    const void* CPUContents() const;

    std::size_t Size() const;
    bool IsValid() const;
    StorageMode Mode() const;

    // Opaque native handle access if needed by internal engine code.
    void* NativeHandle();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace MetalSwift

