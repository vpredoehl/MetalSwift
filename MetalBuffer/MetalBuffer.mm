//
//  Untitled.swift
//  MetalSwift
//
//  Created by Vincent Predoehl on 3/7/26.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalBuffer.hpp"
#include <utility>

namespace MetalSwift
{

static MTLResourceOptions ToResourceOptions(StorageMode mode) {
    switch (mode) {
        case StorageMode::Shared:
            return MTLResourceStorageModeShared;
        case StorageMode::Private:
            return MTLResourceStorageModePrivate;
        default:
            return MTLResourceStorageModeShared;
    }
}

class MetalBuffer::Impl
{
public:
    id<MTLDevice> device = nil;
    id<MTLBuffer> buffer = nil;
    std::size_t size = 0;
    StorageMode mode = StorageMode::Shared;

    Impl() {
        device = MTLCreateSystemDefaultDevice();
    }

    bool Allocate(std::size_t newSize, StorageMode newMode) {
        Release();

        if (!device || newSize == 0) {
            return false;
        }

        mode = newMode;
        size = newSize;

        buffer = [device newBufferWithLength:newSize
                                     options:ToResourceOptions(newMode)];

        if (!buffer) {
            size = 0;
            return false;
        }

        return true;
    }

    void Release() {
        buffer = nil;
        size = 0;
    }

    void* CPUContents() {
        if (!buffer) return nullptr;
        if (mode == StorageMode::Private) return nullptr;
        return [buffer contents];
    }

    const void* CPUContents() const {
        if (!buffer) return nullptr;
        if (mode == StorageMode::Private) return nullptr;
        return [buffer contents];
    }
};

MetalBuffer::MetalBuffer()
: impl_(std::make_unique<Impl>()) {}

MetalBuffer::MetalBuffer(std::size_t size, StorageMode mode)
: impl_(std::make_unique<Impl>()) {
    impl_->Allocate(size, mode);
}

MetalBuffer::~MetalBuffer() = default;

MetalBuffer::MetalBuffer(MetalBuffer&&) noexcept = default;
MetalBuffer& MetalBuffer::operator=(MetalBuffer&&) noexcept = default;

bool MetalBuffer::Allocate(std::size_t size, StorageMode mode) {
    return impl_ && impl_->Allocate(size, mode);
}

void MetalBuffer::Release() {
    if (impl_) impl_->Release();
}

void* MetalBuffer::CPUContents() {
    return impl_ ? impl_->CPUContents() : nullptr;
}

const void* MetalBuffer::CPUContents() const {
    return impl_ ? impl_->CPUContents() : nullptr;
}

std::size_t MetalBuffer::Size() const {
    return impl_ ? impl_->size : 0;
}

bool MetalBuffer::IsValid() const {
    return impl_ && impl_->buffer != nil;
}

StorageMode MetalBuffer::Mode() const {
    return impl_ ? impl_->mode : StorageMode::Shared;
}

void* MetalBuffer::NativeHandle() {
    return impl_ ? (__bridge void*)impl_->buffer : nullptr;
}

} // namespace MetalSwift
