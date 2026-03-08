//
//  MetalBufferTest.cpp
//  MetalBuffer
//
//  Created by Vincent Predoehl on 3/7/26.
//

#include "MetalBuffer.hpp"

using namespace MetalSwift;

int main()
{
    MetalBuffer buffer(1024 * sizeof(float));

    float* data = (float*)buffer.CPUContents();

    for (int i = 0; i < 1024; i++)
        data[i] = i;
    
    return 0;
}
