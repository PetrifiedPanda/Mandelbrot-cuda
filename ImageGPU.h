#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "Image.h"

class ImageGPU {
    friend class Image;
    size_t xDim_, yDim_, channels_;
    uint8_t* bytes_;

   public:
    ImageGPU();
    ImageGPU(size_t xDim, size_t yDim, size_t channels = 3);
    ImageGPU(const ImageGPU& other);
    ImageGPU(ImageGPU&& other);
    ~ImageGPU();

    size_t xDim() const;
    size_t yDim() const;
    size_t channels() const;

    const uint8_t* bytes() const;

    Image toHost() const;

    class Ref {
        friend class ImageGPU;
        size_t xDim_, yDim_, channels_;
        uint8_t* bytes_;

        Ref(size_t xDim, size_t yDim, size_t channels, uint8_t* bytes);
       public:
        __device__ uint8_t& operator()(size_t x, size_t y, size_t channelIndex);
        __device__ const uint8_t& operator()(size_t x, size_t y, size_t channelIndex) const;

        __device__ size_t xDim() const;
        __device__ size_t yDim() const;
        __device__ size_t channels() const;
    };

    Ref getRef();
};