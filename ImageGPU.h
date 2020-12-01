#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "Image.h"

class ImageGPU {
    friend class Image;
    size_t rows_, cols_, channels_;
    unsigned char* bytes_;

   public:
    ImageGPU();
    ImageGPU(size_t rows, size_t cols, size_t channels = 3);
    ImageGPU(const ImageGPU& other);
    ImageGPU(ImageGPU&& other);
    ~ImageGPU();

    size_t rows() const;
    size_t cols() const;
    size_t channels() const;

    Image toHost() const;

    class Ref {
        friend class ImageGPU;
        size_t rows_, cols_, channels_;
        unsigned char* bytes_;

        Ref(size_t rows, size_t cols, size_t channels, unsigned char* bytes);
       public:
        __device__ unsigned char& operator()(size_t row, size_t col, size_t channelIndex);
        __device__ const unsigned char& operator()(size_t row, size_t col, size_t channelIndex) const;

        __device__ size_t rows() const;
        __device__ size_t cols() const;
        __device__ size_t channels() const;
    };

    Ref getRef();
};