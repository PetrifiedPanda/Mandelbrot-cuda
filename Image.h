#pragma once

#include <cstddef>
#include <string>

class ImageGPU;

class Image {
    friend class ImageGPU;
    size_t xDim_, yDim_, channels_;
    unsigned char* bytes_;

   public:
    Image();
    Image(size_t xDim, size_t yDim, size_t channels = 3);
    Image(const Image& other);
    Image(Image&& other);
    ~Image();

    size_t xDim() const;
    size_t yDim() const;
    size_t channels() const;

    ImageGPU toDevice();

    unsigned char& operator()(size_t x, size_t y, size_t channel);
    const unsigned char& operator()(size_t x, size_t y, size_t channel) const;

    void writePPM(const std::string& filename) const;
};

#include "ImageGPU.h"