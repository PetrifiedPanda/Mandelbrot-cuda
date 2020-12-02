#pragma once

#include <cstddef>
#include <string>

class ImageGPU;

class Image {
    friend class ImageGPU;
    size_t rows_, cols_, channels_;
    unsigned char* bytes_;

   public:
    Image();
    Image(size_t rows, size_t cols, size_t channels = 3);
    Image(const Image& other);
    Image(Image&& other);
    ~Image();

    size_t rows() const;
    size_t cols() const;
    size_t channels() const;

    ImageGPU toDevice();

    unsigned char& operator()(size_t row, size_t col, size_t channel);
    const unsigned char& operator()(size_t row, size_t col, size_t channel) const;

    void writePPM(const std::string& filename) const;
};

#include "ImageGPU.h"