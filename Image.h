#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mandelbrot {

class ImageGPU;

class Image {
    friend class ImageGPU;
    size_t rows_, cols_, channels_;
    uint8_t* bytes_;

   public:
    Image();
    Image(size_t rows, size_t cols, size_t channels = 3);
    Image(const Image& other);
    Image(Image&& other);
    ~Image();

    size_t rows() const;
    size_t cols() const;
    size_t channels() const;

    const uint8_t* bytes() const;

    ImageGPU to_device();

    uint8_t& operator()(size_t col, size_t row, size_t channel);
    const uint8_t& operator()(size_t col, size_t row, size_t channel) const;

    void write_ppm(const std::string& filename) const;

    static Image read_ppm(const std::string& filename);
};

} // namespace mandelbrot

#include "ImageGPU.h"

