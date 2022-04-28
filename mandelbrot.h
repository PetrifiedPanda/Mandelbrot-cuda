#pragma once

#include "Image.h"

namespace mandelbrot {

enum class ColorStrategy {
    GRAYSCALE, CONTINUOUS, ESCAPETIME
};

Image mandelbrot_cpu(size_t size, int max_its, double zoom, int x_offset, int y_offset, ColorStrategy strategy, bool invert_colors = false, bool four_channels = false);

Image mandelbrot_gpu(size_t size, int max_its, double zoom, int x_offset, int y_offset, ColorStrategy strategy, bool invert_colors = false, bool four_channels = false);

} // namespace mandelbrot
