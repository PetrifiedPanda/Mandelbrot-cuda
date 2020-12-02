#pragma once

#include "Image.h"

enum class ColorStrategy {
    GRAYSCALE, CONTINUOUS, HISTOGRAM
};

Image mandelbrotCPU(size_t size, int maxIts, ColorStrategy strategy, bool invertColors = false);

Image mandelbrotGPU(size_t size, int maxIts, ColorStrategy strategy, bool invertColors = false);