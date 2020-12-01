#pragma once

#include "Image.h"

enum class ColorStrategy {
    GRAYSCALE, CONTINUOUS, HISTOGRAM
};

Image mandelbrotCPU(int size, int maxIts, ColorStrategy strategy, bool invertColors = false);

Image mandelbrotGPU(int size, int maxIts, ColorStrategy strategy, bool invertColors = false);