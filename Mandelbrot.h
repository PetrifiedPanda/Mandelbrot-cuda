#pragma once

#include "Image.h"

enum class ColorStrategy {
    GRAYSCALE, CONTINUOUS, ESCAPETIME
};

Image mandelbrotCPU(size_t size, int maxIts, double zoom, int xOffset, int yOffset, ColorStrategy strategy, bool invertColors = false);

Image mandelbrotGPU(size_t size, int maxIts, double zoom, int xOffset, int yOffset, ColorStrategy strategy, bool invertColors = false);