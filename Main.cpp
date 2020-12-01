#include <iostream>

#include "Image.h"
#include "Mandelbrot.h"

int main() {
    Image image = mandelbrotGPU(9250, 1000, ColorStrategy::HISTOGRAM);
    image.writePPM("Mandelbrot.ppm");
}