#include <iostream>

#include "Image.h"
#include "Mandelbrot.h"

int main() {
    std::cout << "Generating Image!\n";
    Image image = mandelbrotGPU(9250, 1000, ColorStrategy::HISTOGRAM);
    std::cout << "Saving Image!\n";
    image.writePPM("Mandelbrot.ppm");
}