#include <iostream>
#include <chrono>

#include "Image.h"
#include "Mandelbrot.h"

int main() {
    std::cout << "Generating Image!\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    Image image = mandelbrotGPU(9250, 1000, ColorStrategy::ESCAPETIME);

    auto end = std::chrono::high_resolution_clock::now();
    auto timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Generating the image took " << timeSpan.count() << " seconds!\n";
    
    std::cout << "Saving Image!\n";
    image.writePPM("Mandelbrot.ppm");
}