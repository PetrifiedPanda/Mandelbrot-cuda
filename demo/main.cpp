#include <iostream>
#include <chrono>

#include <Image.h>
#include <mandelbrot.h>

using namespace mandelbrot;

int main() {
    std::cout << "Generating Image!\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    Image image = mandelbrot_gpu(1000, 1000, 1, 0, 0, ColorStrategy::ESCAPETIME);

    auto end = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Generating the image took " << time_span.count() << " seconds!\n";
    
    std::cout << "Saving Image!\n";
    image.write_ppm("Mandelbrot.ppm");
}
