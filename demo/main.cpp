#include <iostream>
#include <chrono>
#include <string_view>

#include <Image.h>
#include <mandelbrot.h>

using namespace mandelbrot;

int main(int argc, char** argv) {
    bool is_cpu = false;
    constexpr std::string_view CPU_STR = "cpu";
    constexpr std::string_view GPU_STR = "gpu";
    if (argc > 1) {
        const std::string_view arg = argv[1];
        if (arg == CPU_STR) {
            is_cpu = true;
        } else if (arg == GPU_STR) {
            is_cpu = false;
        } else {
            std::cerr << "Invalid argument \"" << arg << "\"\n";
            abort();
        }
    }
    
    const std::string_view cpu_gpu_str = is_cpu ? CPU_STR : GPU_STR;
    std::cout << "Generating image on " << cpu_gpu_str << '\n';
    auto start = std::chrono::high_resolution_clock::now();
    
    const size_t size = 1000, max_its = 1000;
    const double zoom = 1.0;
    const int x_offset = 0, y_offset = 0;
    const ColorStrategy strat = ColorStrategy::ESCAPETIME;
    const Image image = is_cpu 
                        ? mandelbrot_cpu(size, max_its, zoom, x_offset, y_offset, strat) 
                        : mandelbrot_gpu(size, max_its, zoom, x_offset, y_offset, strat); 

    auto end = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Generating the image took " << time_span.count() << " seconds!\n";
    
    std::cout << "Saving Image!\n";
    image.write_ppm("Mandelbrot.ppm");
}
