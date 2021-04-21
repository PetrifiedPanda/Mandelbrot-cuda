#include "Mandelbrot.h"

#include <omp.h>

#include "ImageGPU.h"

struct Color {
    uint8_t r, g, b;
    __host__ __device__ constexpr Color() : r(0), g(0), b(0) {}
    __host__ __device__ constexpr Color(uint8_t r, uint8_t g, uint8_t b) : r(r), g(g), b(b) {}
    __host__ __device__ Color& invert() {
        r = 255 - r;
        g = 255 - g;
        b = 255 - b;
        return *this;
    }
};

constexpr size_t c_palette_size = 16;

constexpr Color h_palette[c_palette_size] = {
    Color(66, 30, 15),
    Color(25, 7, 26),
    Color(9, 1, 47),
    Color(4, 4, 73),
    Color(0, 7, 100),
    Color(12, 44, 138),
    Color(24, 82, 177),
    Color(57, 125, 209),
    Color(134, 181, 229),
    Color(211, 236, 248),
    Color(241, 233, 191),
    Color(248, 201, 95),
    Color(255, 170, 0),
    Color(204, 128, 0),
    Color(153, 87, 0),
    Color(106, 52, 3),
};

__constant__ Color d_palette[c_palette_size];

__device__ __host__ double scale(int x, int rangeSize, double begin, double end) {
    return begin + (end - begin) * x / rangeSize;
}

__device__ __host__ double lerp(double start, double end, double amount) {
    return start + amount * (end - start);
}

__device__ __host__ uint8_t lerp(uint8_t start, uint8_t end, double amount) {
    return start + amount * (static_cast<int>(end) - start);
}

__device__ __host__ Color lerp(Color start, Color end, double amount) {
    return Color(lerp(start.r, end.r, amount), lerp(start.g, end.g, amount), lerp(start.b, end.b, amount));
}

__device__ __host__ Color pick_color(ColorStrategy strategy, int iterations, int max_iterations, int x, int y, const Color palette[c_palette_size]) {
    switch (strategy) {
        case ColorStrategy::GRAYSCALE: {
            uint8_t color = scale(iterations, max_iterations, 0, 255);
            return Color(color, color, color);
        }
        case ColorStrategy::CONTINUOUS: { 
            // TODO: Fix
            double double_iterations = static_cast<double>(iterations);
            if (iterations < max_iterations) {
                double log_zn = log(static_cast<double>(x * x + y * y)) / 2;
                double nu = log(log_zn / log(2.0)) / log(2.0);
                double_iterations = double_iterations + 1 - nu;
            }
            double fractional = double_iterations - floor(static_cast<double>(double_iterations));

            Color color1 = palette[static_cast<size_t>(floor(static_cast<double>(double_iterations))) % c_palette_size];
            Color color2 = palette[static_cast<size_t>(floor(static_cast<double>(double_iterations)) + 1) % c_palette_size];

            Color final_color = lerp(color1, color2, fractional);
            return final_color;
        }
        case ColorStrategy::ESCAPETIME: {
            Color clr(0, 0, 0);
            if (iterations < max_iterations)
                clr = palette[iterations % c_palette_size];

            return clr;
        }
    }
    return Color(0, 0, 0);
}

__host__ __device__ int mandelbrot_iteration(int pixel_x, int pixel_y, size_t cols, size_t rows, int max_its, double zoom, int x_offset, int y_offset) {
    double scaled_x_offset = x_offset / static_cast<int>(cols) * 3;
    double scaled_y_offset = y_offset / static_cast<int>(rows) * 2;
    double scaled_x = scale(pixel_x + x_offset, cols, -2 / zoom + scaled_x_offset, 1 / zoom + scaled_x_offset);
    double scaled_y = scale(pixel_y + y_offset, rows, -1 / zoom + scaled_y_offset, 1 / zoom + scaled_y_offset);

    double x = 0.0;
    double y = 0.0; 
    int it = 0;
    while (x * x + y * y <= 4 && it < max_its) {
        double tmp_x = x * x - y * y + scaled_x;
        y = 2 * x * y + scaled_y;
        x = tmp_x;
    
        ++it;
    }
    return it;
}

__host__ __device__ Color color_pixel(size_t x, size_t y, size_t cols, size_t rows, int max_its, double zoom, int x_offset, int y_offset, ColorStrategy strategy, bool invert_colors, const Color palette[16]) {
    int it = mandelbrot_iteration(x, y, cols, rows, max_its, zoom, x_offset, y_offset);

    Color clr = pick_color(strategy, it, max_its, x, y, palette);
    if (invert_colors)
        clr.invert();
    return clr;
}

size_t decide_channels(ColorStrategy strategy, bool four_channels) {
    if (strategy == ColorStrategy::GRAYSCALE)
        return 1;
    else if (four_channels)
        return 4;
    else
        return 3;
}

Image mandelbrot_cpu(size_t size, int max_its, double zoom, int x_offset, int y_offset, ColorStrategy strategy, bool invert_colors, bool four_channels) {
    Image image(size, size * 1.5, decide_channels(strategy, four_channels));
    size_t cols = image.cols();
    size_t rows = image.rows();

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < cols; ++x) {
        for (int y = 0; y < rows; ++y) {
            Color clr = color_pixel(x, y, cols, rows, max_its, zoom, x_offset, y_offset, strategy, invert_colors, h_palette);

            if (strategy == ColorStrategy::GRAYSCALE)
                image(x, y, 0) = clr.r;
            else {
                image(x, y, 0) = clr.r;
                image(x, y, 1) = clr.g;
                image(x, y, 2) = clr.b;
            }
        }
    }

    return image;
}

__device__ int get_thread_id() {
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    return block_id * (blockDim.x * blockDim.y) + thread_id;
}

__global__ void mandelbrot_kernel(ImageGPU::Ref image, int max_its, double zoom, int x_offset, int y_offset, ColorStrategy strategy, bool invert_colors) {
    int pixel_index = get_thread_id();
    int y = pixel_index / image.cols();
    int x = pixel_index - y * image.cols();
    
    if (x < image.cols() && y < image.rows()) {
        Color clr = color_pixel(x, y, image.cols(), image.rows(), max_its, zoom, x_offset, y_offset, strategy, invert_colors, d_palette);

        if (strategy == ColorStrategy::GRAYSCALE)
            image(x, y, 0) = clr.r;
        else {
            image(x, y, 0) = clr.r;
            image(x, y, 1) = clr.g;
            image(x, y, 2) = clr.b;
        }
    }
}


Image mandelbrot_gpu(size_t size, int max_its, double zoom, int x_offset, int y_offset, ColorStrategy strategy, bool invert_colors, bool four_channels) {
    cudaMemcpyToSymbol(d_palette, h_palette, c_palette_size * sizeof(Color));

    ImageGPU gpuImage(size, size * 1.5, decide_channels(strategy, four_channels));
    size_t cols = gpuImage.cols();
    size_t rows = gpuImage.rows();

    int suggestedMinGridSize;
    int suggestedBlockSize;
    cudaOccupancyMaxPotentialBlockSize(&suggestedMinGridSize, &suggestedBlockSize, mandelbrot_kernel);

    unsigned int blockDimX = sqrt(suggestedBlockSize);
    unsigned int blockDimY = blockDimX;
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim(ceil(static_cast<double>(cols) / blockDimX), ceil(static_cast<double>(rows) / blockDimY));

    mandelbrot_kernel<<<gridDim, blockDim>>>(gpuImage.get_ref(), max_its, zoom, x_offset, y_offset, strategy, invert_colors);
    cudaDeviceSynchronize();

    return gpuImage.to_host();
}