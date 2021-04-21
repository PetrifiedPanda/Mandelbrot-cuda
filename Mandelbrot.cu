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

constexpr size_t c_paletteSize = 16;

constexpr Color h_palette[c_paletteSize] = {
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

__constant__ Color d_palette[c_paletteSize];

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

__device__ __host__ Color pickColor(ColorStrategy strategy, int iterations, int maxIterations, int x, int y, const Color palette[c_paletteSize]) {
    switch (strategy) {
        case ColorStrategy::GRAYSCALE: {
            uint8_t color = scale(iterations, maxIterations, 0, 255);
            return Color(color, color, color);
        }
        case ColorStrategy::CONTINUOUS: { 
            // TODO: Fix
            double dIterations = static_cast<double>(iterations);
            if (iterations < maxIterations) {
                double logZN = log(static_cast<double>(x * x + y * y)) / 2;
                double nu = log(logZN / log(2.0)) / log(2.0);
                dIterations = dIterations + 1 - nu;
            }
            double fractional = dIterations - floor(static_cast<double>(dIterations));

            Color color1 = palette[static_cast<size_t>(floor(static_cast<double>(dIterations))) % c_paletteSize];
            Color color2 = palette[static_cast<size_t>(floor(static_cast<double>(dIterations)) + 1) % c_paletteSize];

            Color finalColor = lerp(color1, color2, fractional);
            return finalColor;
        }
        case ColorStrategy::ESCAPETIME: {
            Color clr(0, 0, 0);
            if (iterations < maxIterations)
                clr = palette[iterations % c_paletteSize];

            return clr;
        }
    }
    return Color(0, 0, 0);
}

__host__ __device__ int mandelbrotIteration(int pX, int pY, size_t cols, size_t rows, int maxIts, double zoom, int xOffset, int yOffset) {
    double scaledXOffset = xOffset / static_cast<int>(cols) * 3;
    double scaledYOffset = yOffset / static_cast<int>(rows) * 2;
    double scaledX = scale(pX + xOffset, cols, -2 / zoom + scaledXOffset, 1 / zoom + scaledXOffset);
    double scaledY = scale(pY + yOffset, rows, -1 / zoom + scaledYOffset, 1 / zoom + scaledYOffset);

    double x = 0.0;
    double y = 0.0; 
    int it = 0;
    while (x * x + y * y <= 4 && it < maxIts) {
        double tmpX = x * x - y * y + scaledX;
        y = 2 * x * y + scaledY;
        x = tmpX;
    
        ++it;
    }
    return it;
}

__host__ __device__ Color colorPixel(size_t x, size_t y, size_t cols, size_t rows, int maxIts, double zoom, int xOffset, int yOffset, ColorStrategy strategy, bool invertColors, const Color palette[16]) {
    int it = mandelbrotIteration(x, y, cols, rows, maxIts, zoom, xOffset, yOffset);

    Color clr = pickColor(strategy, it, maxIts, x, y, palette);
    if (invertColors)
        clr.invert();
    return clr;
}

size_t decideChannels(ColorStrategy strategy, bool fourChannels) {
    if (strategy == ColorStrategy::GRAYSCALE)
        return 1;
    else if (fourChannels)
        return 4;
    else
        return 3;
}

Image mandelbrotCPU(size_t size, int maxIts, double zoom, int xOffset, int yOffset, ColorStrategy strategy, bool invertColors, bool fourChannels) {
    Image image(size, size * 1.5, decideChannels(strategy, fourChannels));
    size_t cols = image.cols();
    size_t rows = image.rows();

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < cols; ++x) {
        for (int y = 0; y < rows; ++y) {
            Color clr = colorPixel(x, y, cols, rows, maxIts, zoom, xOffset, yOffset, strategy, invertColors, h_palette);

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

__device__ int getThreadId() {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    return blockId * (blockDim.x * blockDim.y) + threadId;
}

__global__ void mandelbrotKernel(ImageGPU::Ref image, int maxIts, double zoom, int xOffset, int yOffset, ColorStrategy strategy, bool invertColors) {
    int pixelIndex = getThreadId();
    int y = pixelIndex / image.cols();
    int x = pixelIndex - y * image.cols();
    
    if (x < image.cols() && y < image.rows()) {
        Color clr = colorPixel(x, y, image.cols(), image.rows(), maxIts, zoom, xOffset, yOffset, strategy, invertColors, d_palette);

        if (strategy == ColorStrategy::GRAYSCALE)
            image(x, y, 0) = clr.r;
        else {
            image(x, y, 0) = clr.r;
            image(x, y, 1) = clr.g;
            image(x, y, 2) = clr.b;
        }
    }
}


Image mandelbrotGPU(size_t size, int maxIts, double zoom, int xOffset, int yOffset, ColorStrategy strategy, bool invertColors, bool fourChannels) {
    cudaMemcpyToSymbol(d_palette, h_palette, c_paletteSize * sizeof(Color));

    ImageGPU gpuImage(size, size * 1.5, decideChannels(strategy, fourChannels));
    size_t cols = gpuImage.cols();
    size_t rows = gpuImage.rows();

    int suggestedMinGridSize;
    int suggestedBlockSize;
    cudaOccupancyMaxPotentialBlockSize(&suggestedMinGridSize, &suggestedBlockSize, mandelbrotKernel);

    unsigned int blockDimX = sqrt(suggestedBlockSize);
    unsigned int blockDimY = blockDimX;
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim(ceil(static_cast<double>(cols) / blockDimX), ceil(static_cast<double>(rows) / blockDimY));

    mandelbrotKernel<<<gridDim, blockDim>>>(gpuImage.getRef(), maxIts, zoom, xOffset, yOffset, strategy, invertColors);
    cudaDeviceSynchronize();

    return gpuImage.toHost();
}