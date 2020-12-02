#include "Mandelbrot.h"

#include <omp.h>

#include "ImageGPU.h"

struct Color {
    unsigned char r, g, b;
    __host__ __device__ constexpr Color() : r(0), g(0), b() {}
    __host__ __device__ constexpr Color(unsigned char r, unsigned char g, unsigned char b) : r(r), g(g), b(b) {}
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

__constant__ Color d_palette[c_paletteSize] = {
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

__device__ __host__ double scale(int x, int rangeSize, double begin, double end) {
    return begin + (end - begin) * x / rangeSize;
}

__device__ __host__ double lerp(double start, double end, double amount) {
    return start + amount * (end - start);
}

__device__ __host__ Color pickColor(ColorStrategy strategy, int iterations, int maxIterations, int x, int y, const Color palette[c_paletteSize]) {
    switch (strategy) {
        case ColorStrategy::GRAYSCALE: {
            unsigned char color = scale(iterations, maxIterations, 0, 255);
            return Color(color, color, color);
        }
        case ColorStrategy::CONTINUOUS: { 
            // TODO
            double dIterations = iterations;
            if (iterations < maxIterations) {
                double logZN = log(static_cast<double>(x * x + y * y)) / 2;
                double nu = log(logZN / log(2.0)) / log(2.0);
                dIterations = dIterations + 1 - nu;
            }
            double fractional = dIterations - floor(static_cast<double>(dIterations));

            Color color1 = palette[static_cast<size_t>(floor(static_cast<double>(dIterations))) % c_paletteSize];
            Color color2 = palette[static_cast<size_t>(floor(static_cast<double>(dIterations)) + 1) % c_paletteSize];
        
            return Color(lerp(color1.r, color2.r, fractional), lerp(color1.g, color2.g, fractional), lerp(color1.b, color2.b, fractional));
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

__host__ __device__ int mandelbrotIteration(int pX, int pY, size_t rows, size_t cols, int maxIts) {
    double scaledX = scale(pX, rows, -2, 1);
    double scaledY = scale(pY, cols, -1, 1);

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

template <class ImageType>
__host__ __device__ void colorPixel(ImageType& image, size_t x, size_t y, int maxIts, ColorStrategy strategy, bool invertColors, const Color palette[16]) {
    int it = mandelbrotIteration(x, y, image.rows(), image.cols(), maxIts);

    Color clr = pickColor(strategy, it, maxIts, x, y, palette);
    if (invertColors)
        clr.invert();

    if (strategy == ColorStrategy::GRAYSCALE)
        image(x, y, 0) = clr.r;
    else {
        image(x, y, 0) = clr.r;
        image(x, y, 1) = clr.g;
        image(x, y, 2) = clr.b;
    }
}

Image mandelbrotCPU(size_t size, int maxIts, ColorStrategy strategy, bool invertColors) {
    Image image(size * 1.5, size, strategy == ColorStrategy::GRAYSCALE ? 1 : 3);
    size_t rows = image.rows();
    size_t cols = image.cols();

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < rows; ++x) {
        for (int y = 0; y < cols; ++y) {
            colorPixel(image, x, y, maxIts, strategy, invertColors, h_palette);
        }
    }

    return image;
}

__device__ int getThreadId() {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    return blockId * (blockDim.x * blockDim.y) + threadId;
}

__global__ void mandelbrotKernel(ImageGPU::Ref image, int maxIts, ColorStrategy strategy, bool invertColors) {
    int pixelIndex = getThreadId();
    int x = pixelIndex / image.cols();
    int y = pixelIndex - x * image.cols();
    
    if (x < image.rows() && y < image.cols())
        colorPixel(image, x, y, maxIts, strategy, invertColors, d_palette);
}


Image mandelbrotGPU(size_t size, int maxIts, ColorStrategy strategy, bool invertColors) {
    ImageGPU gpuImage(size * 1.5, size, strategy == ColorStrategy::GRAYSCALE ? 1 : 3);
    size_t rows = gpuImage.rows();
    size_t cols = gpuImage.cols();

    int suggestedMinGridSize;
    int suggestedBlockSize;
    cudaOccupancyMaxPotentialBlockSize(&suggestedMinGridSize, &suggestedBlockSize, mandelbrotKernel);

    size_t blockDimX = sqrt(suggestedBlockSize);
    size_t blockDimY = blockDimX;
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim(ceil(static_cast<double>(cols) / blockDimX), ceil(static_cast<double>(rows) / blockDimY));

    mandelbrotKernel<<<gridDim, blockDim>>>(gpuImage.getRef(), maxIts, strategy, invertColors);
    cudaDeviceSynchronize();

    return gpuImage.toHost();
}