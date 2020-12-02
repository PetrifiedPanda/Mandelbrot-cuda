#include "ImageGPU.h"

#include <stdexcept>

#include <cuda.h>

ImageGPU::ImageGPU() : rows_(0), cols_(0), channels_(0), bytes_(nullptr) {}

ImageGPU::ImageGPU(size_t rows, size_t cols, size_t channels) : rows_(rows), cols_(cols), channels_(channels) {
    cudaMalloc(&bytes_, rows_ * cols_ * channels_ * sizeof(unsigned char));
}

ImageGPU::ImageGPU(const ImageGPU& other) : rows_(other.rows_), cols_(other.cols_), channels_(other.channels_) {
    size_t byteSize = rows_ * cols_ * channels_ * sizeof(unsigned char);
    cudaMalloc(&bytes_, byteSize);
    cudaMemcpy(bytes_, other.bytes_, byteSize, cudaMemcpyDeviceToDevice);
}

ImageGPU::ImageGPU(ImageGPU&& other) : rows_(other.rows_), cols_(other.cols_), channels_(other.channels_), bytes_(other.bytes_) {
    other.bytes_ = nullptr;
}

ImageGPU::~ImageGPU() {
    cudaFree(bytes_);
}

size_t ImageGPU::rows() const {
    return rows_;
}

size_t ImageGPU::cols() const {
    return cols_;
}

size_t ImageGPU::channels() const {
    return channels_;
}

Image ImageGPU::toHost() const {
    Image result(rows_, cols_, channels_);
    cudaMemcpy(result.bytes_, bytes_, rows_ * cols_ * channels_ * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    return result;
}

ImageGPU::Ref::Ref(size_t rows, size_t cols, size_t channels, unsigned char* bytes) : rows_(rows), cols_(cols), channels_(channels), bytes_(bytes) {}

// Ref access operations

__device__ unsigned char& ImageGPU::Ref::operator()(size_t row, size_t col, size_t channel) {
    return bytes_[col * rows_ * channels_ + row * channels_ + channel];
}

__device__ const unsigned char& ImageGPU::Ref::operator()(size_t row, size_t col, size_t channel) const {
    return bytes_[col * rows_ * channels_ + row * channels_ + channel];
}

// Ref getters

__device__ size_t ImageGPU::Ref::rows() const {
    return rows_;
}

__device__ size_t ImageGPU::Ref::cols() const {
    return cols_;
}

 __device__ size_t ImageGPU::Ref::channels() const {
    return channels_;
}

ImageGPU::Ref ImageGPU::getRef() {
    return Ref(rows_, cols_, channels_, bytes_);
}