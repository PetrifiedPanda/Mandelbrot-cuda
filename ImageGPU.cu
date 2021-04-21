#include "ImageGPU.h"

#include <stdexcept>

#include <cuda.h>

ImageGPU::ImageGPU() : rows_(0), cols_(0), channels_(0), bytes_(nullptr) {}

ImageGPU::ImageGPU(size_t rows, size_t cols, size_t channels) : rows_(rows), cols_(cols), channels_(channels) {
    cudaMalloc(&bytes_,  cols_ * rows_ * channels_ * sizeof(uint8_t));
}

ImageGPU::ImageGPU(const ImageGPU& other) : rows_(other.rows_), cols_(other.cols_), channels_(other.channels_) {
    size_t byteSize = cols_ * rows_ * channels_ * sizeof(uint8_t);
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

const uint8_t* ImageGPU::bytes() const {
    return bytes_;
}

Image ImageGPU::toHost() const {
    Image result(rows_, cols_, channels_);
    cudaMemcpy(result.bytes_, bytes_, cols_ * rows_ * channels_ * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    return result;
}

ImageGPU::Ref::Ref(size_t rows, size_t cols, size_t channels, uint8_t* bytes) : rows_(rows), cols_(cols), channels_(channels), bytes_(bytes) {}

// Ref access operations

__device__ uint8_t& ImageGPU::Ref::operator()(size_t col, size_t row, size_t channel) {
    return bytes_[row * cols_ * channels_ + col * channels_ + channel];
}

__device__ const uint8_t& ImageGPU::Ref::operator()(size_t col, size_t row, size_t channel) const {
    return bytes_[row * cols_ * channels_ + col * channels_ + channel];
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