#include "ImageGPU.h"

#include <stdexcept>

#include <cuda.h>

ImageGPU::ImageGPU() : xDim_(0), yDim_(0), channels_(0), bytes_(nullptr) {}

ImageGPU::ImageGPU(size_t xDim, size_t yDim, size_t channels) : xDim_(xDim), yDim_(yDim), channels_(channels) {
    cudaMalloc(&bytes_,  xDim_ * yDim_ * channels_ * sizeof(uint8_t));
}

ImageGPU::ImageGPU(const ImageGPU& other) : xDim_(other.xDim_), yDim_(other.yDim_), channels_(other.channels_) {
    size_t byteSize = xDim_ * yDim_ * channels_ * sizeof(uint8_t);
    cudaMalloc(&bytes_, byteSize);
    cudaMemcpy(bytes_, other.bytes_, byteSize, cudaMemcpyDeviceToDevice);
}

ImageGPU::ImageGPU(ImageGPU&& other) : xDim_(other.xDim_), yDim_(other.yDim_), channels_(other.channels_), bytes_(other.bytes_) {
    other.bytes_ = nullptr;
}

ImageGPU::~ImageGPU() {
    cudaFree(bytes_);
}

size_t ImageGPU::xDim() const {
    return xDim_;
}

size_t ImageGPU::yDim() const {
    return yDim_;
}

size_t ImageGPU::channels() const {
    return channels_;
}

const uint8_t* ImageGPU::bytes() const {
    return bytes_;
}

Image ImageGPU::toHost() const {
    Image result(xDim_, yDim_, channels_);
    cudaMemcpy(result.bytes_, bytes_, xDim_ * yDim_ * channels_ * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    return result;
}

ImageGPU::Ref::Ref(size_t xDim, size_t yDim, size_t channels, uint8_t* bytes) : xDim_(xDim), yDim_(yDim), channels_(channels), bytes_(bytes) {}

// Ref access operations

__device__ uint8_t& ImageGPU::Ref::operator()(size_t x, size_t y, size_t channel) {
    return bytes_[y * xDim_ * channels_ + x * channels_ + channel];
}

__device__ const uint8_t& ImageGPU::Ref::operator()(size_t x, size_t y, size_t channel) const {
    return bytes_[y * xDim_ * channels_ + x * channels_ + channel];
}

// Ref getters

__device__ size_t ImageGPU::Ref::xDim() const {
    return xDim_;
}

__device__ size_t ImageGPU::Ref::yDim() const {
    return yDim_;
}

 __device__ size_t ImageGPU::Ref::channels() const {
    return channels_;
}

ImageGPU::Ref ImageGPU::getRef() {
    return Ref(xDim_, yDim_, channels_, bytes_);
}