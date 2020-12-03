#include "Image.h"

#include <fstream>

Image::Image() : xDim_(0), yDim_(0), channels_(0), bytes_(nullptr) {}

Image::Image(size_t xDim, size_t yDim, size_t channels) : xDim_(xDim), yDim_(yDim), channels_(channels) {
    bytes_ = new unsigned char[xDim_ * yDim_ * channels_];
}

Image::Image(const Image& other) : xDim_(other.xDim_), yDim_(other.yDim_), channels_(other.channels_) {
    size_t size = xDim_ * yDim_ * channels_;
    bytes_ = new unsigned char[size];
    for (size_t i = 0; i < size; ++i)
        bytes_[i] = other.bytes_[i];
}

Image::Image(Image&& other) : xDim_(other.xDim_), yDim_(other.yDim_), channels_(other.channels_), bytes_(other.bytes_) {
    other.bytes_ = nullptr;
}

Image::~Image() {
    delete[] bytes_;
}

// Getters

size_t Image::xDim() const {
    return xDim_;
}

size_t Image::yDim() const {
    return yDim_;
}

size_t Image::channels() const {
    return channels_;
}

ImageGPU Image::toDevice() {
    ImageGPU result(yDim_, xDim_, channels_);
    cudaMemcpy(result.bytes_, bytes_, xDim_ * yDim_ * channels_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
    return result;
}

// Access functions

unsigned char& Image::operator()(size_t x, size_t y, size_t channel) {
    return bytes_[y * xDim_ * channels_ + x * channels_ + channel];
}

const unsigned char& Image::operator()(size_t x, size_t y, size_t channel) const {
    return bytes_[y * xDim_ * channels_ + x * channels_ + channel];
}


void Image::writePPM(const std::string& filename) const {
    std::ofstream writer(filename, std::ios::binary);
    std::string ppmFormat = channels_ == 1 ? "P5" : "P6";
    writer << ppmFormat << "\n" << std::to_string(xDim_) << " " << std::to_string(yDim_) << "\n255\n";
    writer.write(reinterpret_cast<char*>(bytes_), yDim_ * xDim_ * channels_);
    writer.close();
}