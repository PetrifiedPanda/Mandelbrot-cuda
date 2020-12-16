#include "Image.h"

#include <fstream>

Image::Image() : xDim_(0), yDim_(0), channels_(0), bytes_(nullptr) {}

Image::Image(size_t xDim, size_t yDim, size_t channels) : xDim_(xDim), yDim_(yDim), channels_(channels) {
    bytes_ = new uint8_t[xDim_ * yDim_ * channels_];
}

Image::Image(const Image& other) : xDim_(other.xDim_), yDim_(other.yDim_), channels_(other.channels_) {
    size_t size = xDim_ * yDim_ * channels_;
    bytes_ = new uint8_t[size];
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
    cudaMemcpy(result.bytes_, bytes_, xDim_ * yDim_ * channels_ * sizeof(uint8_t), cudaMemcpyHostToDevice);
    return result;
}

// Access functions

uint8_t& Image::operator()(size_t x, size_t y, size_t channel) {
    return bytes_[y * xDim_ * channels_ + x * channels_ + channel];
}

const uint8_t& Image::operator()(size_t x, size_t y, size_t channel) const {
    return bytes_[y * xDim_ * channels_ + x * channels_ + channel];
}


void Image::writePPM(const std::string& filename) const {
    std::ofstream writer(filename, std::ios::binary);
    std::string ppmFormat = channels_ == 1 ? "P5" : "P6";
    writer << ppmFormat << "\n" << xDim_ << " " << yDim_ << "\n255\n";
    writer.write(reinterpret_cast<char*>(bytes_), yDim_ * xDim_ * channels_);
    writer.close();
}

Image Image::readPPM(const std::string& filename) {
    std::ifstream reader(filename, std::ios::binary);
    std::string ppmFormat;
    reader >> ppmFormat;
    size_t channels = ppmFormat == "P5" ? 1 : 3;
    size_t xDim, yDim;
    int b;
    reader >> xDim >> yDim >> b;
    if (b != 255)
        throw std::runtime_error("Cannot read file"); // TODO: Make this unnecessary

    reader.ignore(256, '\n');

    Image result(xDim, yDim, channels);
    reader.read(reinterpret_cast<char*>(result.bytes_), xDim * yDim * channels);

    reader.close();
    return result;
}