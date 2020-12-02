#include "Image.h"

#include <fstream>

Image::Image() : rows_(0), cols_(0), channels_(0), bytes_(nullptr) {}

Image::Image(size_t rows, size_t cols, size_t channels) : rows_(rows), cols_(cols), channels_(channels) {
    bytes_ = new unsigned char[rows_ * cols_ * channels_];
}

Image::Image(const Image& other) : rows_(other.rows_), cols_(other.cols_), channels_(other.channels_) {
    size_t size = rows_ * cols_ * channels_;
    bytes_ = new unsigned char[size];
    for (size_t i = 0; i < size; ++i)
        bytes_[i] = other.bytes_[i];
}

Image::Image(Image&& other) : rows_(other.rows_), cols_(other.cols_), channels_(other.channels_), bytes_(other.bytes_) {
    other.bytes_ = nullptr;
}

Image::~Image() {
    delete[] bytes_;
}

// Getters

size_t Image::rows() const {
    return rows_;
}
size_t Image::cols() const {
    return cols_;
}
size_t Image::channels() const {
    return channels_;
}

ImageGPU Image::toDevice() {
    ImageGPU result(rows_, cols_, channels_);
    cudaMemcpy(result.bytes_, bytes_, rows_ * cols_ * channels_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
    return result;
}

// Access functions

unsigned char& Image::operator()(size_t row, size_t col, size_t channel) {
    return bytes_[col * rows_ * channels_ + row * channels_ + channel];
}

const unsigned char& Image::operator()(size_t row, size_t col, size_t channel) const {
    return bytes_[col * rows_ * channels_ + row * channels_ + channel];
}


void Image::writePPM(const std::string& filename) const {
    std::ofstream writer(filename, std::ios::binary);
    std::string ppmFormat = channels_ == 1 ? "P5" : "P6";
    writer << ppmFormat << "\n" << std::to_string(rows_) << " " << std::to_string(cols_) << "\n255\n";
    writer.write(reinterpret_cast<char*>(bytes_), rows_ * cols_ * channels_);
    writer.close();
}