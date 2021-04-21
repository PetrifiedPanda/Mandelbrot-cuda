#include "Image.h"

#include <fstream>

Image::Image() : rows_(0), cols_(0), channels_(0), bytes_(nullptr) {}

Image::Image(size_t rows, size_t cols, size_t channels) : rows_(rows), cols_(cols), channels_(channels) {
    bytes_ = new uint8_t[cols_ * rows_ * channels_];
}

Image::Image(const Image& other) : rows_(other.rows_),cols_(other.cols_), channels_(other.channels_) {
    size_t size = cols_ * rows_ * channels_;
    bytes_ = new uint8_t[size];
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

const uint8_t* Image::bytes() const {
    return bytes_;
}

ImageGPU Image::to_device() {
    ImageGPU result(rows_, cols_, channels_);
    cudaMemcpy(result.bytes_, bytes_, cols_ * rows_ * channels_ * sizeof(uint8_t), cudaMemcpyHostToDevice);
    return result;
}

// Access functions

uint8_t& Image::operator()(size_t col, size_t row, size_t channel) {
    return bytes_[row * cols_ * channels_ + col * channels_ + channel];
}

const uint8_t& Image::operator()(size_t col, size_t row, size_t channel) const {
    return bytes_[row * cols_ * channels_ + col * channels_ + channel];
}


void Image::write_ppm(const std::string& filename) const {
    std::ofstream writer(filename, std::ios::binary);
    std::string ppmFormat = channels_ == 1 ? "P5" : "P6";
    writer << ppmFormat << "\n" << cols_ << " " << rows_ << "\n255\n";
    writer.write(reinterpret_cast<char*>(bytes_), rows_ * cols_ * channels_);
    writer.close();
}

Image Image::read_ppm(const std::string& filename) {
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