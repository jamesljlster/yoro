#include <cmath>
#include <exception>
#include <stdexcept>

#include <torch/script.h>

#include "calc_ops.hpp"
#include "yoro_impl.hpp"

using torch::from_blob;
using torch::ScalarType;
using torch::Tensor;
using torch::tensor;
using torch::indexing::Ellipsis;
using torch::indexing::Slice;
using torch::jit::IValue;
using torch::jit::Object;

namespace yoro_api
{
Tensor from_image(const uint8_t* image, int width, int height, int channels)
{
    return from_blob((void*)image, {1, height, width, channels}, torch::kUInt8)
        .permute({0, 3, 1, 2})
        .contiguous();
}

Tensor from_image(const cv::Mat& image)
{
    return from_image(
        (const uint8_t*)image.ptr<char>(),
        image.cols,
        image.rows,
        image.channels());
}

GeneralDetector::GeneralDetector(
    const char* modelPath, const DeviceType& devType)
{
    // Detect devices and set tensor options
    bool cudaAvail = torch::cuda::is_available();
    switch (devType)
    {
        case DeviceType::Auto:
            if (cudaAvail)
                this->device = torch::kCUDA;
            else
                this->device = torch::kCPU;
            break;

        case DeviceType::CPU:
            this->device = torch::kCPU;
            break;

        case DeviceType::CUDA:
            this->device = torch::kCUDA;
            break;
    }

    if ((this->device == torch::kCUDA) && (!cudaAvail))
    {
        throw std::runtime_error(
            this->make_error_msg("CUDA device is unavailable."));
    }

    this->opt = this->opt.device(this->device).dtype(this->scalarType);

    // Import model and settings
    this->model = torch::jit::load(modelPath);
    Object suffixLayer = model.attr("suffix").toObject();

    this->netWidth = suffixLayer.attr("width").toInt();
    this->netHeight = suffixLayer.attr("height").toInt();

    this->model.to(this->device, this->scalarType);
    this->model.eval();
}

int GeneralDetector::network_width() const { return this->netWidth; }
int GeneralDetector::network_height() const { return this->netHeight; }

torch::jit::IValue GeneralDetector::detect(const Tensor& image)
{
    // Cast and normalize
    Tensor inputs = image.to(this->opt) / 255.0;

    // Resize and forward
    return model.forward({resize(inputs, {this->netHeight, this->netWidth})});
}

std::string GeneralDetector::make_error_msg(const char* msg) const
{
    return std::string("[YORO API (Error)] ") + std::string(msg);
}

std::vector<RBox> YORODetector::Impl::detect(
    const Tensor& image, float confTh, float nmsTh)
{
    int width = this->netWidth;
    int height = this->netHeight;

    // Pad to aspect ratio
    std::tuple<Tensor, std::vector<long>> padRet =
        pad_to_aspect(image, (float)width / height);
    Tensor inputs = std::get<0>(padRet);
    std::vector<long> padParams = std::get<1>(padRet);
    int startX = padParams[0];
    int startY = padParams[2];
    float scale = float(inputs.size(-1)) / float(width);

    // Forward, denormalize and flatten predictions
    std::vector<std::tuple<Tensor, Tensor, Tensor>> fwOutputs;
    auto outList = GeneralDetector::detect(inputs).toList();
    for (const IValue& tup : outList)
    {
        // Extract tensors
        auto tupRef = tup.toTuple()->elements();
        Tensor conf = tupRef[0].toTensor();
        Tensor label = tupRef[1].toTensor();
        Tensor rboxes = tupRef[2].toTensor();

        // Denormalize
        rboxes.index({Ellipsis, Slice(1, 5)}).mul_(scale);
        rboxes.index({Ellipsis, Slice(1, 3)})
            .sub_(tensor({{{{{startX, startY}}}}}, this->device));

        fwOutputs.push_back({conf, label, rboxes});
    }

    auto flatOut = flatten_prediction(fwOutputs);

    // Processing non-maximum suppression
    std::vector<std::vector<RBox>> nmsOut =
        yoro_api::non_maximum_suppression(flatOut, confTh, nmsTh);

    return nmsOut[0];
}

std::vector<RBox> YORODetector::Impl::detect(
    const uint8_t* image,
    int width,
    int height,
    int channels,
    float confTh,
    float nmsTh)
{
    return this->detect(
        from_image(image, width, height, channels), confTh, nmsTh);
}

std::vector<RBox> YORODetector::Impl::detect(
    const cv::Mat& image, float confTh, float nmsTh)
{
    return this->detect(from_image(image), confTh, nmsTh);
}

float RotationDetector::Impl::detect(const torch::Tensor& image)
{
    int width = this->netWidth;
    int height = this->netHeight;

    // Pad to aspect ratio
    Tensor inputs = std::get<0>(pad_to_aspect(image, (float)width / height));

    // Forward
    Tensor outputs = GeneralDetector::detect(inputs).toTensor();

    return outputs.to(torch::kFloat32).item<float>();
}

float RotationDetector::Impl::detect(
    const uint8_t* image, int width, int height, int channels)
{
    return this->detect(from_image(image, width, height, channels));
}

float RotationDetector::Impl::detect(const cv::Mat& image)
{
    return this->detect(from_image(image));
}

}  // namespace yoro_api
