#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

#include <torch/script.h>
#include <opencv2/opencv.hpp>

#include <calc_ops.hpp>

using namespace std;
using namespace cv;

using torch::from_blob;
using torch::ScalarType;
using torch::Tensor;
using torch::indexing::Slice;
using torch::jit::load;
using torch::jit::Object;
using torch::jit::script::Module;

int main(int argc, char* argv[])
try
{
    // Check arguments
    if (argc < 3)
    {
        printf("Usage: %s <model_path> <test_image>\n", argv[0]);
        return -1;
    }

    // Import model and settings
    Module model = load(argv[1]);
    Object yoroLayer = model.attr("yoroLayer").toObject();

    int netWidth = yoroLayer.attr("width").toInt();
    int netHeight = yoroLayer.attr("height").toInt();

    model.eval();

    // Read image
    Mat src = imread(argv[2], IMREAD_COLOR);
    cvtColor(src, src, COLOR_BGR2RGB);

    // Pad to square
    int tarSize = max(src.rows, src.cols);
    Mat mat = Mat(tarSize, tarSize, CV_8UC3, Scalar(0, 0, 0));
    Rect roi = Rect(
        (tarSize - src.cols) / 2, (tarSize - src.rows) / 2, src.cols, src.rows);
    src.copyTo(mat(roi));

    float startX = (tarSize - src.cols) / 2;
    float startY = (tarSize - src.rows) / 2;
    float scale = float(tarSize) / float(netWidth);

    // Resizing
    resize(mat, mat, Size(netWidth, netHeight));

    // Convert image to tensor
    Tensor inputs =
        from_blob(
            mat.ptr<char>(), {1, netHeight, netWidth, 3}, ScalarType::Byte)
            .to(torch::kFloat)
            .permute({0, 3, 1, 2})
            .contiguous() /
        255.0;

    // Forward
    auto outputs = model.forward({inputs}).toTuple();
    auto listRef = outputs->elements();
    Tensor predConf = listRef[0].toTensor();
    Tensor predClass = listRef[1].toTensor();
    Tensor predClassConf = listRef[2].toTensor();
    Tensor predBox = listRef[3].toTensor();
    Tensor predDeg = listRef[4].toTensor();

    // Denormalize
    predBox *= scale;
    predBox.index({"...", Slice(0, 2)}) -= torch::tensor({{{startX, startY}}});

    // Concatenate tensor
    Tensor pred = torch::cat(
                      {predConf.unsqueeze(-1).to(torch::kFloat),
                       predClass.unsqueeze(-1).to(torch::kFloat),
                       predClassConf.unsqueeze(-1),
                       predDeg.unsqueeze(-1),
                       predBox},
                      2)
                      .to(torch::kCPU);

    // Processing non-maximum suppression
    vector<vector<yoro_api::RBox>> nmsOut =
        yoro_api::non_maximum_suppression(pred, 0.9, 0.7);
    for (size_t n = 0; n < nmsOut.size(); n++)
    {
        cout << "Batch " << n << ":" << endl;
        for (size_t i = 0; i < nmsOut[n].size(); i++)
        {
            cout << nmsOut[n][i].to_string() << endl;
        }
        cout << endl;
    }

    return 0;
}
catch (exception& ex)
{
    cout << "Error occurred:" << endl;
    cout << ex.what() << endl;
    return -1;
}
