#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

#include <torch/script.h>
#include <opencv2/opencv.hpp>

#define NET_SIZE 224

using namespace std;
using namespace cv;

using torch::from_blob;
using torch::ScalarType;
using torch::Tensor;
using torch::jit::load;
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

    // Import model
    Module model = load(argv[1]);
    model.eval();

    // Read image
    Mat src = imread(argv[2], IMREAD_COLOR);
    cvtColor(src, src, COLOR_BGR2RGB);

    // Pad to square
    int tarSize = max(src.rows, src.cols);
    Mat mat = Mat(tarSize, tarSize, CV_8UC3, Scalar(0, 0, 0));
    Rect roi = Rect((tarSize - src.cols) / 2, (tarSize - src.rows) / 2,
                    src.cols, src.rows);
    src.copyTo(mat(roi));

    // Resizing
    resize(mat, mat, Size(NET_SIZE, NET_SIZE));

    // Convert image to tensor
    Tensor inputs =
        from_blob(mat.ptr<char>(), {1, NET_SIZE, NET_SIZE, 3}, ScalarType::Byte)
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

    cout << (predConf > 0.9).sum() << endl;

    return 0;
}
catch (exception& ex)
{
    cout << ex.what() << endl;
    return -1;
}
