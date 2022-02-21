#include <iostream>

#include <opencv2/opencv.hpp>

#include <calc_ops.hpp>
#include <yoro_impl.hpp>

using namespace cv;
using namespace std;
using namespace yoro_api;

using torch::from_blob;
using torch::ScalarType;
using torch::Tensor;

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        printf("Usage: %s <image> <width> <height>\n", argv[0]);
        return -1;
    }

    const char* imgPath = argv[1];
    int width = stof(argv[2]);
    int height = stol(argv[3]);

    Mat image = imread(imgPath);
    cout << "Original Width: " << image.cols << endl;
    cout << "Original Height: " << image.rows << endl;

    imshow("Original Image", image);
    waitKey(0);
    destroyAllWindows();

    Mat resized = to_image(resize(from_image(image), {height, width}));
    imshow("Resized Image", resized);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
