#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <yoro_impl.hpp>

using namespace std;
using namespace cv;
using namespace yoro_api;

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: %s <image> <aspect_ratio>\n", argv[0]);
        return -1;
    }

    const char* imgPath = argv[1];
    float aspectRatio = stof(argv[2]);

    std::tuple<cv::Mat, int, int> result;

    result = pad_to_aspect(imread(imgPath, IMREAD_COLOR), aspectRatio);
    imshow("Result", std::get<0>(result));
    waitKey(0);

    result = pad_to_aspect(imread(imgPath, IMREAD_COLOR), aspectRatio);
    int startX = std::get<1>(result);
    int startY = std::get<2>(result);
    printf("startX: %d, startY: %d\n", startX, startY);

    return 0;
}
