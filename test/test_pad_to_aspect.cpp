#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
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

    Mat result = pad_to_aspect(imread(imgPath, IMREAD_COLOR), aspectRatio);
    imshow("Result", result);
    waitKey(0);

    int startX = 0;
    int startY = 0;
    result = pad_to_aspect(
        imread(imgPath, IMREAD_COLOR), aspectRatio, &startX, &startY);
    printf("startX: %d, startY: %d\n", startX, startY);

    return 0;
}
