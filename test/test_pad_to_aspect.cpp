#include <yoro_api.hpp>

#ifdef WITH_OPENCV
#    include <cstdio>
#    include <iostream>
#    include <tuple>

#    include <opencv2/opencv.hpp>

#    include <calc_ops.hpp>
#    include <yoro_impl.hpp>

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

    std::tuple<torch::Tensor, vector<long>> result;

    result =
        pad_to_aspect(from_image(imread(imgPath, IMREAD_COLOR)), aspectRatio);
    imshow("Result", to_image(std::get<0>(result)));
    waitKey(0);

    result =
        pad_to_aspect(from_image(imread(imgPath, IMREAD_COLOR)), aspectRatio);
    int startX = std::get<1>(result)[0];
    int startY = std::get<1>(result)[2];
    printf("startX: %d, startY: %d\n", startX, startY);

    return 0;
}

#else
#    include <iostream>
using namespace std;
int main()
{
    cout << "The test is ignored due to OpenCV support is disabled." << endl;
    return 0;
}
#endif
