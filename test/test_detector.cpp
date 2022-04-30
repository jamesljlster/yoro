#include <yoro_api.hpp>

#ifdef WITH_OPENCV

using namespace yoro_api;
using namespace std;
using namespace cv;

int main(int argc, char* argv[])
try
{
    // Check arguments
    if (argc < 3)
    {
        printf("Usage: %s <model_path> <test_image> [device]\n", argv[0]);
        return -1;
    }

    DeviceType devType = DeviceType::Auto;
    if (argc > 3)
    {
        if (string(argv[3]) == "cpu")
        {
            devType = DeviceType::CPU;
        }
        else if (string(argv[3]) == "cuda")
        {
            devType = DeviceType::CUDA;
        }
    }

    // Import model and load image
    YORODetector detector(argv[1], devType);
    Mat image = imread(argv[2], IMREAD_COLOR);

    // Run detection
    vector<RBox> pred = detector.detect(image, 0.9, 0.7);
    for (size_t i = 0; i < pred.size(); i++)
    {
        cout << pred[i].to_string() << endl;
    }

    return 0;
}
catch (exception& ex)
{
    cout << "Error occurred:" << endl;
    cout << ex.what() << endl;
    return -1;
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
