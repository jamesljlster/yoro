#include <yoro_api.hpp>

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
    RotationDetector detector(argv[1], devType);
    Mat image = imread(argv[2], IMREAD_COLOR);

    // Run detection
    float pred = detector.detect(image);
    cout << "Detection: " << pred << endl;

    return 0;
}
catch (exception& ex)
{
    cout << "Error occurred:" << endl;
    cout << ex.what() << endl;
    return -1;
}
