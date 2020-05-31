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
        printf("Usage: %s <model_path> <test_image>\n", argv[0]);
        return -1;
    }

    // Import model and load image
    Detector detector(argv[1]);
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
