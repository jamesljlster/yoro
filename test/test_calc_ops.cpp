#include <cmath>
#include <exception>
#include <iostream>

#include <calc_ops.hpp>

using namespace std;
using namespace torch;
using namespace yoro_api;

int main()
try
{
    // deg2rad: float
    {
        float deg = 45;
        float rad = deg2rad(deg);
        cout << "=== deg2rad: float ===" << endl;
        cout << rad << endl;
        cout << endl;
    }

    // deg2rad: tensor
    {
        Tensor deg = torch::tensor({45.0});
        Tensor rad = deg2rad(deg);
        cout << "=== deg2rad: tensor ===" << endl;
        cout << rad << endl;
        cout << endl;
    }

    // BBox to corners
    {
        Tensor bbox = torch::tensor({{10, 15, 20, 30}, {100, 200, 40, 60}});
        Tensor corners = bbox_to_corners(bbox);
        cout << "=== bbox to corners ===" << endl;
        cout << "bbox" << endl;
        cout << bbox << endl;
        cout << "corners" << endl;
        cout << corners << endl;
        cout << endl;
    }

    return 0;
}
catch (exception& ex)
{
    cout << ex.what() << endl;
    return -1;
}
