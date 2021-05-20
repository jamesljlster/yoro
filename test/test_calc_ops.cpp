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
