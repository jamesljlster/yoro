#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>

#include <torch/script.h>

using namespace std;

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
    torch::jit::script::Module model = torch::jit::load(argv[1]);

    return 0;
}
catch (exception& ex)
{
    cout << ex.what() << endl;
    return -1;
}
