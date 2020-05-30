#ifndef __YORO_API_HPP__
#define __YORO_API_HPP__

#include <string>

namespace yoro_api
{
struct RBox
{
    RBox(float conf, int label, float degree, float x, float y, float w,
         float h)
        : conf(conf), label(label), degree(degree), x(x), y(y), w(w), h(h)
    {
    }

    operator std::string() const;

    std::string to_string() const;

    float conf;
    int label;
    float degree;
    float x;
    float y;
    float w;
    float h;
};

}  // namespace yoro_api

#endif
