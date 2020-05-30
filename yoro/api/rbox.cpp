#include <sstream>
#include <string>

#include "yoro_api.hpp"

using namespace std;

namespace yoro_api
{
RBox::operator string() const { return this->to_string().c_str(); }

string RBox::to_string() const
{
    stringstream ss;

    ss << "[";
    ss << "conf: " << this->conf << ", ";
    ss << "label: " << this->label << ", ";
    ss << "degree: " << this->degree << ", ";
    ss << "x: " << this->x << ", ";
    ss << "y: " << this->y << ", ";
    ss << "w: " << this->w << ", ";
    ss << "h: " << this->h;
    ss << "]";

    return ss.str();
}

}  // namespace yoro_api
