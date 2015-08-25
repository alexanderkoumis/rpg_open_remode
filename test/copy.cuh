#ifndef RMD_TEST_COPY_CUH
#define RMD_TEST_COPY_CUH

#include <rmd/device_image.cuh>

namespace rmd
{

void copy(
    /*
    const DeviceImage<float> &img,
    DeviceImage<float> &copy
    */
    const float *in, float *out, size_t w, size_t h, size_t s
    );

} // rmd namespace

#endif // RMD_TEST_COPY_CUH
