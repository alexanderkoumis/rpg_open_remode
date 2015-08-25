#include <rmd/device_image.cuh>

namespace rmd
{

__global__
void copyKernel(
    /*
    const DeviceImage<float> *in_dev_ptr,
    DeviceImage<float> *out_dev_ptr
        */
    const float *in, float *out, size_t w, size_t h, size_t s
    )
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= w
     || y >= h)
    return;


  // const DeviceImage<float>  &img = *in_dev_ptr;
  // DeviceImage<float> &copy = *out_dev_ptr;
  // copy(x, y) = img(x, y);
  // copy(x, y) = 1.0f;
  // copy.data[copy.stride*y+x] = img.data[img.stride*y+x];
  out[s*y+x] = in[s*y+x];

}

void copy(
    /*
    const DeviceImage<float> &img,
    DeviceImage<float> &copy */
    const float *in, float *out, size_t w, size_t h, size_t s
    )
{
  dim3 dim_block_;
  dim3 dim_grid_;
  dim_block_.x = 16;
  dim_block_.y = 16;
  dim_grid_.x = (w + dim_block_.x - 1)  / dim_block_.x;
  dim_grid_.y = (h + dim_block_.y - 1) / dim_block_.y;

  copyKernel<<<dim_grid_, dim_block_>>>(in, out, w, h, s);
  cudaDeviceSynchronize();
}

} // rmd namespace

