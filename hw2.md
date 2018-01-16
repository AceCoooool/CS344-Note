# 高斯核模糊图像

主要知识点：

1. 模板的使用

## 1. 内核函数

### 1.1 拆分通道内核

```c
__global__ void separateChannels(const uchar4 *const inputImageRGBA, int numRows, 
                      int numCols, unsigned char *const redChannel,
                      unsigned char *const greenChannel, unsigned char *const blueChannel) {
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x;

    if (p.x >= numCols || p.y >= numRows) // 解决超出边界的情况
        return;
    redChannel[m] = inputImageRGBA[m].x;
    greenChannel[m] = inputImageRGBA[m].y;
    blueChannel[m] = inputImageRGBA[m].z;
}
```

说明：主要将RGB三色通道单独分离开来，从而使模糊核分别作用在三个通道上

### 1.2 模糊操作内核

```c
__global__ void gaussian_blur(const unsigned char *const inputChannel,
                   unsigned char *const outputChannel, int numRows, int numCols,
                   const float *const filter, const int filterWidth) {
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x;  // 二维转一维

    if (p.x >= numCols || p.y >= numRows)
        return;

    float color = 0.0f;

    for (int f_y = 0; f_y < filterWidth; f_y++) {
        for (int f_x = 0; f_x < filterWidth; f_x++) {

            int c_x = p.x + f_x - filterWidth / 2;  // 以(p.x, p.y)为中心点
            int c_y = p.y + f_y - filterWidth / 2;
            c_x = min(max(c_x, 0), numCols - 1);
            c_y = min(max(c_y, 0), numRows - 1);
            float filter_value = filter[f_y * filterWidth + f_x];
            color += filter_value * static_cast<float>(inputChannel[c_y * numCols + c_x]);

        }
    }

    outputChannel[m] = color;
}
```

### 1.3 通道合并内核

```c
__global__ void recombineChannels(const unsigned char *const redChannel,
                       const unsigned char *const greenChannel,
                       const unsigned char *const blueChannel,
                       uchar4 *const outputImageRGBA, int numRows, int numCols) {
    const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                         blockIdx.y * blockDim.y + threadIdx.y);

    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    //make sure we don't try and access memory outside the image
    //by having any threads mapped there return early
    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        return;

    unsigned char red = redChannel[thread_1D_pos];
    unsigned char green = greenChannel[thread_1D_pos];
    unsigned char blue = blueChannel[thread_1D_pos];

    //Alpha should be 255 for no transparency
    uchar4 outputPixel = make_uchar4(red, green, blue, 255);

    outputImageRGBA[thread_1D_pos] = outputPixel;
}
```

## 2. 补充

1. 当执行多个内核时，在运行完每个内核之后，在其后面加上

   ```c
   cudaDeviceSynchronize();   // 检查是否同步完成
   checkCudaErrors(cudaGetLastError());
   ```

   ​

