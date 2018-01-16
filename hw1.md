# 彩色图===>灰度图

主要知识点：

1. 如何写一个内核程序（Map的运用）

## 1. 内核部分

```c
// rgb图像-->灰度图像 内核部分
__global__ void rgba_to_greyscale(const uchar4 *const rgbaImage,
                       unsigned char *const greyImage,
                       int numRows, int numCols) {
	// 对图片中的每个像素点分配一个线程来进行转换操作
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (y < numCols && x < numRows) {
        int index = numRows * y + x;
        uchar4 color = rgbaImage[index];
        unsigned char grey = (unsigned char) (0.299f * color.x + 0.587f * color.y + 0.114f * color.z);
        greyImage[index] = grey;
    }
}
```

## 2. 一些细节

### 2.1 CUDA计时相关函数

在头文件`<cuda_runtime.h>`

```c
struct GpuTimer {
    cudaEvent_t start;    // cuda事件类型
    cudaEvent_t stop;
    GpuTimer()   // 构造函数
    {
        cudaEventCreate(&start);  // 新建一个事件对象
        cudaEventCreate(&stop);
    }
    ~GpuTimer()  // 析构函数
    {
        cudaEventDestroy(start);  // 破坏一个事件对象
        cudaEventDestroy(stop);
    }
    void Start() {
        cudaEventRecord(start, 0);  // 记录一个事件(0代表哪个stream上记录)
    }
    void Stop() {
        cudaEventRecord(stop, 0);
    }
    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);  // 等待事件完成
        cudaEventElapsedTime(&elapsed, start, stop);  // 获取start和stop之间的时间差
        return elapsed;
    }
};
```

### 2.2 释放指针对应的空间

针对`cv::Mat`对象，一般无需显式地释放空间，但`cuda`分配的空间就必须要在程序运行完后清除掉。一般采用的方式：令一个全局变量(指针)也指向该空间，最后释放这个全局变量(指针)。

```c
uchar4 *d_rgbaImage__;
unsigned char *d_greyImage__;

void cleanup() {
    //cleanup
    cudaFree(d_rgbaImage__);
    cudaFree(d_greyImage__);
}
```

