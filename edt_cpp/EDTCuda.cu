#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>

#include "EDTCuda.hpp"

#if EDT_VERSION_COL == 1
// kernel for one dimensional EDT (calculates columns)
__global__ static void edt_col(uchar *in, FLOAT *out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int step = gridDim.x * blockDim.x;
    // one thread per column
    for(int i = x; i < w; i += step) {
        FLOAT b = 1.0;

        for(int j = 1; j < h; j++) {
            unsigned int d = j*w + i;
            unsigned int d1 = d - w;

            if(in[d] != 0) {
                out[d] = out[d1] + b;
                b += 2.0;
            } else {
                out[d] = 0.0;
                b = 1.0;
            }
        }

        b = 1.0;
        for(int j = h - 2 ; j >= 0; j--) {
            unsigned int d = j*w + i;
            unsigned int d1 = d + w;

            if(out[d] > out[d1]) {
                out[d] = out[d1] + b;
                b += 2.0;
            }
            if(out[d] == 0) {
                b = 1.0;
            }
        }
    }
}
#elif EDT_VERSION_COL == 2
__global__ static void edt_col(uchar *in, FLOAT *out, int w, int h) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y;

    unsigned int untilPixel = fminf(x + (w / blockDim.x), h);
    extern __shared__ FLOAT imgCol[];
    for(unsigned int row = threadIdx.x; row < h /* - blockDim.x */; row += blockDim.x) {
        imgCol[row] = in[row*w + y] > 127 ? INFINITY : 0;
    }
    __syncthreads(); // copies column to shared memory correctly

    for(unsigned int row = x; row < h; row += gridDim.x * blockDim.x) {
        FLOAT value = imgCol[row];

        // TODO: see if I can do this faster.

        FLOAT d = 1;
        for(unsigned int row_i = 1; row_i < h - row ; row_i++) {
            value = fminf(value, imgCol[row + row_i] + d);
            d += 1 + 2*row_i;
        }

        d = 1;
        for(unsigned int row_i = 1; row_i < row; row_i++) {
            value = fminf(value, imgCol[row - row_i] + d);
            d += 1 + 2 * row_i;
        }

        out[row*w + y] = value;
    }
}
#elif EDT_VERSION_COL == 3
#define TEST 1
__global__ static void edt_col(uchar *in, FLOAT *out, int w, int h) {
    unsigned int col = blockIdx.x;
    unsigned int thread = threadIdx.x;

    extern __shared__ FLOAT V[];

    // Copy column to shared memory
    for(unsigned int i = thread; i < h; i += blockDim.x) {
        V[i] = INFINITY;
        if(in[col + i*w] == 0) {
            V[i] = i;
        }
    }
    __syncthreads();

    // reduction phase
    unsigned int treeLimit = ceil(log2f(h) - 1);

    for(unsigned int i = 0; i < treeLimit; i++) {
        unsigned int step = 1 << (i + 1);

        for(unsigned int j = thread; j < h; j += blockDim.x) {
            unsigned int prevStep = 1 << i;
            unsigned int from = j + prevStep - 1;
            unsigned int dest = j + step - 1;

            if(dest < h && V[dest] == INFINITY && V[from] != INFINITY)
                V[dest] = V[from];
        }
        __syncthreads();
    }

    __syncthreads();

    // down-sweep phase
    treeLimit = ceil(log2f(h) - 2);
    for(unsigned int i = treeLimit; i > 0; i--) {
        unsigned int step = 1 << (i + 1);
        for(unsigned int j = thread; j < h; j += blockDim.x) {
            unsigned int prevStep = 1 << i;
            unsigned int dest = j + 3*prevStep - 1;
            unsigned int from = j + step - 1;
            if(dest < h && V[dest] == INFINITY && V[from] != INFINITY) {
                V[dest] = V[from];
            }
        }
        __syncthreads();
    }

    __syncthreads();

    treeLimit = ceil(log2f(h) - 1);
    for(unsigned int i = 0; i < treeLimit; i++) {
        unsigned int step = 1 << (i+1);

        for(int j = h - threadIdx.x - 1; j >= 0; j -= blockDim.x) {
            unsigned int prevStep = 1 << i;
            int from = j + step - 1;
            int dest = j + prevStep - 1;
            if(from >= 0 && from < h && dest >= 0 && dest < h &&
                (V[from] - j)*(V[from] - j) < (V[dest] - j)*(V[dest] - j)) {
                    V[dest] = V[from];
                }
        }
        __syncthreads();
    }

    __syncthreads();

    // down-sweep phase
    treeLimit = ceil(log2f(h) - 2);
    for(int i = treeLimit; i > 0; i--) {
        unsigned int step = 1 << (i + 1);
        for(int j = h - thread - 1; j >= 0; j -= blockDim.x) {
            unsigned int prevStep = 1 << i;
            int from = j + 3*prevStep - 1;
            int dest = j + step - 1;
            if(dest >= 0 && from >= 0 && dest < h && from < h 
                && abs(V[from] - j) < abs(V[dest] - j)) {
                V[dest] = V[from];
            }
        }
        __syncthreads();
    }

    __syncthreads();

    // Copỳ back shared memory to output image
    for(unsigned int i = thread; i < h; i += blockDim.x) {
#if TEST == 1
        // Convert closest site index to distance from closest site squared
        out[col + i*w] = (i - V[i])*(i - V[i]);
#else
        out[col + i*w] = V[i];
#endif // TEST
    }
}
#else
#error "EDT_VERSION_COL must be 1, 2 or 3"
#endif /* EDT_VERSION */

#if EDT_ENABLE_ROW
// rows step of the distance transform
__global__ static void edt_row(FLOAT *in, FLOAT *out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int *v = nullptr;
    FLOAT *z = nullptr;

    if(x < h) {
        v = (unsigned int*)malloc(sizeof(unsigned int)*(w+2));
        z = (FLOAT*)malloc(sizeof(FLOAT)*(w+2));
    }

    unsigned int step = gridDim.x * blockDim.x;

    // one thread per row
    for(int i = x; i < h; i += step) {
        FLOAT *f = in + i*w;
        FLOAT *y = out + i*w;

        if(v == nullptr || z == nullptr) {
            break;
        }

        int k = 0;
        v[0] = 0;
        z[0] = -INFINITY;
        z[1] = INFINITY;

        for(int j = 0; j < w; j++) {
            FLOAT s = 0;

            while(
                (s = ((f[j] + j*j) - (f[v[k]] + v[k]*v[k]))/(2*j - 2*v[k]))
                <= z[k]
            ) {
                k = k - 1;
            }

            if(s > z[k]) {
                k = k + 1;
                v[k] = j;
                z[k] = s;
                z[k+1] = INFINITY;
            }
        }

        k = 0;
        for(int j = 0; j < w; j++) {
            while(z[k+1] < j) {
                k = k + 1;
            }

            y[j] = sqrt((j - v[k])*(j - v[k]) + f[v[k]]);
        }
    }
    if (v != nullptr) free(v);
    if(z != nullptr) free(z);
}
#endif /* EDT_ENABLE_ROW */

EDTCuda::EDTCuda(cv::Mat image, unsigned int blocks, unsigned int threads,
    unsigned int yblocks) : blocks(blocks),
    threads(threads), yblocks(yblocks) {
    CV_Assert(image.isContinuous());
    this->image = image;
    this->w = this->image.cols;
    this->h = this->image.rows;
    this->data = image.data;
    d_data = NULL;
    d_out = NULL;
#if EDT_ENABLE_ROW
    d_out_row = NULL;
#endif
}

void EDTCuda::enter() {
    // set cuda heap size limit to our memory requirements
    // for the rows operation

    unsigned long memory_limit = w*h*(2*sizeof(FLOAT) + sizeof(uchar)) +
        (w+2)*(h)*(sizeof(unsigned int) + sizeof(FLOAT));

    std::cout << "Memory limit: " << memory_limit << " B" << std::endl;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, memory_limit);

    // allocate memory in device and copy input data to GPU
    cudaMalloc(&d_data, sizeof(uchar)*w*h);
    cudaMalloc(&d_out, sizeof(FLOAT)*w*h);

#if EDT_ENABLE_ROW
    cudaMalloc(&d_out_row, sizeof(FLOAT)*w*h);
#endif
    cudaMemcpy(d_data, data, sizeof(uchar)*w*h, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

cv::Mat EDTCuda::leave() {
    if(d_out == NULL) {
        throw std::runtime_error("d_out is a NULL pointer");
    }

#if EDT_ENABLE_ROW
    if(d_out_row == NULL) {
        throw std::runtime_error("d_out_row is a NULL pointer");
    }
#endif

    cv::Mat out = cv::Mat(h, w, IMGTYPE); // float 64 bits, 1 channel
    FLOAT *data_out = (FLOAT *)out.data;

    // postprocesado
#if EDT_ENABLE_ROW
    cudaMemcpy(data_out, d_out_row, sizeof(FLOAT)*w*h, cudaMemcpyDeviceToHost);
#else
    cudaMemcpy(data_out, d_out, sizeof(FLOAT)*w*h, cudaMemcpyDeviceToHost);
#endif

    cudaFree(d_out);
#if EDT_ENABLE_ROW
    cudaFree(d_out_row);
#endif
    cudaFree(d_data);

#if !EDT_ENABLE_ROW
    cv::sqrt(out, out);
#endif
    return out;
}

void EDTCuda::run() {
#if EDT_VERSION_COL == 1
    edt_col<<<blocks, threads>>>(d_data, d_out, w, h);
#elif EDT_VERSION_COL == 2
    dim3 threadsPerBlock(threads, 1, 1);
    dim3 blocksPerGrid(blocks, w, 1);

    edt_col<<<blocksPerGrid, threadsPerBlock, sizeof(FLOAT)*h>>>(d_data, d_out, w, h);
#elif EDT_VERSION_COL == 3
    dim3 threadsPerBlock(threads);
    dim3 blocksPerGrid(w);

    edt_col<<<blocksPerGrid, threadsPerBlock, sizeof(FLOAT)*h>>>(d_data, d_out, w, h);
#else
#error "EDT_VERSION_COL must be 1, 2 or 3"
#endif
    cudaDeviceSynchronize();

#if EDT_ENABLE_ROW
    edt_row<<<blocks, threads>>>(d_out, d_out_row, w, h);
    cudaDeviceSynchronize();
#endif // EDT_ENABLE_ROW
}
