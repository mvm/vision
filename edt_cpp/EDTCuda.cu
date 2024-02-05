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
#if EDT_VERSION_ROW == 1
        FLOAT b = 1.0;
#endif

        for(int j = 1; j < h; j++) {
            unsigned int d = j*w + i;
            unsigned int d1 = d - w;

#if EDT_VERSION_ROW == 1
            if(in[d] != 0) {
                out[d] = out[d1] + b;
                b += 2.0;
            } else {
                out[d] = 0.0;
                b = 1.0;
            }
#else
            if(in[d] != 0 ) {
                out[d] = out[d1];
            } else {
                out[d] = j;
            }
#endif
        }

#if EDT_VERSION_ROW == 1
        b = 1.0;
#endif

        for(int j = h - 2 ; j >= 0; j--) {
            unsigned int d = j*w + i;
            unsigned int d1 = d + w;

#if EDT_VERSION_ROW == 1
            if(out[d] > out[d1]) {
                out[d] = out[d1] + b;
                b += 2.0;
            }
            if(out[d] == 0) {
                b = 1.0;
            }
#else
            if((out[d] - j)*(out[d] - j) > (out[d1] - j)*(out[d1] - j)) {
                out[d] = out[d1];
            }
#endif
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
#if EDT_VERSION_ROW == 1
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
#if EDT_VERSION_ROW == 1
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
#elif EDT_VERSION_ROW == 2

#if (EDT_VERSION_COL != 3) && (EDT_VERSION_COL != 1)
#error "EDT_VERSION_COL must be 3 or 1"
#endif

/* Calculate intersection point of bisecting line with the x axis. */
__device__ static FLOAT intersection_point(FLOAT x1, FLOAT y1, FLOAT x2, FLOAT y2, FLOAT row) {
    FLOAT y1p = row - y1;
    FLOAT y2p = row - y2;

    if(x1 == x2) {
        return INFINITY;
    }
    return (x2 - x1)/2 + (y2p + y1p) * (y2p - y1p) / 2 / (x2 - x1);
}

__device__ void find_closest(FLOAT *X, FLOAT *Y, FLOAT *Xout, FLOAT *Yout, unsigned int len, unsigned int row) {
    unsigned int i = 0, end = 0, iout = 0;
    if(X[0] == INFINITY) {
        while(X[end] == INFINITY && end < len) {
            end++;
        }
        while( i < end && i < len) {
            Xout[i] = X[end];
            Yout[i] = Y[end];
            i++;
        }
    }

    iout = i;

    while(1) {
        end = i+1;

        while(end < len && X[end] == INFINITY) {
            end++;
        }

        if(end >= len) {
            for(unsigned int k = iout; k < len; k++) {
                Xout[k] = X[i];
                Yout[k] = Y[i];
            }
            break;
        }

        FLOAT u = intersection_point(X[i], Y[i], X[end], Y[end], row);
        unsigned int j = iout;
        while(j < ceil(i + u) && j < len) {
            Xout[j] = X[i];
            Yout[j] = Y[i];
            j++;
        }
        
        i = end;
        iout = j;
    }
}

__device__ void stack_merge(FLOAT *X, FLOAT *Y, unsigned int len,
    unsigned int row, unsigned int start, unsigned int end) {
    unsigned int stack_end = (end - start)/2;

    for(unsigned int i = start + stack_end; i < len && i < end; i++) {
        while(X[i] == INFINITY && i < len && i < end)
            i++;
        if(i == len || i == end) break;

        unsigned j = i - 1;
        // skip possible empty points
        while(X[j] == INFINITY && j > 0 && j > start) {
            j--;
        }

        while(j > 0 && j > start) {
            FLOAT top_intersection = (FLOAT)j + intersection_point(X[j], Y[j], X[i], Y[i], row);
            unsigned int next_point = j - 1;
            // skip empty points between j and the next point
            while(X[next_point] == INFINITY && next_point > 0 && next_point > start) {
                next_point--;
            }
            if(next_point == 0 || next_point == start) break;

            // X[next_point] != INFINITY
            FLOAT first_intersection = (FLOAT)next_point + intersection_point(X[next_point], Y[next_point], 
                X[j], Y[j], row);
            
            if(ceil(first_intersection) > ceil(top_intersection)) {
                X[j] = INFINITY;
                Y[j] = INFINITY;
                j = next_point;
            } else {
                break;
           }
        }
    }
}

__global__ static void edt_row(FLOAT *in, FLOAT *out, FLOAT *X, FLOAT *Y, FLOAT *Xout, FLOAT *Yout, int w, int h) {
    // one block per row
    unsigned int row = blockIdx.x;
    unsigned int thread = threadIdx.x;

    X = &X[row*w];
    Y = &Y[row*w];
    // set up variables in shared memory
    // we receive an image with the y coordinate of the vertically closest
    // point in the variable `in`
    for(unsigned int i = thread; i < w; i += blockDim.x) {
        Y[i] = in[row*w + i];
        X[i] = i;
    }
    __syncthreads();

    // create initial stacks with 3 elements
    for(unsigned int i = thread * 3; i < w ; i += blockDim.x * 3) {
        unsigned int indexA = i;
        unsigned int indexB = i+1;
        unsigned int indexC = i+2;

        FLOAT u = intersection_point(X[indexA], Y[indexA], X[indexB], Y[indexB], row);
        FLOAT v = intersection_point(X[indexB], Y[indexB], X[indexC], Y[indexC], row);
        if(ceil(indexA + u) > ceil(indexB + v)) {
            X[indexB] = INFINITY;
            Y[indexB] = INFINITY;
        }
    }
    __syncthreads();

    // filter stacks by dominant interval

    for(unsigned int stack_size = 3*2; stack_size < 2*w; stack_size *= 2) {
        for(unsigned int start = stack_size*thread; start < w; start += blockDim.x * stack_size) {
            if(start < w) {
                stack_merge(X, Y, w, row, start, start + stack_size);
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // find closest point given filtered stacks
    
    if(thread == 0) {
        find_closest(X, Y, &Xout[row*w], &Yout[row*w], w, row);
    }
    __syncthreads();

    // set output to distance to computed closest point
    for(unsigned int i = thread; i < w; i += blockDim.x) {
        FLOAT xi = Xout[row*w + i];
        FLOAT yi = Yout[row*w + i];
        out[row*w + i] = sqrt((xi - i)*(xi - i) + (yi - row)*(yi - row));
    }
}
#else
#error "EDT_VERSION_ROW must be 1 or 2"
#endif
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
#if EDT_VERSION_ROW == 2
    d_Xout = NULL;
    d_Yout = NULL;
    d_X = NULL;
    d_Y = NULL;
#endif
#if EDT_ENABLE_ROW
    d_out_row = NULL;
#endif
}

void EDTCuda::enter() {
    // set cuda heap size limit to our memory requirements
    // for the rows operation

#if EDT_VERSION_ROW == 2
    unsigned long memory_limit = w*h*8*sizeof(FLOAT) + w*h*sizeof(uchar);
#else
    unsigned long memory_limit = w*h*(2*sizeof(FLOAT) + sizeof(uchar)) +
        (w+2)*(h)*(sizeof(unsigned int) + sizeof(FLOAT));
#endif

    std::cout << "Memory limit: " << memory_limit << " B" << std::endl;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, memory_limit);

    // allocate memory in device and copy input data to GPU
    cudaMalloc(&d_data, sizeof(uchar)*w*h);
    cudaMalloc(&d_out, sizeof(FLOAT)*w*h);

#if EDT_ENABLE_ROW
#if EDT_VERSION_ROW == 2
    cudaMalloc(&d_Xout, w*h*sizeof(FLOAT));
    cudaMalloc(&d_Yout, w*h*sizeof(FLOAT));
    cudaMalloc(&d_X, w*h*sizeof(FLOAT));
    cudaMalloc(&d_Y, w*h*sizeof(FLOAT));
#endif

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
#if EDT_VERSION_ROW == 2
    cudaFree(d_Xout);
    cudaFree(d_Yout);
    cudaFree(d_X);
    cudaFree(d_Y);
#endif
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
#if EDT_VERSION_ROW == 1
    edt_row<<<blocks, threads>>>(d_out, d_out_row, w, h);
#elif EDT_VERSION_ROW == 2
    dim3 threadsPerBlockRow(threads);
    dim3 blocksPerGridRow(h);

    edt_row<<<blocksPerGridRow,
        threadsPerBlockRow>>> (d_out, d_out_row, d_X, d_Y, d_Xout, d_Yout, w, h);
#else
#error "EDT_VERSION_ROW must be 1 or 2"
#endif
    cudaDeviceSynchronize();
#endif // EDT_ENABLE_ROW
}
