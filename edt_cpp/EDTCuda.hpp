#pragma once
#include <opencv2/opencv.hpp>

#define EDT_ENABLE_ROW 1 // deactivate row processing for debug
#define USE_DOUBLE 1 // used to change types easily
#define EDT_VERSION_COL 1 // can be 1, 2 or 3
#define EDT_VERSION_ROW 1 // can be 1 or 2. 2 can only be used with EDT_VERSION_COL = 3 or 1

#if USE_DOUBLE
#define FLOAT double
#define IMGTYPE CV_64FC1
#else
#define FLOAT float
#define IMGTYPE CV_32FC1
#endif

class EDTCuda {
public:
    EDTCuda(cv::Mat image, unsigned int blocks = 16,
        unsigned int threads = 128, unsigned int yblocks = 8);
    void enter();
    void run();
    cv::Mat leave();
private:
    cv::Mat image;
    unsigned int blocks, threads, yblocks;
    unsigned int w, h;
    uchar *data;

    uchar *d_data;
    FLOAT *d_out;
#if EDT_VERSION_ROW == 2
    FLOAT *d_Xout;
    FLOAT *d_Yout;
    FLOAT *d_X;
    FLOAT *d_Y;
#endif
#if EDT_ENABLE_ROW
    FLOAT *d_out_row;
#endif
};
