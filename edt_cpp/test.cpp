#include <iostream>
#include <opencv2/opencv.hpp>

#include "EDTCuda.hpp"

using namespace cv;

int main(int argc, char **argv) {
    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [file]" << std::endl;
        return -1;
    }

    Mat image;
    image = imread(argv[1], IMREAD_GRAYSCALE); // read image as grayscale
    if(image.data == nullptr) {
        std::cerr << "Couldn't read image: " << argv[1] << std::endl;
        return -1;
    }

    CV_Assert(image.isContinuous());

    std::cout << "Image w=" << image.cols << " h=" << image.rows << std::endl;

    /*
    namedWindow("Original image", WINDOW_NORMAL);
    imshow("Original image", image);
    waitKey(0);
    destroyAllWindows();
    */
    // convert image to binary
    Mat binary_image;
    threshold(image, binary_image, 127, 255, THRESH_BINARY);

    // calculate euclidean distance transform
    Mat out;
    EDTCuda edt = EDTCuda(binary_image);
    edt.enter();

    struct timespec t_before, t_after;
    // procesado
    clock_gettime(CLOCK_REALTIME, &t_before);
    edt.run();
    clock_gettime(CLOCK_REALTIME, &t_after);
    unsigned long ns;
    ns = (t_after.tv_sec - t_before.tv_sec)*1e9 + (t_after.tv_nsec - t_before.tv_nsec);
    std::cout << "Time: " << ns / 1e9 << " s" << std::endl;

    out = edt.leave();

    // convert a matrix of doubles into something we can show
    // in a window, in this case 8-bit unsigned grayscale
    Mat showable = out;
    double max;
    minMaxLoc(showable, NULL, &max);
    showable *= 255.0/max;
    Mat distance;
    showable.convertTo(distance, CV_8UC1);

    imwrite("edt.png", distance);
    return 0;
}
