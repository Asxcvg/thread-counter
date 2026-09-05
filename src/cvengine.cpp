// Medical Gauze Thread Counter
//
// Consumes raw 8-bit monochrome frames (no JPEG decode, no colour conversion).
// The gauze region of interest (ROI) is located by contour analysis and
// cropped; a 2D-DFT of the cropped binary downscaled to 256x256 recovers the
// warp and weft angles. The cropped binary is rotated and directionally eroded
// into vertical and horizontal masks, and thread transitions are counted along
// full-height profiles in each direction.
//
// Copyright (c) 2026 Salem Berbouchi
// SPDX-License-Identifier: Apache-2.0

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <cstdio>

using namespace cv;
using namespace std;

bool DEBUG_MODE = false;

struct Angles {
    double warpAngleRot;   // rotation angle that makes warp threads vertical
    double weftAngleRot;   // rotation angle that makes weft threads horizontal
    double warpFreq;
    double weftFreq;
};

struct ROI {
    Rect rect;
    double boxDim;   // physical sample side length in pixels (10 cm assumed)
};

static Mat readRawFrame(const char* path, int side) {
    FILE* f = fopen(path, "rb");
    if (!f) throw runtime_error("cannot open " + string(path));

    Mat frame(side, side, CV_8UC1);
    size_t n = fread(frame.data, 1, (size_t)side * side, f);
    fclose(f);

    if (n != (size_t)side * side) throw runtime_error("short raw frame read");
    return frame;
}

static Mat preprocess(const Mat& frame) {
    Mat binary;
    normalize(frame, binary, 0, 255, NORM_MINMAX);
    threshold(binary, binary, 127, 255, THRESH_BINARY);

    Mat k = getStructuringElement(MORPH_RECT, Size(2, 2));
    morphologyEx(binary, binary, MORPH_DILATE, k);
    morphologyEx(binary, binary, MORPH_OPEN, k);
    morphologyEx(binary, binary, MORPH_CLOSE, k);
    return binary;
}

static void quadrantShift(Mat& mag) {
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;
    Mat q0(mag, Rect(0, 0, cx, cy)), q1(mag, Rect(cx, 0, cx, cy));
    Mat q2(mag, Rect(0, cy, cx, cy)), q3(mag, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}

static Mat fftMagnitude(const Mat& input) {
    Mat flt;
    input.convertTo(flt, CV_32F);

    Mat padded;
    int rows = getOptimalDFTSize(flt.rows);
    int cols = getOptimalDFTSize(flt.cols);
    copyMakeBorder(flt, padded, 0, rows - flt.rows, 0, cols - flt.cols,
                   BORDER_CONSTANT, Scalar::all(1));

    Mat window;
    createHanningWindow(window, padded.size(), CV_32F);
    multiply(padded, window, padded);

    Mat complex;
    dft(padded, complex, DFT_COMPLEX_OUTPUT);

    Mat planes[2];
    split(complex, planes);

    Mat mag;
    magnitude(planes[0], planes[1], mag);
    normalize(mag, mag, 0, 255, NORM_MINMAX);
    quadrantShift(mag);
    return mag;
}

static Angles extractAngles(const Mat& binary) {
    const int FFT_SIDE = 256;
    Mat small;
    resize(binary, small, Size(FFT_SIDE, FFT_SIDE), 0, 0, INTER_AREA);

    Mat mag = fftMagnitude(small);

    if (DEBUG_MODE) {
        Mat logM;
        log(mag + 1, logM);
        normalize(logM, logM, 0, 255, NORM_MINMAX);
        Mat vis;
        logM.convertTo(vis, CV_8U);
        applyColorMap(vis, vis, COLORMAP_JET);
        imwrite("images/2D_FFT_magnitude.jpeg", vis);
    }

    int cx = mag.cols / 2;
    int cy = mag.rows / 2;

    Mat clusterMap = mag.clone();
    Mat center(clusterMap, Rect(cx - 4, cy - 4, 7, 7));
    center.setTo(Scalar(0));                       // suppress DC spike
    GaussianBlur(clusterMap, clusterMap, Size(5, 5), 0, 0);
    normalize(clusterMap, clusterMap, 0, 255, NORM_MINMAX);
    threshold(clusterMap, clusterMap, 85, 255, THRESH_BINARY);

    Mat locations;
    findNonZero(clusterMap, locations);
    if (locations.rows < 2) {
        throw runtime_error("Not enough clusters found");
    }

    double warpFreq = mag.cols / 2.0;
    double weftFreq = mag.rows / 2.0;
    double warpAngle = 0.0;
    double weftAngle = 0.0;

    for (int i = 0; i < locations.rows; i++) {
        Point p = locations.at<Point>(i, 0);
        double dx = p.x - cx;
        double dy = cy - p.y;
        double angle = atan2(dy, dx) * 180.0 / M_PI;
        double distance = sqrt(dx * dx + dy * dy);

        if (distance < warpFreq && angle > -45.0 && angle < 45.0) {
            warpFreq = distance;
            warpAngle = angle;
        }
        if (distance < weftFreq && angle > 45.0 && angle < 135.0) {
            weftFreq = distance;
            weftAngle = angle;
        }
    }

    Angles r;
    r.warpAngleRot = -warpAngle;
    r.weftAngleRot = -(weftAngle - 90.0);
    r.warpFreq = warpFreq;
    r.weftFreq = weftFreq;
    return r;
}

static ROI detectROI(const Mat& binary) {
    Mat region;
    bitwise_not(binary, region);

    Mat k = getStructuringElement(MORPH_RECT, Size(7, 7));
    morphologyEx(region, region, MORPH_CLOSE, k);
    morphologyEx(region, region, MORPH_OPEN, k);

    vector<vector<Point>> contours;
    findContours(region, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    ROI roi;
    double maxArea = 0.0;
    for (const auto& c : contours) {
        double area = contourArea(c);
        if (area > maxArea) {
            maxArea = area;
            roi.rect = boundingRect(c);
        }
    }
    roi.boxDim = sqrt(maxArea);
    return roi;
}

static Mat rotateBinary(const Mat& binary, double degrees) {
    Point2f center(binary.cols / 2.0, binary.rows / 2.0);
    Mat M = getRotationMatrix2D(center, degrees, 1.0);

    double rad = degrees * CV_PI / 180.0;
    int newWidth = round(binary.cols * abs(cos(rad)) + binary.rows * abs(sin(rad)));
    int newHeight = round(binary.cols * abs(sin(rad)) + binary.rows * abs(cos(rad)));

    M.at<double>(0, 2) += (newWidth / 2.0) - center.x;
    M.at<double>(1, 2) += (newHeight / 2.0) - center.y;

    Mat out;
    warpAffine(binary, out, M, Size(newWidth, newHeight),
               INTER_NEAREST, BORDER_CONSTANT, Scalar(255));
    return out;
}

static Mat directionalErode(const Mat& mask, bool horizontal) {
    Size element = horizontal ? Size(6, 1) : Size(1, 6);
    Mat out;
    morphologyEx(mask, out, MORPH_ERODE, getStructuringElement(MORPH_RECT, element));
    return out;
}

static double detectThreadCount(const Mat& mask) {
    int startCol = static_cast<int>(round(mask.cols * 0.25));
    int endCol = static_cast<int>(round(mask.cols * 0.75));

    double acc = 0.0;
    for (int col = startCol; col < endCol; col++) {
        int transitions = 0;
        for (int row = 1; row < mask.rows; row++) {
            uchar prev = mask.at<uchar>(row - 1, col);
            uchar curr = mask.at<uchar>(row, col);
            if (prev < curr) transitions++;
        }
        acc += transitions;
    }
    return acc / (endCol - startCol);
}

static void printJsonString(ostream& os, const string& s) {
    os << '"';
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '"' || s[i] == '\\') os << '\\';
        os << s[i];
    }
    os << '"';
}

static void printJSON(int warpCount, int weftCount) {
    cout << "{\n"
         << "  \"warp_threads\": " << warpCount << ",\n"
         << "  \"weft_threads\": " << weftCount << "\n"
         << "}" << endl;
}

static void printDebugJSON(const char* path, const Angles& angles,
                           int warpCount, int weftCount, double boxDim) {
    cout << "{\n"
         << "  \"image\": ";
    printJsonString(cout, path);
    cout << ",\n"
         << "  \"warp_threads\": " << warpCount << ",\n"
         << "  \"weft_threads\": " << weftCount << ",\n"
         << "  \"warp_angle_rot\": " << angles.warpAngleRot << ",\n"
         << "  \"weft_angle_rot\": " << angles.weftAngleRot << ",\n"
         << "  \"warp_freq\": " << angles.warpFreq << ",\n"
         << "  \"weft_freq\": " << angles.weftFreq << ",\n"
         << "  \"box_dim\": " << boxDim << "\n"
         << "}" << endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: ./cvengine <frame.raw> [--debug]\n"
             << "  Expects an 8-bit 1024x1024 raw mono frame." << endl;
        return -1;
    }

    DEBUG_MODE = (argc > 2 && string(argv[2]) == "--debug");

    const int SIDE = 1024;

    Mat frame = readRawFrame(argv[1], SIDE);
    Mat binary = preprocess(frame);
    if (DEBUG_MODE) imwrite("images/binary.jpg", binary);

    ROI roi = detectROI(binary);
    Mat roiImage = binary(roi.rect);

    Angles angles = extractAngles(roiImage);

    Mat warpRotated = rotateBinary(roiImage, angles.warpAngleRot);
    Mat weftRotated = rotateBinary(roiImage, angles.weftAngleRot);
    if (DEBUG_MODE) {
        imwrite("images/rotated_binary_warp.jpg", warpRotated);
        imwrite("images/rotated_binary_weft.jpg", weftRotated);
    }

    Mat warpMask = directionalErode(warpRotated, false);
    Mat weftMask = directionalErode(weftRotated, true);
    if (DEBUG_MODE) {
        imwrite("images/vertical_mask.jpg", warpMask);
        imwrite("images/horizontal_mask.jpg", weftMask);
    }

    int warpCount = static_cast<int>(round(detectThreadCount(warpMask.t())));
    int weftCount = static_cast<int>(round(detectThreadCount(weftMask)));;

    if (DEBUG_MODE)
        printDebugJSON(argv[1], angles, warpCount, weftCount, roi.boxDim);
    else
        printJSON(warpCount, weftCount);

    return 0;
}