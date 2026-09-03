/*
 * Medical Gauze Thread Counter
 *
 * Sequential pipeline consuming raw 8-bit monochrome frames, so there is no JPEG
 * decode and no colour conversion. The 2D-DFT, evaluated on a 256x256
 * downscale of the binary, yields the warp and weft angles. The binary is
 * rotated and directionally eroded into vertical and horizontal masks, and
 * thread transitions are counted in the central band, scaled by the box
 * dimension to threads per 10 cm. Every stage is timed with std::chrono.
 *
 * Copyright (c) 2026 Salem Berbouchi
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <chrono>
#include <nlohmann/json.hpp>

using namespace cv;
using namespace std;
using json = nlohmann::json;

bool DEBUG_MODE = false;

class Timer {
    chrono::high_resolution_clock::time_point s;
public:
    Timer() : s(chrono::high_resolution_clock::now()) {}
    double ms() const {
        return chrono::duration<double, milli>(chrono::high_resolution_clock::now() - s).count();
    }
};

struct Angles {
    double warpAngleRot;   // rotation angle for warp (vertical) mask
    double weftAngleRot;   // rotation angle for weft (horizontal) mask
    double warpFreq;
    double weftFreq;
};

static Angles extractAngles(const Mat& gray) {
    const int FFT_SIDE = 256;
    Mat small;
    resize(gray, small, Size(FFT_SIDE, FFT_SIDE), 0, 0, INTER_AREA);

    Mat flt;
    small.convertTo(flt, CV_32F);

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

    int cx = mag.cols / 2;
    int cy = mag.rows / 2;
    Mat q0(mag, Rect(0, 0, cx, cy)), q1(mag, Rect(cx, 0, cx, cy));
    Mat q2(mag, Rect(0, cy, cx, cy)), q3(mag, Rect(cx, cy, cx, cy));
    Mat t;
    q0.copyTo(t); q3.copyTo(q0); t.copyTo(q3);
    q1.copyTo(t); q2.copyTo(q1); t.copyTo(q2);

    if (DEBUG_MODE) {
        Mat logM;
        log(mag + 1, logM);
        normalize(logM, logM, 0, 255, NORM_MINMAX);
        Mat vis;
        logM.convertTo(vis, CV_8U);
        applyColorMap(vis, vis, COLORMAP_JET);
        imwrite("2D_FFT_magnitude.jpeg", vis);
    }

    Mat magClust = mag.clone();
    Mat nroi(magClust, Rect(cx - 4, cy - 4, 7, 7));
    nroi.setTo(Scalar(0));
    GaussianBlur(magClust, magClust, Size(5, 5), 0, 0);
    normalize(magClust, magClust, 0, 255, NORM_MINMAX);
    threshold(magClust, magClust, 85, 255, THRESH_BINARY);

    Mat locations;
    findNonZero(magClust, locations);
    if (locations.rows < 2) {
        throw runtime_error("Not enough clusters found");
    }

    double warpFreq = mag.cols / 2.0;
    double weftFreq = mag.rows / 2.0;
    double warpAngle = 0.0;
    double weftAngle = 0.0;

    for (int i = 0; i < locations.rows; i++) {
        double dx = locations.at<Point>(i, 0).x - cx;
        double dy = cy - locations.at<Point>(i, 0).y;
        double angle = atan2(dy, dx) * 180.0 / M_PI;
        double distance = sqrt(dx * dx + dy * dy);
        if (distance < warpFreq && angle > -45.0 && angle < 45.0) {
            warpFreq = distance;
            warpAngle = angle;
        }
        if (distance < weftFreq && (angle > 45.0 && angle < 135.0)) {
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

static double detectThreadDensity(const Mat& image) {
    int startRow = static_cast<int>(round(image.rows * 0.25));
    int endRow = static_cast<int>(round(image.rows * 0.75));
    int startCol = static_cast<int>(round(image.cols * 0.25));
    int endCol = static_cast<int>(round(image.cols * 0.75));

    double acc = 0.0;
    for (int j = startCol; j < endCol; j++) {
        int transitions = 0;
        for (int i = startRow; i < endRow; i++) {
            uchar prev = image.at<uchar>(i - 1, j);
            uchar curr = image.at<uchar>(i, j);
            if (prev < curr) transitions++;
        }
        acc += transitions;
    }
    return acc / (endCol - startCol) / (endRow - startRow);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: ./cvengine <frame.raw> [--debug]\n"
                "  Expects an 8-bit 1024x1024 raw mono frame."
             << endl;
        return -1;
    }

    DEBUG_MODE = (argc > 2 && string(argv[2]) == "--debug");

    const int SIDE = 1024;
    double boxDim = 0.0;
    json steps;

    Mat pf;   // primary frame matrix (preprocessed binary / masks), reused
    Mat sf;   // scratch/secondary matrix, reused

    Timer t0;
    {
        Timer t;
        FILE* f = fopen(argv[1], "rb");
        if (!f) throw runtime_error("cannot open " + string(argv[1]));
        pf.create(SIDE, SIDE, CV_8UC1);
        size_t n = fread(pf.data, 1, (size_t)SIDE * SIDE, f);
        fclose(f);
        if (n != (size_t)SIDE * SIDE) throw runtime_error("short raw frame read");
        steps["raw_ingest"] = t.ms();
    }

    {
        Timer t;
        normalize(pf, pf, 0, 255, NORM_MINMAX);
        threshold(pf, pf, 127, 255, THRESH_BINARY);
        Mat k = getStructuringElement(MORPH_RECT, Size(2, 2));
        morphologyEx(pf, pf, MORPH_DILATE, k);
        morphologyEx(pf, pf, MORPH_OPEN, k);
        morphologyEx(pf, pf, MORPH_CLOSE, k);
        steps["preprocess"] = t.ms();
    }

    if (DEBUG_MODE) imwrite("binary.jpg", pf);

    Angles ang;
    {
        Timer t;
        ang = extractAngles(pf);
        steps["extractAngles"] = t.ms();
    }

    Rect roi;
    double maxArea = 0.0;
    {
        Timer t;
        Mat inv;
        bitwise_not(pf, inv);
        Mat k = getStructuringElement(MORPH_RECT, Size(7, 7));
        morphologyEx(inv, inv, MORPH_CLOSE, k);
        morphologyEx(inv, inv, MORPH_OPEN, k);
        vector<vector<Point>> contours;
        findContours(inv, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (const auto& c : contours) {
            double area = contourArea(c);
            if (area > maxArea) { maxArea = area; roi = boundingRect(c); }
        }
        boxDim = sqrt(maxArea);
        steps["detectROI"] = t.ms();
    }

    Mat rotWarp;
    Mat rotWeft;
    {
        Timer t;
        Point2f center(pf.cols / 2.0, pf.rows / 2.0);

        Mat M1 = getRotationMatrix2D(center, ang.warpAngleRot, 1.0);
        double ar1 = ang.warpAngleRot * CV_PI / 180.0;
        int nw1 = round(pf.cols * abs(cos(ar1)) + pf.rows * abs(sin(ar1)));
        int nh1 = round(pf.cols * abs(sin(ar1)) + pf.rows * abs(cos(ar1)));
        M1.at<double>(0, 2) += (nw1 / 2.0) - center.x;
        M1.at<double>(1, 2) += (nh1 / 2.0) - center.y;
        warpAffine(pf, rotWarp, M1, Size(nw1, nh1), INTER_NEAREST, BORDER_CONSTANT, Scalar(255));

        Mat M2 = getRotationMatrix2D(center, ang.weftAngleRot, 1.0);
        double ar2 = ang.weftAngleRot * CV_PI / 180.0;
        int nw2 = round(pf.cols * abs(cos(ar2)) + pf.rows * abs(sin(ar2)));
        int nh2 = round(pf.cols * abs(sin(ar2)) + pf.rows * abs(cos(ar2)));
        M2.at<double>(0, 2) += (nw2 / 2.0) - center.x;
        M2.at<double>(1, 2) += (nh2 / 2.0) - center.y;
        warpAffine(pf, rotWeft, M2, Size(nw2, nh2), INTER_NEAREST, BORDER_CONSTANT, Scalar(255));
        steps["rotate"] = t.ms();
    }

    if (DEBUG_MODE) {
        imwrite("rotated_binary_warp.jpg", rotWarp);
        imwrite("rotated_binary_weft.jpg", rotWeft);
    }

    {
        Timer t;
        morphologyEx(rotWarp, rotWarp, MORPH_ERODE, getStructuringElement(MORPH_RECT, Size(1, 6)));
        morphologyEx(rotWeft, rotWeft, MORPH_ERODE, getStructuringElement(MORPH_RECT, Size(6, 1)));
        steps["erode_masks"] = t.ms();
    }

    if (DEBUG_MODE) {
        imwrite("vertical_mask.jpg", rotWarp);
        imwrite("horizontal_mask.jpg", rotWeft);
    }

    int warpCount, weftCount;
    {
        Timer t;
        double wden = detectThreadDensity(rotWarp.t());
        double hden = detectThreadDensity(rotWeft);
        warpCount = static_cast<int>(round(wden * boxDim));
        weftCount = static_cast<int>(round(hden * boxDim));
        steps["count"] = t.ms();
    }

    double total = t0.ms();

    if (DEBUG_MODE) {
        json out;
        out["image"] = argv[1];
        out["size"] = {pf.cols, pf.rows};
        out["warp_threads"] = warpCount;
        out["weft_threads"] = weftCount;
        out["warp_angle_rot"] = ang.warpAngleRot;
        out["weft_angle_rot"] = ang.weftAngleRot;
        out["warp_freq"] = ang.warpFreq;
        out["weft_freq"] = ang.weftFreq;
        out["box_dim"] = boxDim;
        out["steps_ms"] = steps;
        out["total_ms"] = total;
        cout << out.dump(2) << endl;
    } else {
        json out;
        out["warp_threads"] = warpCount;
        out["weft_threads"] = weftCount;
        cout << out.dump(2) << endl;
    }
    return 0;
}
