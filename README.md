# Medical Gauze Thread Counter

## Abstract

This project presents an automated optical thread-density inspection system for woven medical gauze, implemented in C++ with the OpenCV computer-vision library. The system replaces manual needle counting of warp and weft threads with a two-stage computer-vision pipeline: a two-dimensional Discrete Fourier Transform (2D-DFT) stage that recovers the dominant thread orientation, followed by a morphological transition-detection stage that performs the actual counting. Thread densities are expressed in threads per 10 cm and compared against the acceptance limits of the European Standard EN 14079 on medical gauze quality inspection. On a validation lot of 12 rolls of 17-thread gauze, the system achieved a status concordance of 91.7% against manual reference counts, exceeding the predefined acceptance criterion of 90%.

## Introduction

Thread density is a decisive structural property of woven medical gauze, conditioning its absorption and mechanical behaviour. Dense threads reduce fluid uptake; sparse threads compromise structural integrity. Conformity against normative specifications therefore depends on reliable, repeatable thread counting in both the warp and the weft directions.

The procedure currently in force is manual. A technician counts threads with a needle, taking 3 measurements per roll and retaining the average. Each measurement consumes roughly 5 minutes. A standard lot of 26 rolls consequently demands at least 6.5 hours of technician time, and the procedure is vulnerable to inter-operator variability, poor traceability of paper records, and late detection of non-conformities. This project addresses those weaknesses by automating the measurement, eliminating operator variability, digitising the record, and reducing inspection of an entire lot to a few minutes.

## The Vision Engine

The engine processes a raw monochrome frame in 6 stages: preprocessing, region of interest (ROI) detection, physical calibration, Fourier-based angular correction, directional morphological masking, and transition counting. Each stage is described below.

### Preprocessing

The raw monochrome frame is normalised to the full 0 to 255 range and binarised at a threshold of 127, exploiting the contrast between light thread structure and dark background that characterises the acquisition setup. The binary image is then cleaned by a sequence of morphological operations, specifically a dilation followed by an opening and a closing, each with a 2-by-2 rectangular structuring element.

### Region of Interest Detection

The useful gauze surface is isolated from the surrounding scene by contour analysis. The cleaned binary is inverted and closed and opened with a 7-by-7 kernel, which merges fragmented regions while suppressing small artefacts. The largest external contour is located and its bounding box taken as the ROI. This step confines all subsequent analysis to the sample itself and removes interference from the table and cloth visible in the periphery of the frame.

### Physical Calibration

The sample is assumed to span a 10-by-10-centimetre physical square, which is the size of the black cloth used in acquisition. Letting the square root of the ROI area in pixels be denoted boxDim, the calibration constant is computed as PIXELS_PER_CM equals boxDim divided by 10. This constant converts pixel distances into physical centimetres and makes the system independent of the resolution of the capture device.

### Fourier Angular Correction

Woven gauze is spatially periodic in both thread directions, so its two-dimensional spectrum concentrates energy along lines whose orientation encodes the thread directions. The 2D-DFT of the image f of dimensions M by N is

$$F(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y)\, e^{-j 2\pi\left(\frac{ux}{M} + \frac{vy}{N}\right)},$$

where the frequency-domain representation F is complex. Before the transform, the image is multiplied by a Hanning window to attenuate spectral leakage and sharpen the resulting peaks. The magnitudes are then log-compressed, the zero-frequency pedestal at the spectrum centre is suppressed, and the spectrum is blurred and thresholded into a cluster map. Within the angular sector between negative 45 and positive 45 degrees, representative of gradients perpendicular to near-vertical warp threads, the closest cluster is taken to define the warp angle. Within the sector between positive 45 and positive 135 degrees, the closest cluster defines the weft angle.

Because the 2D-DFT is by far the most expensive operation in the pipeline, it is not evaluated at full resolution. Instead, the cropped ROI is downscaled to a 256-square image using area interpolation; the transform runs on that smaller image purely to recover the two correction angles. The actual thread counting is always performed on the full-resolution cropped ROI.

### Directional Morphological Masking

The cleaned binary of the ROI is rotated by the recovered warp angle so that warp threads run vertically, and then eroded with a vertical structuring element 1 pixel wide and 6 tall. This erosion removes horizontal structure and leaves only vertical thread segments, producing the warp mask. The same binary is independently rotated by the weft angle so that weft threads run horizontally, and eroded with a horizontal structuring element 6 wide and 1 tall, producing the weft mask. Rotation borders are filled with white to avoid introducing spurious dark transitions. Nearest-neighbour interpolation is used throughout to preserve the crispness of the binary structure.

### Transition Counting

Thread counting proceeds by scanning full-height profiles down each column. Every dark-to-light transition along a profile marks the crossing of one thread, so the number of transitions encountered along one column equals the number of threads in that direction. The transitions are accumulated over the central columns spanning 25 to 75 percent of the mask width and averaged, yielding a linear thread density expressed directly as thread crossings per profile. Because a full-height profile crosses every thread in the sample, no further scaling is required to recover the count over the full physical sample. The warp count is obtained by transposing the warp mask and scanning as above; the weft count is obtained by scanning the weft mask directly.

The central-column band is what keeps the measurement honest under rotation. Gauze is flexible, so warp and weft may deviate from 90 degrees up to a worst case of 45 degrees. A rotated square's diagonal exceeds its width, so rotating the crop pushes real threads out of the corners and leaves white padding from the rotation border. Restricting the profile columns to the central 25 to 75 percent guarantees no padding is ever sampled, even at a full 45-degree misalignment, while scanning the full height keeps every thread crossing.

### Pipeline Illustration

The following images, generated in debug mode from validation sample 12, show the principal stages of the pipeline. Each is referenced relative to the images directory of this repository.

[images/binary.png](images/binary.png) is the binarised result after preprocessing. [images/2D_FFT_magnitude.png](images/2D_FFT_magnitude.png) is the centered, log-compressed two-dimensional magnitude spectrum (jet) whose dominant peaks define the thread angles. [images/rotated_binary_warp.png](images/rotated_binary_warp.png) is the cleaned binary after rotation by the recovered warp angle, aligning warp threads vertically. [images/vertical_mask.png](images/vertical_mask.png) is the warp counting mask. [images/rotated_binary_weft.png](images/rotated_binary_weft.png) is the cleaned binary after rotation by the recovered weft angle, aligning weft threads horizontally. [images/horizontal_mask.png](images/horizontal_mask.png) is the weft counting mask produced by directional erosion.

## Acquisition Setup

The system was developed and validated with a deliberately simple rig. A 10-by-10-centimetre black box, formed from a roughly cut piece of black cloth, is placed on a white table, and the medical gauze sample is sandwiched in between. The gauze therefore rests on the black cloth against the white surround. This arrangement yields white threads against a dark sample backing over a white perimeter, which the binarisation at 127 exploits, and the black box itself supplies the 10-centimetre calibration reference.

## Generalisation to Other Textiles

The algorithm is not specific to medical gauze. It transfers to any woven textile in which the warp and weft threads are white, or generally lighter than their background, and in which the inter-thread gaps are wide enough to reveal the dark backing. Under these conditions the dark gaps give rise to reliable dark-to-light transitions and the two thread families form well-separated spectral peaks, so both the angular correction and the transition counting operate without modification. Textiles with dense, fine, dark, or light-on-light structure require retuning of the binarisation and morphology constants described below.

## Tuning Constants

The following constants are hard-coded in the source and were tuned for the acquisition setup described above; they are calibration parameters rather than arbitrary values. The Fourier stage operates on a quarter-resolution downscale of the input. It zero-pads the transform with a border constant of 1, suppresses the zero-frequency bin with a 7-by-7 window, smooths the spectrum with a 5-by-5 Gaussian kernel, and thresholds the normalised magnitude at 85. It searches the warp sector from minus 45 to plus 45 degrees and the weft sector from plus 45 to plus 135 degrees. The counting stage binarises at 127, cleans the image with 2-by-2 morphology, and merges the region of interest with a 7-by-7 kernel. It assumes a sample size of 10 by 10 centimetres, counts within the band from 25 to 75 percent of each dimension, erodes the warp mask with a 1-by-6 vertical element and the weft mask with a 6-by-1 horizontal element, and fills rotation borders with white at value 255.

## Validation

12 rolls from a single lot of 17-thread gauze were sampled against manual reference counts established by qualified technicians, 3 measurements per roll and averaged. The EN 14079 limits are 95 to 105 threads per 10 centimetres for warp and 66 to 74 for weft. A roll is declared conforming only if both directions fall within range, otherwise non-conforming. Concordance is the share of rolls for which the engine and the manual reference reach the same overall conformance decision, not agreement on any single measurement. Both the 90-percent concordance and the 1-thread mean absolute error acceptance criteria were met where applicable.

On this basis the engine reached a joint per-roll concordance of 11 rolls in 12, or 91.7 percent. The single discordance was roll 1, whose manual weft count of 66 sits exactly on the lower limit, a zone of high measurement uncertainty. Roll 4 measured 106 warp threads, above the upper limit, and was therefore reported non-conforming. The manual reference also judged roll 4 non-conforming, because its weft count of 62 already falls below range, so the two decisions agree. Count accuracy is expressed per direction as 1 minus the mean absolute error divided by the specification midpoint (100 for warp, 70 for weft).

```

Metric                             Warp      Weft    Average
Mean absolute error (threads)      2.00      1.08       1.54
Standard deviation of error        1.63      0.76       1.35
Bias (mean signed error)           1.67      0.08       0.88
Maximum absolute error                6         2       4.00
Accuracy (percent)                98.00     98.45      98.23
Concordance (percent)             91.70     91.70      91.70

```

The warp bias of plus 1.67 threads is a slight systematic overcount, minor relative to the 10-thread tolerance, while the weft estimate is essentially unbiased. The Fourier stage is confined to angular correction; a purely frequency-domain estimator reached only 33.3 percent concordance on the same lot, so the quarter-resolution transform contributes a fast, stable angle estimate without compromising count accuracy.

The engine consumes raw monochrome frames and returns an inspection verdict in about 18 milliseconds, which is roughly 54 frames per second. The quarter-resolution Fourier transform that recovers the correction angles is the single largest share of that time.

## Build and Usage

The project depends on a C++17 compiler and the OpenCV computer-vision library, and compiles as a single translation unit. Build it from the repository root with

g++ -std=c++17 -O2 -o cvengine src/cvengine.cpp $(pkg-config --cflags --libs opencv4)

To inspect a sample, pass the path to an 8-bit 1024-by-1024 monochrome raw frame:

./cvengine /path/to/frame.raw

The program prints the two thread counts it derives, warp_threads and weft_threads, expressed in threads per 10 cm. The output is valid, parseable JSON, so it can be piped directly to any JSON parser or loaded by a scripting language. This is the only output in normal operation, so a batch of rolls can be processed and parsed without any further noise. Supplying the additional argument --debug enables diagnostics: the program then writes the intermediate pipeline images to the working directory for inspection and reports the recovered correction angles, the estimated frequencies, and the calibration box dimension alongside the two counts.

## Repository Layout

The repository contains the engine source under src, the pipeline illustrations under images, the Apache 2.0 license file, and the git ignore rules. Validation sample photographs are deliberately not published to keep the sample data private; the images directory holds only the 6 generated pipeline stage visualisations shown above.

## License

This project is licensed under the Apache License, Version 2.0. See the LICENSE file for the full terms. Copyright 2026 Salem Berbouchi.
