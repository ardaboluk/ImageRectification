#pragma once

#include "opencv2/opencv.hpp"

class Rectification {
public:
	static float** getEpilines(float** pointCorrespondences, int numPoints, double** fundamentalMatrix);
	static void drawEpilines(float** epilines, int numLines, cv::Mat image1, cv::Mat image2);
};
