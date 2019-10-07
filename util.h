#pragma once

#include <vector>
#include <utility>
#include <string>
#include "opencv2/opencv.hpp"

class Util {
public:
	static void displayMat(cv::Mat& cvMatrix, std::string explanation);
	static std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> extractMatches(std::string image1Filename, std::string image2Filename);
};
