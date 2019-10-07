#pragma once

#include "opencv2/opencv.hpp"

class Estimator {
private:
	// 8x9 homogeneous linear system
	cv::Mat hms;
	void buildHMSMatrix();
	std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsListNormalized;
public:
	Estimator(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsListNormalized);
	// it's argument is corresponding points in the form {{x_1, y_1, x'_1, y'_1}, {x_2, y_2, x'_2, y'_2}, ...}
	cv::Mat estimateFundamentalMatrix();
};