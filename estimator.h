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
	cv::Mat denormalizeFundamentalMatrix(cv::Mat fundamentalMatrix, cv::Mat normalizationMat1, cv::Mat normalizationMat2);
	cv::Mat estimateFundamentalMatrix();
};