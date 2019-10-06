#pragma once

#include "opencv2/opencv.hpp"

class Preprocessing {

private:
	static cv::Mat getNormalizationMat(std::vector<cv::Point3f> points);
public:
	/*
	* pointCorresponces shoud be in format (X,Y,X',Y')
	*/
	static std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> normalizeCoordinates(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>, int image1Width, int image1Height, int image2Width, int image2Height);

	static double** denormalizeFundamentalMatrix(double** fmat, float image1Width, float image1Height, float image2Width, float image2Height);
};
