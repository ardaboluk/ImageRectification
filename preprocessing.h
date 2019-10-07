#pragma once

#include "opencv2/opencv.hpp"

class Preprocessing {
	
public:
	/*
	* pointCorresponces shoud be in format (X,Y,X',Y')
	*/
	static cv::Mat getNormalizationMat(std::vector<cv::Point3f> points);

	static std::vector<cv::Point3f> transformPointsToHomogen(std::vector<cv::Point2f> points);

	static std::vector<cv::Point3f> normalizeCoordinates(std::vector<cv::Point3f> points, cv::Mat normalizationMat);

	static cv::Mat denormalizeFundamentalMatrix(cv::Mat fundamentalMatrix, cv::Mat normalizationMat1, cv::Mat normalizationMat2);
};
