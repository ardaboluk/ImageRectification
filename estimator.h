#pragma once

#include "opencv2/opencv.hpp"

class Estimator {
private:
	// 8x9 homogeneous linear system
	cv::Mat hms;
	void buildHMSMatrix();
	std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsListNormalized;
	// origin is assumed to be at (0,0)
	
public:
	Estimator(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsListNormalized);
	cv::Mat denormalizeFundamentalMatrix(cv::Mat fundamentalMatrix, cv::Mat normalizationMat1, cv::Mat normalizationMat2);
	cv::Mat estimateFundamentalMatrix();
	cv::Mat estimateFundamentalMatrix_opencv();
	cv::Mat estimateHomography2(cv::Point2d epipole2, cv::Size image2Size);
	cv::Mat estimateHomography1(cv::Mat fundamentalMat, cv::Mat homography2, cv::Point2d epipole1, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList);
	cv::Point2d estimateEpipole(std::vector<cv::Point3d> epilines);
	std::pair<cv::Mat, cv::Mat> estimateHomographyMatrices(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList, cv::Size imageSize, cv::Mat fundamentalMatrix);
	std::pair<cv::Mat, cv::Mat> estimateHomographyMatrices_openCV(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList, cv::Size imageSize, cv::Mat fundamentalMatrix);
};