#pragma once

#include "opencv2/opencv.hpp"

class Estimator {
private:
	// 8x9 homogeneous linear system
	cv::Mat buildHMSMatrix(std::vector<cv::Point2f> correspondingPoints1, std::vector<cv::Point2f> correspondingPoints2);
public:
	cv::Mat denormalizeFundamentalMatrix(cv::Mat fundamentalMatrix, cv::Mat normalizationMat1, cv::Mat normalizationMat2);
	cv::Mat estimateFundamentalMatrix(std::vector<cv::Point2f> correspondingPoints1, std::vector<cv::Point2f> correspondingPoints2);
	std::pair<std::vector<cv::Point3d>, std::vector<cv::Point3d>> getEpilines(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondenceList, cv::Mat fundamentalMatrix);
	cv::Mat estimateFundamentalMatrix_opencv(std::vector<cv::Point2f> correspondingPoints1, std::vector<cv::Point2f> correspondingPoints2);
	cv::Mat estimateHomography2(cv::Point2d epipole2, cv::Size image2Size);
	cv::Mat estimateHomography1(cv::Mat fundamentalMat, cv::Mat homography2, cv::Point2d epipole2, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList);
	cv::Point2d estimateEpipole(std::vector<cv::Point3d> epilines);
	std::pair<cv::Mat, cv::Mat> estimateHomographyMatrices_openCV(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList, cv::Size imageSize, cv::Mat fundamentalMatrix);
};