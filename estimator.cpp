
#include <iostream>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "estimator.h"
#include "preprocessing.h"
#include "util.h"

Estimator::Estimator(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsListNormalized){
	this->correspondingPointsListNormalized = correspondingPointsListNormalized;
	hms = cv::Mat(cv::Size(9, 8), CV_64FC1);
	buildHMSMatrix();
}

void Estimator::buildHMSMatrix() {
	std::vector<cv::Point2f> correspondingPoints1 = correspondingPointsListNormalized.first;
	std::vector<cv::Point2f> correspondingPoints2 = correspondingPointsListNormalized.second;
	for (int i = 0; i < 8; i++) {
		cv::Point2f point = correspondingPoints1[i];
		cv::Point2f pointPrime = correspondingPoints2[i];
		hms.at<double>(i,0) = (double)pointPrime.x * point.x; // x' * x
		hms.at<double>(i,1) = (double)pointPrime.x * point.y; // x' * y
		hms.at<double>(i,2) = pointPrime.x; // x'
		hms.at<double>(i,3) = (double)pointPrime.y * point.x; // y' * x
		hms.at<double>(i,4) = (double)pointPrime.y * point.y; // y' * y
		hms.at<double>(i,5) = pointPrime.y; // y'
		hms.at<double>(i,6) = point.x; // x
		hms.at<double>(i,7) = point.y; // y
		hms.at<double>(i,8) = 1.0; // 1
	}
}

cv::Mat Estimator::denormalizeFundamentalMatrix(cv::Mat fundamentalMatrix, cv::Mat normalizationMat1, cv::Mat normalizationMat2){
	cv::Mat normalizationMat1_64F;
	cv::Mat normalizationMat2_64F;
	normalizationMat1.convertTo(normalizationMat1_64F, CV_64FC1);
	normalizationMat2.convertTo(normalizationMat2_64F, CV_64FC1);
	
	cv::Mat denormalizedFundamentalMatrix = normalizationMat2_64F.t() * fundamentalMatrix * normalizationMat1_64F;

	double fmat22 = denormalizedFundamentalMatrix.at<double>(2,2);
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			denormalizedFundamentalMatrix.at<double>(i,j) /= fmat22;
		}
	}

	return denormalizedFundamentalMatrix;
}

cv::Mat Estimator::estimateFundamentalMatrix(){
	cv::Mat w, u, vt;
	// u * diag(w) * vt
	cv::SVDecomp(hms, w, u, vt, cv::SVD::FULL_UV);
	cv::Mat v = vt.t();
	cv::Mat fundamentalMatrix = cv::Mat::zeros(cv::Size(3, 3), CV_64FC1);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			fundamentalMatrix.at<double>(i, j) = v.col(8).at<double>(i * 3 + j);
		}
	}
	// enforce rank-2 constraint
	cv::Mat w1, u1, vt1;
	cv::SVDecomp(fundamentalMatrix, w1, u1, vt1, cv::SVD::FULL_UV);
	w1.at<double>(2,0) = 0;
	cv::Mat fundamentalMatrixRank2 = u1 * cv::Mat::diag(w1) * vt1;

	return fundamentalMatrixRank2;
}

/* 
* If this method will be called, the private member correspondingPointsListNormalized should be given
* corresponding points that aren't normalized.
*/
cv::Mat Estimator::estimateFundamentalMatrix_opencv(){
	std::vector<cv::Point2f> correspondingPoints1 = correspondingPointsListNormalized.first;
	std::vector<cv::Point2f> correspondingPoints2 = correspondingPointsListNormalized.second;
	cv::Mat FMat;
	FMat = cv::findFundamentalMat(correspondingPoints1, correspondingPoints2, CV_FM_8POINT);
	return FMat;
}

std::pair<cv::Mat, cv::Mat> Estimator::estimateHomographyMatrices_openCV(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList, cv::Size imageSize, cv::Mat fundamentalMatrix){
	cv::Mat homographyMat1;
	cv::Mat homographyMat2;

	std::vector<cv::Point2f> correspondingPoints1 = correspondingPointsList.first;
	std::vector<cv::Point2f> correspondingPoints2 = correspondingPointsList.second;

	// std::cout << "correspondingPoints1" << std::endl << correspondingPoints1 << std::endl;
	// std::cout << "correspondingPoints2" << std::endl << correspondingPoints2 << std::endl;
	// std::cout << "fundamental matrix" << std::endl << fundamentalMatrix << std::endl;
	// std::cout << "image rows " << imageSize.height << " image cols " << imageSize.width << std::endl;

	cv::stereoRectifyUncalibrated(correspondingPoints1, correspondingPoints2, fundamentalMatrix, imageSize, homographyMat1, homographyMat2, 3);

	std::pair<cv::Mat, cv::Mat> homographyMatrices;
	homographyMatrices.first = homographyMat1;
	homographyMatrices.second = homographyMat2;
	return homographyMatrices;
}