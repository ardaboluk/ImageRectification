
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

	cv::Mat denormalizedFundamentalMatrix = (normalizationMat2.t() * fundamentalMatrix) * normalizationMat1;
	return denormalizedFundamentalMatrix;
}

cv::Mat Estimator::estimateFundamentalMatrix(){
	cv::Mat w, u, vt;
	// u * diag(w) * vt
	cv::SVDecomp(hms, w, u, vt, cv::SVD::FULL_UV);
	cv::Mat fundamentalMatrix = cv::Mat::zeros(cv::Size(3, 3), CV_64FC1);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			fundamentalMatrix.at<double>(i, j) = vt.col(8).at<double>(i * 3 + j);
		}
	}
	// enforce rank-2 constraint
	cv::Mat w1, u1, vt1;
	cv::SVDecomp(fundamentalMatrix, w1, u1, vt1, cv::SVD::FULL_UV);
	int minInd = 0;
	double minValue = *w1.ptr<double>(0);
	for (int i = 1; i < w1.rows; i++) {
		if (*w1.ptr<double>(i) < minValue) {
			minValue = *w1.ptr<double>(i);
			minInd = i;
		}
	}
	*w1.ptr<double>(minInd) = 0;
	cv::Mat fundamentalMatrixRank2 = u1 * cv::Mat::diag(w1) * vt1;

	return fundamentalMatrixRank2;
}

/*double** Estimator::estimateFundamentalMatrix(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList) {

	// find widths and heights in both images for normalization and denormalization
	float maxX1 = abs(pointCorrespondences[0][0]), maxY1 = abs(pointCorrespondences[0][1]), 
		maxX2 = abs(pointCorrespondences[0][2]), maxY2 = abs(pointCorrespondences[0][3]);
	for (int i = 1; i < numPoints; i++) {
		if (abs(pointCorrespondences[i][0]) > maxX1) {
			maxX1 = abs(pointCorrespondences[i][0]);
		}
		if (abs(pointCorrespondences[i][1]) > maxY1) {
			maxY1 = abs(pointCorrespondences[i][1]);
		}
		if (abs(pointCorrespondences[i][2]) > maxX2) {
			maxX2 = (pointCorrespondences[i][2]);
		}
		if (abs(pointCorrespondences[i][3]) > maxY2) {
			maxY2 = abs(pointCorrespondences[i][3]);
		}
	}

	float image1Width = maxX1;
	float image1Height = maxY1;
	float image2Width = maxX2;
	float image2Height = maxY2;

	// normalize image points
	float** normalizedPointCorrespondences = Preprocessing::normalizeCoordinates(pointCorrespondences, image1Width, image1Height, image2Width, image2Height, numPoints);

	// fundamental matrix
	double** FMatrix = new double* [3];
	FMatrix[0] = new double[3];
	FMatrix[1] = new double[3];
	FMatrix[2] = new double[3];
	
	buildHMSMatrix(normalizedPointCorrespondences);
	cv::Mat hmsMat(8, 9, CV_64FC1, hms);
	cv::Mat w, u, vt;
	// u * diag(w) * vt
	cv::SVDecomp(hmsMat, w, u, vt, cv::SVD::FULL_UV);

	cv::Mat fundamentalMatrix = cv::Mat::zeros(cv::Size(3, 3), CV_64FC1);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			fundamentalMatrix.at<double>(i, j) = vt.col(8).at<double>(i * 3 + j);
		}
	}	

	// enforce rank-2 constraint
	cv::Mat w1, u1, vt1;
	cv::SVDecomp(fundamentalMatrix, w1, u1, vt1, cv::SVD::FULL_UV);
	int minInd = 0;
	double minValue = *w1.ptr<double>(0);
	for (int i = 1; i < w1.rows; i++) {
		if (*w1.ptr<double>(i) < minValue) {
			minValue = *w1.ptr<double>(i);
			minInd = i;
		}
	}
	*w1.ptr<double>(minInd) = 0;

	cv::Mat FMatCV = u1 * cv::Mat::diag(w1) * vt1;

	// fill our matrix
	for (int i = 0; i < 3; i++) {
		double* rowPtr = FMatCV.ptr<double>(i);
		for (int j = 0; j < 3; j++) {
			FMatrix[i][j] = rowPtr[j];
		}
	}

	// de-normalize the fundamental matrix
	// denormalize the fundamental matrix
	double** denormalizedFMatrix = Preprocessing::denormalizeFundamentalMatrix(FMatrix, image1Width, image1Height, image2Width, image2Height);

	// release resources
	hmsMat.release();
	w.release();
	u.release();
	vt.release();
	w1.release();
	u1.release();
	vt1.release();
	fundamentalMatrix.release();
	FMatCV.release();

	for (int i = 0; i < numPoints; i++) {
		delete[] normalizedPointCorrespondences[i];
	}
	delete[] normalizedPointCorrespondences;

	for (int i = 0; i < 3; i++) {
		delete[] FMatrix[i];
	}
	delete[] FMatrix;

	return denormalizedFMatrix;
}*/