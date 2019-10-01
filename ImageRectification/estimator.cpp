
#include <iostream>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "estimator.h"
#include "preprocessing.h"
#include "util.h"

void Estimator::buildHMSMatrix(float** pointCorrespondences) {
	for (int i = 0; i < 8; i++) {
		hms[i][0] = (double)pointCorrespondences[i][2] * pointCorrespondences[i][0]; // x' * x
		hms[i][1] = (double)pointCorrespondences[i][2] * pointCorrespondences[i][1]; // x' * y
		hms[i][2] = pointCorrespondences[i][2]; // x'
		hms[i][3] = (double)pointCorrespondences[i][3] * pointCorrespondences[i][0]; // y' * x
		hms[i][4] = (double)pointCorrespondences[i][3] * pointCorrespondences[i][1]; // y' * y
		hms[i][5] = pointCorrespondences[i][3]; // y'
		hms[i][6] = pointCorrespondences[i][0]; // x
		hms[i][7] = pointCorrespondences[i][1]; // y
		hms[i][8] = 1.0; // 1
	}
}

double** Estimator::estimateFundamentalMatrix(float** pointCorrespondences, int numPoints) {

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
}

cv::Mat Estimator::estimateMatrixDebug(float** pointCorrespondences, int numPoints) {

	cv::Mat points1 = cv::Mat(cv::Size(2, numPoints), CV_64FC1);
	cv::Mat points2 = cv::Mat(cv::Size(2, numPoints), CV_64FC1);
	for (int i = 0; i < numPoints; i++) {
		points1.at<double>(i, 0) = pointCorrespondences[i][0];
		points1.at<double>(i, 1) = pointCorrespondences[i][1];
		points2.at<double>(i, 0) = pointCorrespondences[i][2];
		points2.at<double>(i, 1) = pointCorrespondences[i][3];
	}

	cv::Mat FMat = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);

	return FMat;
}