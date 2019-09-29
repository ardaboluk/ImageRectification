
#include "opencv2/opencv.hpp"
#include "estimator.h"

void Estimator::buildHMSMatrix(float** pointCorrespondences) {
	for (int i = 0; i < 8; i++) {
		hms[i][0] = pointCorrespondences[i][0] * pointCorrespondences[i][2]; // x * x'
		hms[i][1] = pointCorrespondences[i][0] * pointCorrespondences[i][3]; // x * y'
		hms[i][2] = pointCorrespondences[i][0]; // x
		hms[i][3] = pointCorrespondences[i][1] * pointCorrespondences[i][2]; // y * x'
		hms[i][4] = pointCorrespondences[i][1] * pointCorrespondences[i][3]; // y * y'
		hms[i][5] = pointCorrespondences[i][1]; // y
		hms[i][6] = pointCorrespondences[i][2]; // x'
		hms[i][7] = pointCorrespondences[i][3]; // y'
		hms[i][1] = 1; // 1
	}
}

float** Estimator::estimateFundamentalMatrix(float** pointCorrespondences) {

	// fundamental matrix
	float** FMatrix = new float* [3];
	FMatrix[0] = new float[3];
	FMatrix[1] = new float[3];
	FMatrix[2] = new float[3];
	
	buildHMSMatrix(pointCorrespondences);
	cv::Mat hmsMat(8, 9, CV_32FC1, pointCorrespondences);
	cv::Mat w, u, vt;
	// u*w*vt
	cv::SVD::compute(hmsMat, w, u, vt);

	// enforce rank-2 constraint
	int minInd = 0;
	float minValue = *w.ptr<float>(0);
	for (int i = 1; i < 3; i++) {
		if (*w.ptr<float>(i) < minValue) {
			minValue = *w.ptr<float>(i);
			minInd = i;
		}
	}
	*w.ptr<float>(minInd) = 0;
	cv::Mat FMatCV = u * cv::Mat::diag(w) * vt;

	// fill our matrix
	for (int i = 0; i < 3; i++) {
		float* rowPtr = FMatCV.ptr<float>(i);
		for (int j = 0; j < 3; j++) {
			FMatrix[i][j] = rowPtr[j];
		}
	}

	hmsMat.release();
	w.release();
	u.release();
	vt.release();
	FMatCV.release();

	return FMatrix;
}