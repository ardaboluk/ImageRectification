
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
	// u * diag(w) * v
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

/*
* Epilines are represented by cv::Point3d in the form (a, b, c) where ax + by + c = 0
*/
cv::Point2d Estimator::estimateEpipole(std::vector<cv::Point3d> epilines){
	cv::Point2d epipole;

	// line equations Ax = 0 where A = [e1 e2 ...].t(), ek = [ak bk ck] and x = [x1 x2 1].t()
	cv::Mat A(cv::Size(2, epilines.size()), CV_64FC1);
	cv::Mat b(cv::Size(1, epilines.size()), CV_64FC1);
	for(int i = 0; i < (int)epilines.size(); i++){
		A.at<double>(i,0) = epilines[i].x;
		A.at<double>(i,1) = epilines[i].y;

		b.at<double>(i,0) = -(epilines[i].z);
	}

	cv::Mat W, U, Vt, V;
	cv::SVD::compute(A, W, U, Vt);
	V = Vt.t();

	double threshold = 10e-12 * abs(W.at<double>(0,0));

	cv::Mat Z(cv::Size(1, W.rows), CV_64FC1);
	for(int i = 0; i < W.rows; i++){
		double w_i = W.at<double>(i,0);
		if(w_i > threshold){
			Z.at<double>(i,0) = 1 / w_i;
		}else{
			Z.at<double>(i,0) = 0;
		}
	}
	cv::Mat diagZ = cv::Mat::diag(Z);

	cv::Mat epipoleMat = V * diagZ * U.t() * b;
	epipole.x = epipoleMat.at<double>(0,0);
	epipole.y = epipoleMat.at<double>(1,0);

	return epipole;
}

cv::Mat Estimator::estimateHomography2(cv::Point2d epipole2, cv::Size image2Size){
	// Reference:
	// web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf

	cv::Mat H2;

	cv::Mat T = cv::Mat::zeros(cv::Size(3,3), CV_64FC1);
	cv::Mat R = cv::Mat::zeros(cv::Size(3,3), CV_64FC1);
	cv::Mat epipole2Mat(cv::Size(1,3), CV_64FC1);

	epipole2Mat.at<double>(0,0) = epipole2.x;
	epipole2Mat.at<double>(1,0) = epipole2.y;
	epipole2Mat.at<double>(2,0) = 1.0;

	T.at<double>(0,0) = 1.0;
	T.at<double>(0,2) = -((double)image2Size.width / 2);
	T.at<double>(1,1) = 1.0;
	T.at<double>(1,2) = -((double)image2Size.height / 2);
	T.at<double>(2,2) = 1.0;

	cv::Mat Te2 = T * epipole2Mat;

	double Te2x = Te2.at<double>(0,0);
	double Te2y = Te2.at<double>(1,0);
	double alpha = Te2x >= 0 ? 1.0 : -1.0;
	double hypotenuse = sqrt(pow(Te2x,2) + pow(Te2y,2));
	R.at<double>(0,0) = alpha * (Te2x / hypotenuse);
	R.at<double>(0,1) = alpha * (Te2y / hypotenuse);
	R.at<double>(0,2) = 0;
	R.at<double>(1,0) = (-alpha) * (Te2y / hypotenuse);
	R.at<double>(1,1) = alpha * (Te2x / hypotenuse);
	R.at<double>(1,2) = 0;
	R.at<double>(2,0) = 0;
	R.at<double>(2,1) = 0;
	R.at<double>(2,2) = 1.0;

	double f = cv::Mat(R * T * epipole2Mat).at<double>(0,0);

	cv::Mat G = cv::Mat::diag(cv::Mat(cv::Point3d(1.0, 1.0, 1.0)));
	G.at<double>(2,0) = -1.0/f;

	H2 = T.inv() * G * R * T;

	return H2;
}

cv::Mat Estimator::estimateHomography1(cv::Mat fundamentalMat, cv::Mat homography2, cv::Point2d epipole1, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList){

	cv::Mat H1;

	cv::Mat skewEx = cv::Mat::zeros(cv::Size(3,3), CV_64FC1);
	skewEx.at<double>(0,1) = -1.0;
	skewEx.at<double>(0,2) = epipole1.y;
	skewEx.at<double>(1,0) = 1.0;
	skewEx.at<double>(1,2) = -(epipole1.x);
	skewEx.at<double>(2,0) = -(epipole1.y);
	skewEx.at<double>(2,1) = epipole1.x;

	double data[] = {epipole1.x, epipole1.y, 1.0};
	cv::Mat M = skewEx * fundamentalMat + cv::Mat(cv::Size(1,3), CV_64FC1, data) * cv::Mat::ones(cv::Size(3,1), CV_64FC1);

	std::vector<cv::Point2f> correspondingPoints1 = correspondingPointsList.first;
	std::vector<cv::Point2f> correspondingPoints2 = correspondingPointsList.second;

	std::vector<cv::Point3d> transformedP;
	std::vector<cv::Point3d> transformedPPrime;

	std::vector<cv::Point3f> points3 = Preprocessing::transformPointsToHomogen(correspondingPoints1);
	std::vector<cv::Point3f> pointsPrime3 = Preprocessing::transformPointsToHomogen(correspondingPoints2);

	for(int i = 0; i < points3.size(); i++){
		cv::Mat pointMat3;
		cv::Mat pointPrimeMat3;
		cv::Mat(points3[i]).convertTo(pointMat3, CV_64FC1);
		cv::Mat(pointsPrime3[i]).convertTo(pointPrimeMat3, CV_64FC1);

		cv::Mat transformedPointMat3 = homography2 * M * pointMat3;
		cv::Mat transformedPointPrimeMat3 = homography2 * pointPrimeMat3;
		cv::Point3d transformedPoint3(transformedPointMat3.at<double>(0,0) / transformedPointMat3.at<double>(2,0), transformedPointMat3.at<double>(1,0) / transformedPointMat3.at<double>(2,0), 1.0);
		cv::Point3d transformedPointPrime3(transformedPointPrimeMat3.at<double>(0,0) / transformedPointPrimeMat3.at<double>(2,0), transformedPointPrimeMat3.at<double>(1,0) / transformedPointPrimeMat3.at<double>(2,0), 1.0);

		transformedP.push_back(transformedPoint3);
		transformedPPrime.push_back(transformedPointPrime3);

		
	}

	return H1;
}

 std::pair<cv::Mat, cv::Mat> Estimator::estimateHomographyMatrices(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList, cv::Size imageSize, cv::Mat fundamentalMatrix){

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