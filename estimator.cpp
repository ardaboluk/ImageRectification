
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>

#include "opencv2/opencv.hpp"
#include "estimator.h"
#include "preprocessing.h"
#include "util.h"

cv::Mat Estimator::buildHMSMatrix8x9(std::vector<cv::Point2f> correspondingPoints1, std::vector<cv::Point2f> correspondingPoints2) {
	cv::Mat hms(cv::Size(9,8), CV_64FC1);
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
	return hms;
}

cv::Mat Estimator::buildHMSMatrixNx9(std::vector<cv::Point2f> correspondingPoints1, std::vector<cv::Point2f> correspondingPoints2){
	cv::Mat hms(cv::Size(9,8), CV_64FC1);
	for (int i = 0; i < correspondingPoints1.size(); i++) {
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
	return hms;
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

cv::Mat Estimator::estimateFundamentalMatrix8Point(std::vector<cv::Point2f> correspondingPoints1, std::vector<cv::Point2f> correspondingPoints2){
	cv::Mat hms = buildHMSMatrix8x9(correspondingPoints1, correspondingPoints2);
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

cv::Mat Estimator::estimateFundamentalMatrixNPoint(std::vector<cv::Point2f> correspondingPoints1, std::vector<cv::Point2f> correspondingPoints2){
	cv::Mat hms = buildHMSMatrixNx9(correspondingPoints1, correspondingPoints2);
	cv::Mat w, u, vt;
	// u * diag(w) * v
	cv::SVDecomp(hms, w, u, vt, cv::SVD::FULL_UV);
	cv::Mat v = vt.t();
	cv::Mat fundamentalMatrix = cv::Mat::zeros(cv::Size(3, 3), CV_64FC1);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			fundamentalMatrix.at<double>(i, j) = v.col(v.cols - 1).at<double>(i * 3 + j);
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
* Returns an array of epipolar lines.
* Each row represents 2 epipolar lines.
* The first one is the line on the second image corresponding the to point in the first image.
* The second one is the line on the first image corresponding the to point in the second image.
*/
std::pair<std::vector<cv::Point3d>, std::vector<cv::Point3d>> Estimator::getEpilines(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondenceList, cv::Mat fundamentalMatrix) {
	
	std::pair<std::vector<cv::Point3d>, std::vector<cv::Point3d>> epilines;

	for (int i = 0; i < (int)correspondenceList.first.size(); i++) {
		cv::Point3d point1;
		cv::Point3d point2;
		cv::Mat point1Mat;
		cv::Mat point2Mat;
		point1.x = correspondenceList.first[i].x;
		point1.y = correspondenceList.first[i].y;
		point1.z = 1.0;
		point1Mat = cv::Mat(point1);
		point2.x = correspondenceList.second[i].x;
		point2.y = correspondenceList.second[i].y;
		point2.z = 1.0;
		point2Mat = cv::Mat(point2);

		cv::Mat line2Mat = fundamentalMatrix * point1Mat;
		cv::Mat line1Mat = fundamentalMatrix.t() * point2Mat;
		cv::Point3d line1;
		cv::Point3d line2;
		line1.x = line1Mat.at<double>(0,0);
		line1.y = line1Mat.at<double>(1,0);
		line1.z = line1Mat.at<double>(2,0);
		line2.x = line2Mat.at<double>(0,0);
		line2.y = line2Mat.at<double>(1,0);
		line2.z = line2Mat.at<double>(2,0);

		epilines.first.push_back(line1);
		epilines.second.push_back(line2);
	}

	return epilines;
}

/* 
* If this method will be called, the private member correspondingPointsListNormalized should be given
* corresponding points that aren't normalized.
*/
cv::Mat Estimator::estimateFundamentalMatrix_opencv(std::vector<cv::Point2f> correspondingPoints1, std::vector<cv::Point2f> correspondingPoints2){
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

cv::Mat Estimator::estimateHomography1(cv::Mat fundamentalMat, cv::Mat homography2, cv::Point2d epipole2, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList){

	cv::Mat H1;

	cv::Mat skewEx = cv::Mat::zeros(cv::Size(3,3), CV_64FC1);
	skewEx.at<double>(0,1) = -1.0;
	skewEx.at<double>(0,2) = epipole2.y;
	skewEx.at<double>(1,0) = 1.0;
	skewEx.at<double>(1,2) = -(epipole2.x);
	skewEx.at<double>(2,0) = -(epipole2.y);
	skewEx.at<double>(2,1) = epipole2.x;

	double data[] = {epipole2.x, epipole2.y, 1.0};
	cv::Mat M = skewEx * fundamentalMat + cv::Mat(cv::Size(1,3), CV_64FC1, data) * cv::Mat::ones(cv::Size(3,1), CV_64FC1);

	std::vector<cv::Point2f> correspondingPoints1 = correspondingPointsList.first;
	std::vector<cv::Point2f> correspondingPoints2 = correspondingPointsList.second;

	cv::Mat transformedP(cv::Size(3,correspondingPoints1.size()), CV_64FC1);
	cv::Mat transformedPPrimeX(cv::Size(1,correspondingPoints1.size()), CV_64FC1);

	std::vector<cv::Point3f> points3 = Preprocessing::transformPointsToHomogen(correspondingPoints1);
	std::vector<cv::Point3f> pointsPrime3 = Preprocessing::transformPointsToHomogen(correspondingPoints2);

	for(int i = 0; i < (int)points3.size(); i++){
		cv::Mat pointMat3;
		cv::Mat pointPrimeMat3;
		cv::Mat(points3[i]).convertTo(pointMat3, CV_64FC1);
		cv::Mat(pointsPrime3[i]).convertTo(pointPrimeMat3, CV_64FC1);

		cv::Mat transformedPointMat3 = homography2 * M * pointMat3;
		cv::Mat transformedPointPrimeMat3 = homography2 * pointPrimeMat3;

		transformedP.at<double>(i,0) = transformedPointMat3.at<double>(0,0) / transformedPointMat3.at<double>(2,0);
		transformedP.at<double>(i,1) = transformedPointMat3.at<double>(1,0) / transformedPointMat3.at<double>(2,0);
		transformedP.at<double>(i,2) = 1.0;

		transformedPPrimeX.at<double>(i,0) = transformedPointPrimeMat3.at<double>(0,0) / transformedPointPrimeMat3.at<double>(2,0);
	}

	cv::Mat a;//(cv::Size(1,3), CV_64FC1);

	cv::solve(transformedP, transformedPPrimeX, a, cv::DECOMP_SVD);

	double a1 = a.at<double>(0,0);
	double a2 = a.at<double>(1,0);
	double a3 = a.at<double>(2,0);

	cv::Mat HA = cv::Mat::eye(cv::Size(3,3), CV_64FC1);
	HA.at<double>(0,0) = a1;
	HA.at<double>(0,1) = a2;
	HA.at<double>(0,2) = a3;

	H1 = HA * homography2 * M;

	return H1;
}


/*
* This method is used in test_fundamentalMatrixOpencv in test_estimator.
*/
std::pair<cv::Mat, cv::Mat> Estimator::estimateHomographyMatrices_openCV(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList, cv::Size imageSize, cv::Mat fundamentalMatrix){
	cv::Mat homographyMat1;
	cv::Mat homographyMat2;

	std::vector<cv::Point2f> correspondingPoints1 = correspondingPointsList.first;
	std::vector<cv::Point2f> correspondingPoints2 = correspondingPointsList.second;

	cv::stereoRectifyUncalibrated(correspondingPoints1, correspondingPoints2, fundamentalMatrix, imageSize, homographyMat1, homographyMat2, 3);

	std::pair<cv::Mat, cv::Mat> homographyMatrices;
	homographyMatrices.first = homographyMat1;
	homographyMatrices.second = homographyMat2;
	return homographyMatrices;
}