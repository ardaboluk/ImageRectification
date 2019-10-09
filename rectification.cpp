
#include <iostream>

#include "rectification.h"
#include "util.h"
#include "preprocessing.h"
#include "estimator.h"

Rectification::Rectification(std::string image1FileName, std::string image2FileName){
	image1 = cv::imread(image1FileName);
	image2 = cv::imread(image2FileName);
}

cv::Mat Rectification::warpImage(cv::Mat image, cv::Mat homography){
	cv::Mat warpedImage;
	cv::warpPerspective(image, warpedImage, homography, cv::Size(image.cols, image.rows));
	return warpedImage;
}

std::pair<cv::Mat, cv::Mat> Rectification::rectifyImages(){
	std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList = Util::extractMatches(image1, image2, 8);
	std::vector<cv::Point2f> correspondingPoints1 = correspondingPointsList.first;
	std::vector<cv::Point2f> correspondingPoints2 = correspondingPointsList.second;
	std::vector<cv::Point3f> correspondingPoints1_3f =  Preprocessing::transformPointsToHomogen(correspondingPoints1);
	std::vector<cv::Point3f> correspondingPoints2_3f =  Preprocessing::transformPointsToHomogen(correspondingPoints2);

	cv::Mat normMat1 = Preprocessing::getNormalizationMat(correspondingPoints1_3f);
	cv::Mat normMat2 = Preprocessing::getNormalizationMat(correspondingPoints2_3f);

	std::vector<cv::Point3f> correspondingPoints1_3f_normalized = Preprocessing::normalizeCoordinates(correspondingPoints1_3f, normMat1);
	std::vector<cv::Point3f> correspondingPoints2_3f_normalized = Preprocessing::normalizeCoordinates(correspondingPoints2_3f, normMat2);

	std::vector<cv::Point2f> correspondingPoints1_normalized = Preprocessing::transformPointsToNonHomogen(correspondingPoints1_3f_normalized);
	std::vector<cv::Point2f> correspondingPoints2_normalized = Preprocessing::transformPointsToNonHomogen(correspondingPoints2_3f_normalized);

	std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList_normalized;
	correspondingPointsList_normalized.first = correspondingPoints1_normalized;
	correspondingPointsList_normalized.second = correspondingPoints2_normalized;

	Estimator estimator(correspondingPointsList_normalized);
	cv::Mat fundamentalMatrix = estimator.estimateFundamentalMatrix();
	cv::Mat fundamentalMatrixDenormalized = estimator.denormalizeFundamentalMatrix(fundamentalMatrix, normMat1, normMat2);

	std::cout << fundamentalMatrixDenormalized << std::endl;

	std::pair<std::vector<cv::Point3d>, std::vector<cv::Point3d>> epilines = getEpilines(correspondingPointsList, fundamentalMatrixDenormalized);
	cv::Mat image1WithEpilines = image1.clone();
	cv::Mat image2WithEpilines = image2.clone();
	drawEpilines(epilines, image1WithEpilines, image2WithEpilines);
	std::string epilines1WindowName = "Epilines1";
	std::string epilines2WindowName = "Epilines2";
	cv::namedWindow(epilines1WindowName);
	cv::namedWindow(epilines2WindowName);
	cv::imshow(epilines1WindowName, image1WithEpilines);
	cv::imshow(epilines2WindowName, image2WithEpilines);
	cv::waitKey(0);

	std::pair<cv::Mat, cv::Mat> homographyMatrices = estimator.estimateHomographyMatrices_openCV(correspondingPointsList, cv::Size(image1.cols, image1.rows), fundamentalMatrixDenormalized);
	cv::Mat homographyMat1 = homographyMatrices.first;
	cv::Mat homographyMat2 = homographyMatrices.second;
	
	cv::Mat warpedImage1;
	cv::Mat warpedImage2;

	warpedImage1 = warpImage(image1, homographyMat1);
	warpedImage2 = warpImage(image2, homographyMat2);

	std::pair<cv::Mat, cv::Mat> warpedImages;
	warpedImages.first = warpedImage1;
	warpedImages.second = warpedImage2;

	return warpedImages;
}

/*
* Returns an array of epipolar lines.
* Each row represents 2 epipolar lines.
* The first one is the line on the second image corresponding the to point in the first image.
* The second one is the line on the first image corresponding the to point in the second image.
*/
std::pair<std::vector<cv::Point3d>, std::vector<cv::Point3d>> Rectification::getEpilines(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondenceList, cv::Mat fundamentalMatrix) {
	
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

		cv::Mat line1Mat = fundamentalMatrix * point1Mat;
		cv::Mat line2Mat = fundamentalMatrix.t() * point2Mat;
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

std::vector<cv::Mat> Rectification::getEpilinesDebug(float** pointCorrespondences, int numPoints, cv::Mat fundamentalMatrix) {
	cv::Mat points1 = cv::Mat(cv::Size(2, numPoints), CV_64FC1);
	cv::Mat points2 = cv::Mat(cv::Size(2, numPoints), CV_64FC1);
	for (int i = 0; i < numPoints; i++) {
		points1.at<double>(i, 0) = pointCorrespondences[i][0];
		points1.at<double>(i, 1) = pointCorrespondences[i][1];
		points2.at<double>(i, 0) = pointCorrespondences[i][2];
		points2.at<double>(i, 1) = pointCorrespondences[i][3];
	}
	cv::Mat lines1(cv::Size(3, numPoints), CV_64FC1);
	cv::Mat lines2(cv::Size(3, numPoints), CV_64FC1);

	cv::computeCorrespondEpilines(points1, 1, fundamentalMatrix, lines2);
	cv::computeCorrespondEpilines(points2, 2, fundamentalMatrix, lines1);

	std::vector<cv::Mat> lines;
	lines.push_back(lines1);
	lines.push_back(lines2);

	return lines;
}

void Rectification::drawEpilines(std::pair<std::vector<cv::Point3d>, std::vector<cv::Point3d>> epilines, cv::Mat image1, cv::Mat image2) {

	for (int i = 0; i < (int)epilines.first.size(); i++) {

		double a1 = epilines.first[i].x; double b1 = epilines.first[i].y; double c1 = epilines.first[i].z;
		double a2 = epilines.second[i].x; double b2 = epilines.second[i].y; double c2 = epilines.second[i].z;

		double p1StartX = 0;
		double p1StartY = -c1 / b1;
		double p1EndX = image1.cols;
		double p1EndY = -(a1 * image1.cols + c1) / b1;
		cv::Point2d p1Start(p1StartX, p1StartY);
		cv::Point2d p1End(p1EndX, p1EndY);

		double p2StartX = 0;
		double p2StartY = -c2 / b2;
		double p2EndX = image2.cols;
		double p2EndY = -(a2 * image2.cols + c2) / b2;
		cv::Point2d p2Start(p2StartX, p2StartY);
		cv::Point2d p2End(p2EndX, p2EndY);

		cv::line(image1, p1Start, p1End, cv::Scalar(0, 0, 255));
		cv::line(image2, p2Start, p2End, cv::Scalar(0, 0, 255));
	}
}

void Rectification::drawEpilinesDebug(std::vector<cv::Mat> epilines, int numLines, cv::Mat image1, cv::Mat image2) {
	cv::Mat lines1 = epilines.at(0);
	cv::Mat lines2 = epilines.at(1);

	for (int i = 0; i < numLines; i++) {
		double a1 = lines1.at<double>(i,0), b1 = lines1.at<double>(i, 1), c1 = lines1.at<double>(i, 2);
		double a2 = lines2.at<double>(i, 0), b2 = lines2.at<double>(i, 1), c2 = lines2.at<double>(i, 2);

		double p1StartX = 0;
		double p1StartY = -c1 / b1;
		double p1EndX = image1.cols;
		double p1EndY = -(a1 * image1.cols + c1) / b1;
		cv::Point2d p1Start(p1StartX, p1StartY);
		cv::Point2d p1End(p1EndX, p1EndY);

		double p2StartX = 0;
		double p2StartY = -c2 / b2;
		double p2EndX = image2.cols;
		double p2EndY = -(a2 * image2.cols + c2) / b2;
		cv::Point2d p2Start(p2StartX, p2StartY);
		cv::Point2d p2End(p2EndX, p2EndY);

		cv::line(image1, p1Start, p1End, cv::Scalar(0, 0, 0));
		cv::line(image2, p2Start, p2End, cv::Scalar(0, 0, 0));
	}
}

void Rectification::rectifyUncalibratedCV(float** pointCorrespondences, int numCorrespondences, double** fundamentalMatrix, 
	int imageRows, int imageCols, double** H1, double** H2) {

	cv::Mat points1(cv::Size(2, numCorrespondences), CV_32FC1);
	cv::Mat points2(cv::Size(2, numCorrespondences), CV_32FC1);
	for (int i = 0; i < numCorrespondences; i++) {
		points1.at<float>(i, 0) = pointCorrespondences[i][0];
		points1.at<float>(i, 1) = pointCorrespondences[i][1];
		points2.at<float>(i, 0) = pointCorrespondences[i][2];
		points2.at<float>(i, 1) = pointCorrespondences[i][3];
	}

	cv::Mat FMat(cv::Size(3, 3), CV_64FC1);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			FMat.at<double>(i, j) = fundamentalMatrix[i][j];
		}
	}

	cv::Mat H1Mat(cv::Size(3, 3), CV_64FC1);
	cv::Mat H2Mat(cv::Size(3, 3), CV_64FC1);

	//DEBUG
	std::cout << points1 << std::endl;
	std::cout << std::endl;
	std::cout << points2 << std::endl;
	std::cout << std::endl;
	std::cout << FMat << std::endl;
	std::cout << std::endl;

	cv::stereoRectifyUncalibrated(points1, points2, FMat, cv::Size(imageCols, imageRows), H1Mat, H2Mat);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			H1[i][j] = H1Mat.at<double>(i, j);
			H2[i][j] = H2Mat.at<double>(i, j);
		}
	}

	//DEBUG
	std::cout << H1Mat << std::endl;
	std::cout << std::endl;
	std::cout << H2Mat << std::endl;
}

void Rectification::rectifyUncalibratedDebug(float** pointCorrespondences, int numCorrespondences, cv::Mat fundamentalMatrix,
	int imageRows, int imageCols, cv::Mat H1, cv::Mat H2) {
	cv::Mat points1(cv::Size(2, numCorrespondences), CV_32FC1);
	cv::Mat points2(cv::Size(2, numCorrespondences), CV_32FC1);
	for (int i = 0; i < numCorrespondences; i++) {
		points1.at<float>(i, 0) = pointCorrespondences[i][0];
		points1.at<float>(i, 1) = pointCorrespondences[i][1];
		points2.at<float>(i, 0) = pointCorrespondences[i][2];
		points2.at<float>(i, 1) = pointCorrespondences[i][3];
	}

	cv::stereoRectifyUncalibrated(points1, points2, fundamentalMatrix, cv::Size(imageCols, imageRows), H1, H2);

	//DEBUG
	std::cout << H1 << std::endl;
	std::cout << std::endl;
	std::cout << H2 << std::endl;
}