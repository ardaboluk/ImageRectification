
#include <iostream>
#include <vector>
#include <limits>
#include <exception>

#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "estimator.h"
#include "util.h"
#include "rectification.h"
#include "correspondenceChooser.h"

//DEBUG
using namespace cv;

// These methods check a given fundamental matrix if they pass the test xFx' = 0 for each given point
void checkFundamentalMatrix(double**, float**);
void checkFundamentalMatrix(cv::Mat&, cv::Mat&, cv::Mat&);
void getFileNamesFromUser();

void debugWithOpenCV(std::string image1FileName, std::string image2FileName);
void debugSaveCorrespondencesToMat(cv::Mat correspondences, std::string filename);

std::string image1FileName;
std::string image2FileName;

int __main(int argc, char* argv[]) {

	std::string custom_or_debug(argv[1]);
	if(custom_or_debug.compare("custom") == 0){
		Rectification rectificator("img1.jpg", "img2.jpg");
		std::pair<cv::Mat, cv::Mat> rectifiedImages =  rectificator.rectifyImages();
		cv::namedWindow("RectifiedImage1");
		cv::namedWindow("RectifiedImage2");
		cv::imshow("RectifiedImage1", rectifiedImages.first);
		cv::imshow("RectifiedImage2", rectifiedImages.second);
		cv::waitKey(0);
	}else if(custom_or_debug.compare("debug") == 0){
		debugWithOpenCV("img1.jpg", "img2.jpg");
	}	
	
	/*if (argc > 2) {
		image1FileName = argv[1];
		image2FileName = argv[2];
	}
	else {
		getFileNamesFromUser();
	}

	cv::Mat image1 = cv::imread(image1FileName);
	cv::Mat image2 = cv::imread(image2FileName);

	// Choose correspondences manually
	CorrespondenceChooser::initializeImages(image1, image2);
	cv::namedWindow("Correspondence");
	cv::setMouseCallback("Correspondence", CorrespondenceChooser::CallBackFunc, 0);

	while (true) {
		imshow("Correspondence", CorrespondenceChooser::getResultImageToDisplay());
		if (CorrespondenceChooser::isWindowClosed) {
			break;
		}
		cv::waitKey(1);
	}

	float** pointCorrespondences = CorrespondenceChooser::getPointCorrespondences();

	// Estimate fundamental matrix
	Estimator estimator;

	double** FMatrix = estimator.estimateFundamentalMatrix(pointCorrespondences, CorrespondenceChooser::numCorrespondences);

	// Calculate and draw epipolar lines
	float** epilines = Rectification::getEpilines(pointCorrespondences, CorrespondenceChooser::numCorrespondences, FMatrix);
	cv::Mat image1EpilinesCopy = image1.clone();
	cv::Mat image2EpilinesCopy = image2.clone();
	Rectification::drawEpilines(epilines, CorrespondenceChooser::numCorrespondences, image1EpilinesCopy, image2EpilinesCopy);
	cv::namedWindow("Epilines1");
	cv::namedWindow("Epilines2");
	cv::imshow("Epilines1", image1EpilinesCopy);
	cv::imshow("Epilines2", image2EpilinesCopy);
	cv::waitKey(0);

	// Calculate the homography matrices
	double** H1 = new double* [3];
	double** H2 = new double* [3];
	for (int i = 0; i < 3; i++) {
		H1[i] = new double[3];
		H2[i] = new double[3];
	}
	Rectification::rectifyUncalibratedCV(pointCorrespondences, CorrespondenceChooser::numCorrespondences, FMatrix, image1.rows, image1.cols, H1, H2);
	cv::Mat H1Mat(cv::Size(3, 3), CV_64FC1);
	cv::Mat H2Mat(cv::Size(3, 3), CV_64FC1);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			H1Mat.at<double>(i, j) = H1[i][j];
			H2Mat.at<double>(i, j) = H2[i][j];
		}
	}

	//DEBUG
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			std::cout << H1Mat.at<double>(i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			std::cout << H2Mat.at<double>(i, j) << " ";
		}
		std::cout << std::endl;
	}

	// Transform images using the homography matrices
	cv::Mat image1Transformed(image1.size(), image1.type());
	cv::Mat image2Transformed(image2.size(), image2.type());
	cv::warpPerspective(image1, image1Transformed, H1Mat, image1Transformed.size());
	cv::warpPerspective(image2, image2Transformed, H2Mat, image2Transformed.size());

	// show the transformed images
	cv::namedWindow("Image1 transformed");
	cv::namedWindow("Image2 transformed");
	cv::imshow("Image1 transformed", image1Transformed);
	cv::imshow("Image2 transformed", image2Transformed);
	cv::waitKey(0);


	// Release resources
	for (int i = 0; i < CorrespondenceChooser::numCorrespondences; i++) {
		delete[] pointCorrespondences[i];
		delete[] epilines[i];
	}
	delete[] pointCorrespondences;
	delete[] epilines;

	for (int i = 0; i < 3; i++) {
		delete[] FMatrix[i];
		delete[] H1[i];
		delete[] H2[i];
	}
	delete[] FMatrix;
	delete[] H1;
	delete[] H2;*/

	return 0;
}

void getFileNamesFromUser() {

	std::string outputFileName = "";

	std::cout << "Please enter the file name of the first image." << std::endl;
	std::string tmp = "";
	while (tmp == "") {
		std::cin >> tmp;
	}
	image1FileName = tmp;

	std::cout << "Please enter the file name of the second image." << std::endl;
	tmp = "";
	while (tmp == "") {
		std::cin >> tmp;
	}
	image2FileName = tmp;
}


void checkFundamentalMatrix(double** fundamentalMatrix, float** correspondences) {
	/*
	* Checks if the given fundamental matrix satisfies the equation xFx' ~ 0 for each point.
	*/
	double fmatArray[3][3];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			fmatArray[i][j] = fundamentalMatrix[i][j];
		}
	}

	cv::Mat fmat = cv::Mat(3, 3, CV_64FC1, fmatArray);

	std::cout << "Verifying the fundamental matrix using line equation." << std::endl;
	for (int i = 0; i < 8; i++) {
		double tmpXArr[3] = { correspondences[i][0], correspondences[i][1], 1.0 };
		double tmpXPrimeArr[3] = { correspondences[i][2], correspondences[i][3], 1.0 };
		cv::Mat xVec = cv::Mat(1, 3, CV_64FC1, tmpXArr);
		cv::Mat xPrimeVec = cv::Mat(3, 1, CV_64FC1, tmpXPrimeArr);
		cv::Mat resultMat = xVec * fmat * xPrimeVec;
		std::cout << resultMat << std::endl;
	}
	std::cout << std::endl;
}


void debugWithOpenCV(std::string image1FileName, std::string image2FileName){

	cv::Mat image1 = cv::imread(image1FileName);
	cv::Mat image2 = cv::imread(image2FileName);
	std::pair<std::vector<Point2f>, std::vector<Point2f>> correspondingPointsList = Util::extractMatches(image1, image2, 8);
	std::vector<Point2f> correspondingPoints1 = correspondingPointsList.first;
	std::vector<Point2f> correspondingPoints2 = correspondingPointsList.second;
	
	cv::Mat FMat;
	FMat = cv::findFundamentalMat(correspondingPoints1, correspondingPoints2, CV_FM_8POINT);
	std::cout << FMat << std::endl;

	cv::Mat H1, H2;
	
	cv::stereoRectifyUncalibrated(correspondingPoints1, correspondingPoints2, FMat, 
		cv::Size(image1.cols, image1.rows), H1, H2, 3);

	std::vector<cv::Vec3f> linesOnImage2, linesOnImage1;
	cv::computeCorrespondEpilines(correspondingPoints1, 1, FMat, linesOnImage2);
	cv::computeCorrespondEpilines(correspondingPoints2, 2, FMat, linesOnImage1);

	cv::Mat image1WithEpilines = image1.clone();
	cv::Mat image2WithEpilines = image2.clone();

	for(auto it = linesOnImage1.begin(); it != linesOnImage1.end(); ++it){
		double a1 = (*it)[0];
		double b1 = (*it)[1];
		double c1 = (*it)[2];
		double a1Transformed = a1; // H1.at<double>(0,0) * a1 + H1.at<double>(0,1) * b1 + H1.at<double>(0,2) * c1;
		double b1Transformed = b1; // H1.at<double>(1,0) * a1 + H1.at<double>(1,1) * b1 + H1.at<double>(1,2) * c1;
		double c1Transformed = c1; // H1.at<double>(2,0) * a1 + H1.at<double>(2,1) * b1 + H1.at<double>(2,2) * c1;
		double p1StartX = 0;
		double p1StartY = -c1Transformed / b1Transformed;
		double p1EndX = image1.cols;
		double p1EndY = -(a1Transformed * image1.cols + c1Transformed) / b1Transformed;
		cv::Point2d p1Start(p1StartX, p1StartY);
		cv::Point2d p1End(p1EndX, p1EndY);

		cv::line(image1WithEpilines, p1Start, p1End, cv::Scalar(0,0,255));
	}

	for(auto it = linesOnImage2.begin(); it != linesOnImage2.end(); ++it){
		double a2 = (*it)[0];
		double b2 = (*it)[1];
		double c2 = (*it)[2];
		double a2Transformed = a2; // H2.at<double>(0,0) * a2 + H2.at<double>(0,1) * b2 + H2.at<double>(0,2) * c2;
		double b2Transformed = b2; // H2.at<double>(1,0) * a2 + H2.at<double>(1,1) * b2 + H2.at<double>(1,2) * c2;
		double c2Transformed = c2; // H2.at<double>(2,0) * a2 + H2.at<double>(2,1) * b2 + H2.at<double>(2,2) * c2;
		double p2StartX = 0;
		double p2StartY = -c2Transformed / b2Transformed;
		double p2EndX = image2.cols;
		double p2EndY = -(a2Transformed * image2.cols + c2Transformed) / b2Transformed;
		cv::Point2d p2Start(p2StartX, p2StartY);
		cv::Point2d p2End(p2EndX, p2EndY);

		cv::line(image2WithEpilines, p2Start, p2End, cv::Scalar(0,0,255));
	}

	cv::namedWindow("Epilines1");
	cv::imshow("Epilines1", image1WithEpilines);
	cv::namedWindow("Epilines2");
	cv::imshow("Epilines2", image2WithEpilines);
	cv::waitKey(0);

	cv::Mat image1Warped, image2Warped;

	cv::warpPerspective(image1WithEpilines, image1Warped, H1, cv::Size(image1.cols, image1.rows));
	cv::warpPerspective(image2WithEpilines, image2Warped, H2, cv::Size(image2.cols, image2.rows));

	cv::Mat imageConcatenated;
	cv::hconcat(image1Warped, image2Warped, imageConcatenated);	

	cv::namedWindow("Rectified");
	cv::imshow("Rectified", imageConcatenated);
	cv::waitKey(0);
}