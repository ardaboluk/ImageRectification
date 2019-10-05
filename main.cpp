
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
void debugEstimator(std::string image1FileName, std::string image2FileName);
void debugSaveCorrespondencesToMat(cv::Mat correspondences, std::string filename);

std::string image1FileName;
std::string image2FileName;

int main(int argc, char* argv[]) {

	debugWithOpenCV("img1.jpg", "img2.jpg");
	
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

void checkFundamentalMatrix(cv::Mat& fundamentalMatrix, cv::Mat& points1, cv::Mat& points2) {
	/*
	* This function works with cv::Mat objects.
	* Checks if the given fundamental matrix satisfies the equation xFx' ~ 0 for each point.
	*/
	std::cout << "Verifying the fundamental matrix using line equation." << std::endl;
	for (int i = 0; i < points1.rows; i++) {
		double points1Arr[3] = { points1.at<float>(i, 0), points1.at<float>(i, 1), 1.0f };
		double points2Arr[3] = { points2.at<float>(i, 0), points2.at<float>(i, 1), 1.0f };
		cv::Mat currentPoint1Vec(cv::Size(3, 1), CV_64FC1, points1Arr);
		cv::Mat currentPoint2Vec(cv::Size(3, 1), CV_64FC1, points2Arr);
		cv::Mat resultMat = currentPoint1Vec * fundamentalMatrix * currentPoint2Vec.t();
		std::cout << resultMat << std::endl;
	}
	std::cout << std::endl;
}

void debugWithOpenCV(std::string image1FileName, std::string image2FileName){

	cv::Mat image1 = cv::imread(image1FileName);
	cv::Mat image2 = cv::imread(image2FileName);

	std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

    detector->detect ( image1,keypoints_1 );
    detector->detect ( image2,keypoints_2 );

    descriptor->compute ( image1, keypoints_1, descriptors_1 );
    descriptor->compute ( image2, keypoints_2, descriptors_2 );

    Mat outimg1;
    drawKeypoints( image1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("ORB特征点",outimg1);

    std::vector<DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, matches );

    double min_dist=10000, max_dist=0;

    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    std::vector< DMatch > good_matches;
    for ( int i = 0; (unsigned int)i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }

    Mat img_match;
    Mat img_goodmatch;
    drawMatches ( image1, keypoints_1, image2, keypoints_2, matches, img_match );
    drawMatches ( image1, keypoints_1, image2, keypoints_2, good_matches, img_goodmatch );
    imshow ( "所有匹配点对", img_match );
    imshow ( "优化后匹配点对", img_goodmatch );
    waitKey(0);

	std::vector<cv::Point2f> correspondingPoints1;
	std::vector<cv::Point2f> correspondingPoints2;
	for(int i = 0; i < good_matches.size(); i++){
		int idx1 = good_matches[i].trainIdx;
		int idx2 = good_matches[i].queryIdx;
		correspondingPoints1.push_back(keypoints_1[idx1].pt);
		correspondingPoints2.push_back(keypoints_2[idx2].pt);
	}
	
	cv::Mat FMat;
	FMat = cv::findFundamentalMat(correspondingPoints1, correspondingPoints2);
	cv::Mat H1, H2;

	std::cout << correspondingPoints1.size() << " " << correspondingPoints2.size() << std::endl;
	std::cout << FMat.rows << " " << FMat.cols << std::endl;
	
	cv::stereoRectifyUncalibrated(correspondingPoints1, correspondingPoints2, FMat, 
		cv::Size(image1.cols, image1.rows), H1, H2);
	cv::Mat image1_rectified;
	cv::Mat image2_rectified;
	cv::warpPerspective(image1, image1_rectified, H1, cv::Size(image1.cols, image1.rows));
	cv::warpPerspective(image2, image2_rectified, H2, cv::Size(image2.cols, image2.rows));

	cv::namedWindow("Rectified1");
	cv::imshow("Rectified1", image1_rectified);
	cv::namedWindow("Rectified2");
	cv::imshow("Rectified2", image2_rectified);
	cv::waitKey(0);
}

void debugEstimator(std::string image1FileName, std::string image2FileName){
	//init point correspondences
	/*float** pointCorrespondences = new float*[8];
	for(int i = 0; i < 8; i++){
		pointCorrespondences[i] = new float[4];
	}

	//find fundamental matrix using OpenCV
	Estimator estimator;
	cv::Mat FMat = estimator.estimateMatrixDebug(pointCorrespondences, CorrespondenceChooser::numCorrespondences);
	std::cout << FMat.type() << std::endl;
	std::vector<cv::Mat> lines = Rectification::getEpilinesDebug(pointCorrespondences, CorrespondenceChooser::numCorrespondences, FMat);
	cv::Mat image1 = cv::imread("img1.jpg");
	cv::Mat image2 = cv::imread("img2.jpg");
	cv::Mat image1EpilinesCopyDebug = image1.clone();
	cv::Mat image2EpilinesCopyDebug = image2.clone();
	Rectification::drawEpilinesDebug(lines, CorrespondenceChooser::numCorrespondences, image1EpilinesCopyDebug, image2EpilinesCopyDebug);
	cv::namedWindow("Epilines1 Debug");
	cv::namedWindow("Epilines2 Debug");
	cv::imshow("Epilines1 Debug", image1EpilinesCopyDebug);
	cv::imshow("Epilines2 Debug", image2EpilinesCopyDebug);
	cv::waitKey(0);
	cv::Mat H1CV(cv::Size(3, 3), CV_64FC1);
	cv::Mat H2CV(cv::Size(3, 3), CV_64FC1);
	Rectification::rectifyUncalibratedDebug(pointCorrespondences, CorrespondenceChooser::numCorrespondences, FMat, image1.rows, image1.cols, H1CV, H2CV);
	cv::Mat image1TransformedDebug(image1.size(), image1.type());
	cv::Mat image2TransformedDebug(image2.size(), image2.type());
	cv::warpPerspective(image1, image1TransformedDebug, H1CV, image1TransformedDebug.size());
	cv::warpPerspective(image2, image2TransformedDebug, H2CV, image2TransformedDebug.size());
	cv::namedWindow("Image1 transformed debug");
	cv::namedWindow("Image2 transformed debug");
	cv::imshow("Image1 transformed debug", image1TransformedDebug);
	cv::imshow("Image2 transformed debug", image2TransformedDebug);
	cv::waitKey(0);
	for (int i = 0; i < CorrespondenceChooser::numCorrespondences; i++) {
		delete[] pointCorrespondences[i];
	}
	delete[] pointCorrespondences;*/
}