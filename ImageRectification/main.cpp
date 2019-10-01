
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

// These methods check a given fundamental matrix if they pass the test xFx' = 0 for each given point
void checkFundamentalMatrix(double**, float**);
void checkFundamentalMatrix(cv::Mat&, cv::Mat&, cv::Mat&);

int main() {

	cv::Mat image1 = cv::imread("img1.jpg");
	cv::Mat image2 = cv::imread("img2.jpg");

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

	//DEBUG
	/*cv::Mat FMat = estimator.estimateMatrixDebug(pointCorrespondences, CorrespondenceChooser::numCorrespondences);
	std::cout << FMat.type() << std::endl;
	std::vector<cv::Mat> lines = Rectification::getEpilinesDebug(pointCorrespondences, CorrespondenceChooser::numCorrespondences, FMat);
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
	cv::waitKey(0);*/

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
	delete[] H2;

	return 0;
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

void testEstimator() {
	cv::Mat image1 = cv::imread("img1.jpg", 0);
	cv::Mat image2 = cv::imread("img2.jpg", 0);
	if (!image1.data || !image2.data)
		return;

	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;

	cv::Ptr<cv::ORB> surf = cv::ORB::create();

	surf->detect(image1, keypoints1);
	surf->detect(image2, keypoints2);

	std::cout << "Number of SURF points (1): " << keypoints1.size() << std::endl;
	std::cout << "Number of SURF points (2): " << keypoints2.size() << std::endl;

	cv::Mat descriptors1, descriptors2;
	surf->compute(image1, keypoints1, descriptors1);
	surf->compute(image2, keypoints2, descriptors2);

	cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create();

	std::vector<cv::DMatch> matches;
	matcher->match(descriptors1, descriptors2, matches);

	std::cout << "Number of matched points: " << matches.size() << std::endl;

	std::vector<cv::DMatch> selMatches;

	selMatches.push_back(matches[14]);
	selMatches.push_back(matches[16]);
	selMatches.push_back(matches[141]);
	selMatches.push_back(matches[146]);
	selMatches.push_back(matches[235]);
	selMatches.push_back(matches[238]);
	selMatches.push_back(matches[274]);
	selMatches.push_back(matches[278]);

	std::vector<int> pointIndexes1;
	std::vector<int> pointIndexes2;
	for (auto it = selMatches.begin(); it != selMatches.end(); ++it)
	{
		// Get the indexes of the selected matched keypoints
		pointIndexes1.push_back(it->queryIdx);
		pointIndexes2.push_back(it->trainIdx);
	}

	std::vector<cv::Point2f> selPoints1, selPoints2;
	cv::KeyPoint::convert(keypoints1, selPoints1, pointIndexes1);
	cv::KeyPoint::convert(keypoints2, selPoints2, pointIndexes2);

	float** pointCorrespondences = new float* [8];
	for (int i = 0; i < 8; i++) {
		pointCorrespondences[i] = new float[4];
	}

	for (int i = 0; i < 8; i++) {
		pointCorrespondences[i][0] = selPoints1[i].x;
		pointCorrespondences[i][1] = selPoints1[i].y;
		pointCorrespondences[i][2] = selPoints2[i].x;
		pointCorrespondences[i][3] = selPoints2[i].y;
	}

	Estimator estimator;
	double** FMatrix = estimator.estimateFundamentalMatrix(pointCorrespondences, selPoints1.size());
	//DEBUG
	Util::displayMat(FMatrix, 3, 3, "FMatrix");
	checkFundamentalMatrix(FMatrix, pointCorrespondences);

	//DEBUG
	cv::Mat FMatrix_debug = estimator.estimateMatrixDebug(pointCorrespondences, selPoints1.size());
	Util::displayMat(FMatrix_debug, "FMatrix_debug");
	cv::Mat selPoints1Mat = cv::Mat(selPoints1);
	cv::Mat selPoints2Mat = cv::Mat(selPoints2);
	checkFundamentalMatrix(FMatrix_debug, selPoints1Mat, selPoints2Mat);

	float** epilines = Rectification::getEpilines(pointCorrespondences, selPoints1.size(), FMatrix);
	cv::Mat image1EpilinesCopy = image1.clone();
	cv::Mat image2EpilinesCopy = image2.clone();
	Rectification::drawEpilines(epilines, selPoints1.size(), image1EpilinesCopy, image2EpilinesCopy);
	cv::namedWindow("Epilines1");
	cv::namedWindow("Epilines2");
	cv::imshow("Epilines1", image1EpilinesCopy);
	cv::imshow("Epilines2", image2EpilinesCopy);
	cv::waitKey(0);

	//DEBUG
	cv::Mat image1EpilinesDebugCopy = image1.clone();
	cv::Mat image2EpilinesDebugCopy = image2.clone();
	std::vector<cv::Vec3f> lines1;
	std::vector<cv::Vec3f> lines2;
	cv::computeCorrespondEpilines(cv::Mat(selPoints1), 1, FMatrix_debug, lines1);
	cv::computeCorrespondEpilines(cv::Mat(selPoints2), 1, FMatrix_debug.t(), lines2);
	for (auto it = lines1.begin(); it != lines1.end(); ++it){
		cv::line(image2EpilinesDebugCopy, cv::Point(0, -(*it)[2] / (*it)[1]),
			cv::Point(image2EpilinesDebugCopy.cols, -((*it)[2] + (*it)[0] * image2EpilinesDebugCopy.cols) / (*it)[1]),
			cv::Scalar(255, 255, 255));
	}
	for (auto it = lines2.begin(); it != lines2.end(); ++it) {
		cv::line(image1EpilinesDebugCopy, cv::Point(0, -(*it)[2] / (*it)[1]),
			cv::Point(image1EpilinesDebugCopy.cols, -((*it)[2] + (*it)[0] * image1EpilinesDebugCopy.cols) / (*it)[1]),
			cv::Scalar(255, 255, 255));
	}
	cv::namedWindow("EpilinesDebug1");
	cv::namedWindow("EpilinesDebug2");
	cv::imshow("EpilinesDebug1", image1EpilinesDebugCopy);
	cv::imshow("EpilinesDebug2", image2EpilinesDebugCopy);
	cv::waitKey(0);

	//DEBUG
	Util::displayMat(epilines, selPoints1.size(), 6, "Epilines");

	image1.release();
	image2.release();
	image1EpilinesCopy.release();
	image2EpilinesCopy.release();

	for (int i = 0; i < selPoints1.size(); i++) {
		delete[] pointCorrespondences[i];
		delete[] epilines[i];
	}
	delete[] pointCorrespondences;
	delete[] epilines;

	for (int i = 0; i < 3; i++) {
		delete[] FMatrix[i];
	}
	delete[] FMatrix;
}

void testFundamental() {

	// from
	// github.com/vinjn/opencv-2-cookbook-src/blob/master/Chapter%2009/estimateF.cpp

	// Read input images
	cv::Mat image1 = cv::imread("img1.jpg", 0);
	cv::Mat image2 = cv::imread("img2.jpg", 0);
	if (!image1.data || !image2.data)
		return;

	// Display the images
	cv::namedWindow("Right Image");
	cv::imshow("Right Image", image1);
	cv::namedWindow("Left Image");
	cv::imshow("Left Image", image2);

	// vector of keypoints
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;

	// Construction of the SURF feature detector
	auto surf = cv::ORB::create();

	// Detection of the SURF features
	surf->detect(image1, keypoints1);
	surf->detect(image2, keypoints2);

	std::cout << "Number of SURF points (1): " << keypoints1.size() << std::endl;
	std::cout << "Number of SURF points (2): " << keypoints2.size() << std::endl;

	// Draw the kepoints
	cv::Mat imageKP;
	cv::drawKeypoints(image1, keypoints1, imageKP, cv::Scalar(255, 255, 255),
		cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("Right SURF Features");
	cv::imshow("Right SURF Features", imageKP);
	cv::drawKeypoints(image2, keypoints2, imageKP, cv::Scalar(255, 255, 255),
		cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("Left SURF Features");
	cv::imshow("Left SURF Features", imageKP);

	// Extraction of the SURF descriptors
	cv::Mat descriptors1, descriptors2;
	surf->compute(image1, keypoints1, descriptors1);
	surf->compute(image2, keypoints2, descriptors2);

	std::cout << "descriptor matrix size: " << descriptors1.rows << " by " << descriptors1.cols
		<< std::endl;

	// Construction of the matcher
	auto matcher = cv::BFMatcher::create();

	// Match the two image descriptors
	std::vector<cv::DMatch> matches;
	matcher->match(descriptors1, descriptors2, matches);

	std::cout << "Number of matched points: " << matches.size() << std::endl;

	// Select few Matches
	std::vector<cv::DMatch> selMatches;

	/* between church01 and church03 */
	selMatches.push_back(matches[14]);
	selMatches.push_back(matches[16]);
	selMatches.push_back(matches[141]);
	selMatches.push_back(matches[146]);
	selMatches.push_back(matches[235]);
	selMatches.push_back(matches[238]);
	selMatches.push_back(matches[274]);
	selMatches.push_back(matches[278]);

	// Draw the selected matches
	cv::Mat imageMatches;
	cv::drawMatches(
		image1, keypoints1, // 1st image and its keypoints
		image2, keypoints2, // 2nd image and its keypoints
		//					selMatches,			// the matches
		matches,                    // the matches
		imageMatches,               // the image produced
		cv::Scalar(255, 255, 255)); // color of the lines
	cv::namedWindow("Matches");
	cv::imshow("Matches", imageMatches);

	// Convert 1 vector of keypoints into
	// 2 vectors of Point2f
	std::vector<int> pointIndexes1;
	std::vector<int> pointIndexes2;
	for (auto it = selMatches.begin(); it != selMatches.end(); ++it)
	{

		// Get the indexes of the selected matched keypoints
		pointIndexes1.push_back(it->queryIdx);
		pointIndexes2.push_back(it->trainIdx);
	}

	// Convert keypoints into Point2f
	std::vector<cv::Point2f> selPoints1, selPoints2;
	cv::KeyPoint::convert(keypoints1, selPoints1, pointIndexes1);
	cv::KeyPoint::convert(keypoints2, selPoints2, pointIndexes2);

	// check by drawing the points
	auto it = selPoints1.begin();
	while (it != selPoints1.end())
	{

		// draw a circle at each corner location
		cv::circle(image1, *it, 3, cv::Scalar(255, 255, 255), 2);
		++it;
	}

	it = selPoints2.begin();
	while (it != selPoints2.end())
	{

		// draw a circle at each corner location
		cv::circle(image2, *it, 3, cv::Scalar(255, 255, 255), 2);
		++it;
	}

	// Compute F matrix from 8 matches
	cv::Mat fundemental = cv::findFundamentalMat(cv::Mat(selPoints1), // points in first image
		cv::Mat(selPoints2), // points in second image
		cv::FM_8POINT);       // 7-point method
	std::cout << "M = " << std::endl << " " << fundemental << std::endl << std::endl;

	//DEBUG
	cv::Mat selPoints1Mat = cv::Mat(selPoints1);
	cv::Mat selPoints2Mat = cv::Mat(selPoints2);
	checkFundamentalMatrix(fundemental, selPoints1Mat, selPoints2Mat);

	std::cout << "F-Matrix size= " << fundemental.rows << "," << fundemental.cols << std::endl;

	// draw the left points corresponding epipolar lines in right image
	std::vector<cv::Vec3f> lines1;
	cv::computeCorrespondEpilines(cv::Mat(selPoints1), // image points
		1,                   // in image 1 (can also be 2)
		fundemental,         // F matrix
		lines1);             // vector of epipolar lines

// for all epipolar lines
	for (auto it = lines1.begin(); it != lines1.end(); ++it)
	{

		// draw the epipolar line between first and last column
		cv::line(image2, cv::Point(0, -(*it)[2] / (*it)[1]),
			cv::Point(image2.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]),
			cv::Scalar(255, 255, 255));
	}

	// draw the left points corresponding epipolar lines in left image
	std::vector<cv::Vec3f> lines2;
	cv::computeCorrespondEpilines(cv::Mat(selPoints2), 2, fundemental, lines2);
	for (auto it = lines2.begin(); it != lines2.end(); ++it)
	{

		// draw the epipolar line between first and last column
		cv::line(image1, cv::Point(0, -(*it)[2] / (*it)[1]),
			cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]),
			cv::Scalar(255, 255, 255));
	}

	// Display the images with points and epipolar lines
	cv::namedWindow("Right Image Epilines");
	cv::imshow("Right Image Epilines", image1);
	cv::namedWindow("Left Image Epilines");
	cv::imshow("Left Image Epilines", image2);

	/*
	std::nth_element(matches.begin(),    // initial position
					 matches.begin()+matches.size()/2, // 50%
					 matches.end());     // end position
	// remove all elements after the
	matches.erase(matches.begin()+matches.size()/2, matches.end());
*/
// Convert keypoints into Point2f
	std::vector<cv::Point2f> points1, points2;
	for (auto it = matches.begin(); it != matches.end(); ++it)
	{

		// Get the position of left keypoints
		float x = keypoints1[it->queryIdx].pt.x;
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x, y));
		// Get the position of right keypoints
		x = keypoints2[it->trainIdx].pt.x;
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x, y));
	}

	std::cout << points1.size() << " " << points2.size() << std::endl;

	// Compute F matrix using RANSAC
	std::vector<uchar> inliers(points1.size(), 0);
	fundemental = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2), // matching points
		inliers,      // match status (inlier ou outlier)
		cv::FM_RANSAC, // RANSAC method
		1,            // distance to epipolar line
		0.98);        // confidence probability

// Read input images
	image1 = cv::imread("img1.jpg", 0);
	image2 = cv::imread("img2.jpg", 0);

	// Draw the epipolar line of few points
	cv::computeCorrespondEpilines(cv::Mat(selPoints1), 1, fundemental, lines1);
	for (auto it = lines1.begin(); it != lines1.end(); ++it)
	{

		cv::line(image2, cv::Point(0, -(*it)[2] / (*it)[1]),
			cv::Point(image2.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]),
			cv::Scalar(255, 255, 255));
	}

	cv::computeCorrespondEpilines(cv::Mat(selPoints2), 2, fundemental, lines2);
	for (auto it = lines2.begin(); it != lines2.end(); ++it)
	{

		cv::line(image1, cv::Point(0, -(*it)[2] / (*it)[1]),
			cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]),
			cv::Scalar(255, 255, 255));
	}

	// Draw the inlier points
	std::vector<cv::Point2f> points1In, points2In;
	auto itPts = points1.begin();
	auto itIn = inliers.begin();
	while (itPts != points1.end())
	{

		// draw a circle at each inlier location
		if (*itIn)
		{
			cv::circle(image1, *itPts, 3, cv::Scalar(255, 255, 255), 2);
			points1In.push_back(*itPts);
		}
		++itPts;
		++itIn;
	}

	itPts = points2.begin();
	itIn = inliers.begin();
	while (itPts != points2.end())
	{

		// draw a circle at each inlier location
		if (*itIn)
		{
			cv::circle(image2, *itPts, 3, cv::Scalar(255, 255, 255), 2);
			points2In.push_back(*itPts);
		}
		++itPts;
		++itIn;
	}

	// Display the images with points
	cv::namedWindow("Right Image Epilines (RANSAC)");
	cv::imshow("Right Image Epilines (RANSAC)", image1);
	cv::namedWindow("Left Image Epilines (RANSAC)");
	cv::imshow("Left Image Epilines (RANSAC)", image2);

	cv::findHomography(cv::Mat(points1In), cv::Mat(points2In), inliers, cv::RANSAC, 1.);

	// Read input images
	image1 = cv::imread("img1.jpg", 0);
	image2 = cv::imread("img2.jpg", 0);

	// Draw the inlier points
	itPts = points1In.begin();
	itIn = inliers.begin();
	while (itPts != points1In.end())
	{

		// draw a circle at each inlier location
		if (*itIn)
			cv::circle(image1, *itPts, 3, cv::Scalar(255, 255, 255), 2);

		++itPts;
		++itIn;
	}

	itPts = points2In.begin();
	itIn = inliers.begin();
	while (itPts != points2In.end())
	{

		// draw a circle at each inlier location
		if (*itIn)
			cv::circle(image2, *itPts, 3, cv::Scalar(255, 255, 255), 2);

		++itPts;
		++itIn;
	}

	// Display the images with points
	cv::namedWindow("Right Image Homography (RANSAC)");
	cv::imshow("Right Image Homography (RANSAC)", image1);
	cv::namedWindow("Left Image Homography (RANSAC)");
	cv::imshow("Left Image Homography (RANSAC)", image2);

	cv::waitKey();
	return;
}