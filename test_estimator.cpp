
#include <vector>
#include <string>
#include <cassert>
#include <limits>

#include "util.h"
#include "preprocessing.h"
#include "estimator.h"

std::vector<double> checkFundamentalMatrix(cv::Mat fundamentalMatrix, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2);
void test_estimateFundamentalMatrix();
void test_fundamentalMatrixOpencv();

int __main(){

    test_fundamentalMatrixOpencv();
    test_estimateFundamentalMatrix();

    return 0;
}

std::vector<double> checkFundamentalMatrix(cv::Mat fundamentalMatrix, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2) {
    std::vector<double> results;
	std::cout << "Verifying the fundamental matrix using line equation." << std::endl;
	for (int i = 0; i < (int)points1.size(); i++) {
		cv::Mat point1Mat(cv::Size(1,3), CV_64FC1);
        cv::Mat point2Mat(cv::Size(1,3), CV_64FC1);
        point1Mat.at<double>(0,0) = (double)points1[i].x;
        point1Mat.at<double>(1,0) = (double)points1[i].y;
        point1Mat.at<double>(2,0) = 1.0;
        point2Mat.at<double>(0,0) = (double)points2[i].x;
        point2Mat.at<double>(1,0) = (double)points2[i].y;
        point2Mat.at<double>(2,0) = 1.0;

		cv::Mat resultMat = point2Mat.t() * fundamentalMatrix * point1Mat;
        results.push_back(resultMat.at<double>(0,0));

		std::cout << resultMat << std::endl;
	}
	return results;
}


void test_estimateFundamentalMatrix(){

    std::vector<std::string> imageFileNames1{"img1.jpg", "Lab_1.jpg", "Building_1.jpg"};
    std::vector<std::string> imageFileNames2{"img2.jpg", "Lab_2.jpg", "Building_2.jpg"};

    for(int i = 0; i < (int)imageFileNames1.size(); i++){
        std::string currentImageFileName1 = imageFileNames1[i];
        std::string currentImageFileName2 = imageFileNames2[i];
        cv::Mat image1 = cv::imread(currentImageFileName1);
        cv::Mat image2 = cv::imread(currentImageFileName2);

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

        std::vector<double> results = checkFundamentalMatrix(fundamentalMatrixDenormalized, correspondingPoints1, correspondingPoints2);
        double epsilond = std::numeric_limits<double>::epsilon() * 100;
        for(auto it = results.begin(); it != results.end(); ++it){
            assert((*it) < epsilond);
        }
    }
}

void test_fundamentalMatrixOpencv(){

    std::vector<std::string> imageFileNames1{"img1.jpg", "Lab_1.jpg", "Building_1.jpg"};
    std::vector<std::string> imageFileNames2{"img2.jpg", "Lab_2.jpg", "Building_2.jpg"};

    for(int i = 0; i < (int)imageFileNames1.size(); i++){
        std::string currentImageFileName1 = imageFileNames1[i];
        std::string currentImageFileName2 = imageFileNames2[i];
        cv::Mat image1 = cv::imread(currentImageFileName1);
        cv::Mat image2 = cv::imread(currentImageFileName2);

        std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsListTmp = Util::extractMatches(image1, image2, 8);
        std::vector<cv::Point2f> correspondingPoints1Tmp;
        std::vector<cv::Point2f> correspondingPoints2Tmp;
        for(int i = 0; i < 8; i++){
            correspondingPoints1Tmp.push_back(correspondingPointsListTmp.first[i]);
            correspondingPoints2Tmp.push_back(correspondingPointsListTmp.second[i]);
        }
        std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList;
        correspondingPointsList.first = correspondingPoints1Tmp;
        correspondingPointsList.second = correspondingPoints2Tmp;

        std::vector<cv::Point2f> correspondingPoints1 = correspondingPointsList.first;
        std::vector<cv::Point2f> correspondingPoints2 = correspondingPointsList.second;
        
        Estimator estimator(correspondingPointsList);
        cv::Mat fundamentalMatrix = estimator.estimateFundamentalMatrix_opencv();

        std::vector<double> results = checkFundamentalMatrix(fundamentalMatrix, correspondingPointsListTmp.first, correspondingPointsListTmp.second);
        double epsilond = std::numeric_limits<double>::epsilon() * 100;
        for(auto it = results.begin(); it != results.end(); ++it){
            assert((*it) < epsilond);
        }
    }
}
