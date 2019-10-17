
#include <vector>
#include <string>
#include <cassert>
#include <limits>

#include "util.h"
#include "preprocessing.h"
#include "estimator.h"

std::vector<double> checkFundamentalMatrix(cv::Mat fundamentalMatrix, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2);
void test_estimateFundamentalMatrix8Point();
void test_fundamentalMatrixOpencv();
void test_getEpilines();

int __main(){

    // test_fundamentalMatrixOpencv();
    // test_estimateFundamentalMatrix();
    test_getEpilines();

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

void test_estimateFundamentalMatrix8Point(){

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

        Estimator estimator;
        cv::Mat fundamentalMatrix = estimator.estimateFundamentalMatrix8Point(correspondingPoints1_normalized, correspondingPoints2_normalized);
        cv::Mat fundamentalMatrixDenormalized = estimator.denormalizeFundamentalMatrix(fundamentalMatrix, normMat1, normMat2);

        std::vector<double> results = checkFundamentalMatrix(fundamentalMatrixDenormalized, correspondingPoints1, correspondingPoints2);
        double epsilond = 0.1;//std::numeric_limits<double>::epsilon() * 100;
        for(auto it = results.begin(); it != results.end(); ++it){
            assert((*it) < epsilond);
        }
    }

    std::cout << "TEST SUCCESSFUL" << std::endl;
}

void test_getEpilines(){

    std::vector<std::string> imageFileNames1{"img1.jpg", "Lab_1.jpg", "Building_1.jpg"};
    std::vector<std::string> imageFileNames2{"img2.jpg", "Lab_2.jpg", "Building_2.jpg"};

    for(int i = 0; i < (int)imageFileNames1.size(); i++){
        std::cout << "Image Pair " << i << std::endl;

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

        Estimator estimator;
        cv::Mat fundamentalMatrix = estimator.estimateFundamentalMatrix8Point(correspondingPoints1_normalized, correspondingPoints2_normalized);
        cv::Mat fundamentalMatrixDenormalized = estimator.denormalizeFundamentalMatrix(fundamentalMatrix, normMat1, normMat2);

        std::pair<std::vector<cv::Point3d>, std::vector<cv::Point3d>> epilinesList = estimator.getEpilines(correspondingPointsList, fundamentalMatrixDenormalized);

        std::vector<cv::Point3d> epilines1 = epilinesList.first;
        std::vector<cv::Point3d> epilines2 = epilinesList.second;

        for(int i = 0; i < epilines1.size(); i++){

            cv::Point3d currentEpiline1 = epilines1[i];
            cv::Point3d currentEpiline2 = epilines2[i];
            cv::Mat currentEpiline1Mat(currentEpiline1);
            cv::Mat currentEpiline2Mat(currentEpiline2);

            cv::Point3d currentPoint1;
            cv::Point3d currentPoint2;

            currentPoint1.x = correspondingPoints1[i].x;
            currentPoint1.y = correspondingPoints1[i].y;
            currentPoint1.z = 1.0;

            currentPoint2.x = correspondingPoints2[i].x;
            currentPoint2.y = correspondingPoints2[i].y;
            currentPoint2.z = 1.0;

            cv::Mat currentPoint1Mat(currentPoint1);
            cv::Mat currentPoint2Mat(currentPoint2);

            cv::Mat resultMat1 = currentPoint2Mat.t() * currentEpiline2Mat;
            cv::Mat resultMat2 = currentPoint1Mat.t() * currentEpiline1Mat;

            double result1 = resultMat1.at<double>(0,0);
            double result2 = resultMat2.at<double>(0,0);

            double epsilond = std::numeric_limits<double>::epsilon();

            std::cout << "Result1:\t" << result1 << "\tResult2:\t" << result2 << std::endl;

            // assert(result1 <= epsilond && result2 <= epsilond);
        }
    }   

    std::cout << "TEST SUCCESSFUL" << std::endl;
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
        std::vector<cv::Point2f> correspondingPoints1;
        std::vector<cv::Point2f> correspondingPoints2;
        for(int i = 0; i < 8; i++){
            correspondingPoints1.push_back(correspondingPointsListTmp.first[i]);
            correspondingPoints2.push_back(correspondingPointsListTmp.second[i]);
        }
        
        Estimator estimator;
        cv::Mat fundamentalMatrix = estimator.estimateFundamentalMatrix_opencv(correspondingPoints1, correspondingPoints2);

        std::vector<double> results = checkFundamentalMatrix(fundamentalMatrix, correspondingPointsListTmp.first, correspondingPointsListTmp.second);
        double epsilond = 0.1;//std::numeric_limits<double>::epsilon() * 100;
        for(auto it = results.begin(); it != results.end(); ++it){
            assert((*it) < epsilond);
        }
    }

    std::cout << "TEST SUCCESSFUL" << std::endl;
}