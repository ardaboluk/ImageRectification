
#include <cmath>
#include <iostream>

#include "preprocessing.h"

// Reference: github.com/DzReal/Normalized-Eight-Point-Algorithm
// Reference: stackoverflow.com/questions/52940822/what-is-the-correct-way-to-normalize-corresponding-points-before-estimation-of-f

std::vector<cv::Point3f> Preprocessing::transformPointsToHomogen(std::vector<cv::Point2f> points){
	std::vector<cv::Point3f> pointsHomogen;
	for(std::vector<cv::Point2f>::iterator it = points.begin(); it != points.end(); ++it){
		cv::Point3f tmpPoint;
		tmpPoint.x = (*it).x; 
		tmpPoint.y = (*it).y; 
		tmpPoint.z = 1.0f; 
		pointsHomogen.push_back(tmpPoint);
	}
	return pointsHomogen;
}

std::vector<cv::Point2f> Preprocessing::transformPointsToNonHomogen(std::vector<cv::Point3f> points){
	std::vector<cv::Point2f> nonHomogenPoints;
	for(auto it = points.begin(); it != points.end(); ++it){
		cv::Point2f tmpPoint;
		tmpPoint.x = (*it).x / (*it).z;
		tmpPoint.y = (*it).y / (*it).z;
		nonHomogenPoints.push_back(tmpPoint);
	}
	return nonHomogenPoints;
}

cv::Mat Preprocessing::getNormalizationMat(std::vector<cv::Point3f> points) {

	cv::Mat transformMat(cv::Size(3,3), CV_32FC1);

	// find mean
	cv::Point3f centroid;
	for(int i = 0; i < points.size(); i++){
		centroid += points[i];
	}
	centroid.x /= points.size();
	centroid.y /= points.size();
	centroid.z /= points.size();

	// compute the average distance to the centroid
	float avgDistance;
	for(int i = 0; i < points.size(); i++){
		cv::Point3f diffPoint = points[i] - centroid;
		avgDistance += sqrtf(powf(diffPoint.x, 2.0f) + powf(diffPoint.y, 2.0f) + powf(diffPoint.z, 2.0f));
	}
	avgDistance /= points.size();

	// craft the normalization matrix
	float sqrt2f = sqrtf(2.0f);
	
	transformMat.at<float>(0,0) = sqrt2f / avgDistance;
	transformMat.at<float>(0,1) = 0;
	transformMat.at<float>(0,2) = -sqrt2f / avgDistance * centroid.x;

	transformMat.at<float>(1,0) = 0;
	transformMat.at<float>(1,1) = sqrt2f / avgDistance;
	transformMat.at<float>(1,2) = -sqrt2f / avgDistance * centroid.y;

	transformMat.at<float>(2,0) = 0;
	transformMat.at<float>(2,1) = 0;
	transformMat.at<float>(2,2) = 1;

	return transformMat;
}

std::vector<cv::Point3f> Preprocessing::normalizeCoordinates(std::vector<cv::Point3f> points, cv::Mat normalizationMat) {
	std::vector<cv::Point3f> normalizedPoints;
	for (std::vector<cv::Point3f>::iterator it = points.begin(); it != points.end(); ++it) {
		cv::Mat tmpPointMat((*it), true);
		cv::Mat normalizedPointsMat = normalizationMat * tmpPointMat;
		cv::Point3f tmpPointNormalized;
		tmpPointNormalized.x = normalizedPointsMat.at<float>(0,0);
		tmpPointNormalized.y = normalizedPointsMat.at<float>(1,0);
		tmpPointNormalized.z = normalizedPointsMat.at<float>(2,0);
		normalizedPoints.push_back(tmpPointNormalized);
	}
	return normalizedPoints;
}