
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
		tmpPoint.x = (float)((double)(*it).x / (*it).z);
		tmpPoint.y = (float)((double)(*it).y / (*it).z);
		nonHomogenPoints.push_back(tmpPoint);
	}
	return nonHomogenPoints;
}

cv::Mat Preprocessing::getNormalizationMat(std::vector<cv::Point3f> points) {

	std::vector<cv::Point3d> pointsd;
	for(auto it = points.begin(); it != points.end(); ++it){
		cv::Point3d tmpPoint;
		tmpPoint.x = (double)(*it).x;
		tmpPoint.y = (double)(*it).y;
		tmpPoint.z = (double)(*it).z;
		pointsd.push_back(tmpPoint);
	}
	cv::Mat transformMat(cv::Size(3,3), CV_64FC1);

	// find mean
	cv::Point3d centroid;
	for(int i = 0; i < (int)pointsd.size(); i++){
		centroid += pointsd[i];
	}
	centroid.x /= pointsd.size();
	centroid.y /= pointsd.size();
	centroid.z /= pointsd.size();

	// compute the average distance to the centroid
	double avgDistance;
	for(int i = 0; i < (int)pointsd.size(); i++){
		cv::Point3d diffPoint = pointsd[i] - centroid;
		avgDistance += sqrtf(pow(diffPoint.x, 2.0) + pow(diffPoint.y, 2.0) + pow(diffPoint.z, 2.0));
	}
	avgDistance /= pointsd.size();

	// craft the normalization matrix
	double sqrt2d = sqrt(2.0);
	
	transformMat.at<double>(0,0) = sqrt2d / avgDistance;
	transformMat.at<double>(0,1) = 0;
	transformMat.at<double>(0,2) = -sqrt2d / avgDistance * centroid.x;

	transformMat.at<double>(1,0) = 0;
	transformMat.at<double>(1,1) = sqrt2d / avgDistance;
	transformMat.at<double>(1,2) = -sqrt2d / avgDistance * centroid.y;

	transformMat.at<double>(2,0) = 0;
	transformMat.at<double>(2,1) = 0;
	transformMat.at<double>(2,2) = 1;

	return transformMat;
}

std::vector<cv::Point3f> Preprocessing::normalizeCoordinates(std::vector<cv::Point3f> points, cv::Mat normalizationMat) {
	std::vector<cv::Point3f> normalizedPoints;
	for (std::vector<cv::Point3f>::iterator it = points.begin(); it != points.end(); ++it) {
		cv::Point3d tmpPoint3d;
		tmpPoint3d.x = (double)(*it).x;
		tmpPoint3d.y = (double)(*it).y;
		tmpPoint3d.z = (double)(*it).z;
		cv::Mat tmpPoint3dMat(tmpPoint3d, true);
		cv::Mat normalizedPointsMat = normalizationMat * tmpPoint3dMat;
		cv::Point3f tmpPointNormalized;
		tmpPointNormalized.x = (float)normalizedPointsMat.at<double>(0,0);
		tmpPointNormalized.y = (float)normalizedPointsMat.at<double>(1,0);
		tmpPointNormalized.z = (float)normalizedPointsMat.at<double>(2,0);
		normalizedPoints.push_back(tmpPointNormalized);
	}
	return normalizedPoints;
}