#pragma once

#include <vector>
#include <string>

#include "opencv2/opencv.hpp"

class Rectification {
private:
	cv::Mat image1;
	cv::Mat image2;

	cv::Mat warpImage(cv::Mat image, cv::Mat homography);

public:
	Rectification(std::string image1FileName, std::string image2FileName);
	std::pair<cv::Mat, cv::Mat> rectifyImages();
	std::pair<std::vector<cv::Point3d>, std::vector<cv::Point3d>> getEpilines(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondenceList, cv::Mat fundamentalMatrix);
	void drawEpilines(std::pair<std::vector<cv::Point3d>, std::vector<cv::Point3d>> epilines, cv::Mat image1, cv::Mat image2);
	
	static std::vector<cv::Mat> getEpilinesDebug(float** pointCorrespondences, int numPoints, cv::Mat fundamentalMatrix);
	
	static void drawEpilinesDebug(std::vector<cv::Mat> epilines, int numLines, cv::Mat image1, cv::Mat image2);
	// H1 and H2 2d arrays should be allocated before calling this method
	// Also, images that will be rectified should be of the same size
	static void rectifyUncalibratedCV(float** pointCorrespondences, int numCorrespondences, double** fundamentalMatrix, 
		int imageRows, int imageCols, double** H1, double** H2);
	static void rectifyUncalibratedDebug(float** pointCorrespondences, int numCorrespondences, cv::Mat fundamentalMatrix,
		int imageRows, int imageCols, cv::Mat H1, cv::Mat H2);
};
