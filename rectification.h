#pragma once

#include <vector>
#include <string>

#include "opencv2/opencv.hpp"

class Rectification {
private:
	cv::Mat image1;
	cv::Mat image2;

public:
	Rectification(std::string image1FileName, std::string image2FileName);
	void rectifyImages();

	static float** getEpilines(float** pointCorrespondences, int numPoints, double** fundamentalMatrix);
	static std::vector<cv::Mat> getEpilinesDebug(float** pointCorrespondences, int numPoints, cv::Mat fundamentalMatrix);
	static void drawEpilines(float** epilines, int numLines, cv::Mat image1, cv::Mat image2);
	static void drawEpilinesDebug(std::vector<cv::Mat> epilines, int numLines, cv::Mat image1, cv::Mat image2);
	// H1 and H2 2d arrays should be allocated before calling this method
	// Also, images that will be rectified should be of the same size
	static void rectifyUncalibratedCV(float** pointCorrespondences, int numCorrespondences, double** fundamentalMatrix, 
		int imageRows, int imageCols, double** H1, double** H2);
	static void rectifyUncalibratedDebug(float** pointCorrespondences, int numCorrespondences, cv::Mat fundamentalMatrix,
		int imageRows, int imageCols, cv::Mat H1, cv::Mat H2);
};
