#pragma once

#include <vector>
#include <utility>
#include <string>
#include "opencv2/opencv.hpp"

class Util {
public:
	static double** matMul(double** mat1, double** mat2, int mat1Rows, int mat1Cols, int mat2Rows, int mat2Cols);
	static double** transpose(double** mat, int numRows, int numCols);
	static void displayMat(cv::Mat& cvMatrix, std::string explanation);
	static void displayMat(double** matrix, int numRows, int numCols, std::string explanation);
	static void displayMat(float** matrix, int numRows, int numCols, std::string explanation);
	static std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> extractMatches(std::string image1Filename, std::string image2Filename);
};
