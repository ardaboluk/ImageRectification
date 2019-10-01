
#include <iostream>
#include "util.h"

double** Util::matMul(double** mat1, double** mat2, int mat1Rows, int mat1Cols, int mat2Rows, int mat2Cols) {
	if (mat1Cols != mat2Rows) {
		return NULL;
	}

	double** resultMat = new double* [mat1Rows];
	for (int i = 0; i < mat1Rows; i++) {
		resultMat[i] = new double[mat2Cols];
	}

	for (int i = 0; i < mat1Rows; i++) {
		for (int j = 0; j < mat2Cols; j++) {
			double currentElement = 0;
			for (int k = 0; k < mat1Cols; k++) {
				currentElement += mat1[i][k] * mat2[k][j];
			}
			resultMat[i][j] = currentElement;
		}
	}

	return resultMat;
}

double** Util::transpose(double** mat, int numRows, int numCols) {

	double** resultMat = new double* [numCols];
	for (int i = 0; i < numCols; i++) {
		resultMat[i] = new double[numRows];
		for (int j = 0; j < numRows; j++) {
			resultMat[i][j] = mat[j][i];
		}
	}
	return resultMat;
}

void Util::displayMat(cv::Mat& cvMatrix, std::string explanation) {
	std::cout << explanation << std::endl;
	std::cout << cvMatrix << std::endl;
	std::cout << std::endl;
}

void Util::displayMat(double** matrix, int numRows, int numCols, std::string explanation) {
	std::cout << explanation << std::endl;
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			std::cout << matrix[i][j] << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void Util::displayMat(float** matrix, int numRows, int numCols, std::string explanation) {
	std::cout << explanation << std::endl;
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			std::cout << matrix[i][j] << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}