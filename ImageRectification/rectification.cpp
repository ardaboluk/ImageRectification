
#include "rectification.h"
#include "util.h"

/*
* Returns an array of epipolar lines.
* Each row represents 2 epipolar lines.
* The first one is the line on the second image corresponding the to point in the first image.
* The second one is the line on the first image corresponding the to point in the second image.
*/
float** Rectification::getEpilines(float** pointCorrespondences, int numPoints, double** fundamentalMatrix) {
	
	float** epilines = new float* [numPoints];
	for (int i = 0; i < numPoints; i++) {
		epilines[i] = new float[6];
	}

	double** fundamental_transpose = Util::transpose(fundamentalMatrix, 3, 3);

	for (int i = 0; i < numPoints; i++) {
		double** point1 = new double* [3];
		double** point2 = new double* [3];
		for (int j = 0; j < 3; j++) {
			point1[j] = new double[1];
			point2[j] = new double[1];
		}
		
		point1[0][0] = pointCorrespondences[i][0];
		point1[1][0] = pointCorrespondences[i][1];
		point1[2][0] = 1.0;
		point2[0][0] = pointCorrespondences[i][2];
		point2[1][0] = pointCorrespondences[i][3];
		point2[2][0] = 1.0;

		double** line1 = Util::matMul(fundamentalMatrix, point1, 3, 3, 3, 1);
		double** line2 = Util::matMul(fundamental_transpose, point2, 3, 3, 3, 1);

		for (int j = 0; j < 3; j++) {
			epilines[i][j] = (float)line1[j][0];
			epilines[i][j + 3] = (float)line2[j][0];
		}

		for (int j = 0; j < 3; j++) {
			delete[] point1[j];
			delete[] point2[j];
			delete[] line1[j];
			delete[] line2[j];
		}
		delete[] point1;
		delete[] point2;
		delete[] line1;
		delete[] line2;
	}

	for (int i = 0; i < 3; i++) {
		delete[] fundamental_transpose[i];
	}
	delete[] fundamental_transpose;

	return epilines;
}

void Rectification::drawEpilines(float** epilines, int numLines, cv::Mat image1, cv::Mat image2) {

	for (int i = 0; i < numLines; i++) {
		float a1 = epilines[i][0], b1 = epilines[i][1], c1 = epilines[i][2];
		float a2 = epilines[i][3], b2 = epilines[i][4], c2 = epilines[i][5];

		float p1StartX = 0;
		float p1StartY = -c1 / b1;
		float p1EndX = image1.cols;
		float p1EndY = -(a1 * image1.cols + c1) / b1;
		cv::Point2f p1Start(p1StartX, p1StartY);
		cv::Point2f p1End(p1EndX, p1EndY);

		float p2StartX = 0;
		float p2StartY = -c2 / b2;
		float p2EndX = image2.cols;
		float p2EndY = -(a2 * image2.cols + c2) / b2;
		cv::Point2f p2Start(p2StartX, p2StartY);
		cv::Point2f p2End(p2EndX, p2EndY);

		cv::line(image1, p1Start, p1End, cv::Scalar(0, 0, 0));
		cv::line(image2, p2Start, p2End, cv::Scalar(0, 0, 0));
	}
}