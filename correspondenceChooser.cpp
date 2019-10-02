
#include "opencv2/opencv.hpp"
#include "correspondenceChooser.h"
#include <iostream>

Point_Image CorrespondenceChooser::imageToChooseFrom;
std::vector<float*> CorrespondenceChooser::pointCorrepondences;
cv::Mat CorrespondenceChooser::image1, CorrespondenceChooser::image2;
cv::Mat CorrespondenceChooser::imageToDisplay1, CorrespondenceChooser::imageToDisplay1WithText, 
	CorrespondenceChooser::imageToDisplay2, CorrespondenceChooser::imageToDisplay2WithText;
cv::Mat CorrespondenceChooser::resultImageToDisplay;
float CorrespondenceChooser::currentImage1PointX, CorrespondenceChooser::currentImage1PointY, 
CorrespondenceChooser::currentImage2PointX, CorrespondenceChooser::currentImage2PointY;
bool CorrespondenceChooser::isWindowClosed;
int CorrespondenceChooser::numCorrespondences;

void CorrespondenceChooser::initializeImages(cv::Mat image1, cv::Mat image2) {
	CorrespondenceChooser::image1 = image1.clone();
	CorrespondenceChooser::image2 = image2.clone();
	CorrespondenceChooser::imageToDisplay1 = image1.clone();
	CorrespondenceChooser::imageToDisplay2 = image2.clone();
	CorrespondenceChooser::imageToDisplay1WithText = image1.clone();
	CorrespondenceChooser::imageToDisplay2WithText = image2.clone();
	CorrespondenceChooser::isWindowClosed = false;
	CorrespondenceChooser::numCorrespondences = 0;

	cv::Mat matArray[] = { image1, image2 };
	cv::hconcat(matArray, 2, resultImageToDisplay);
}

void CorrespondenceChooser::CallBackFunc(int event, int x, int y, int flags, void* userdata){
	if (event == cv::EVENT_LBUTTONDOWN) {
		if (x < image1.cols && imageToChooseFrom == Point_Image::IMAGE1) {
			currentImage1PointX = x;
			currentImage1PointY = y;
			imageToChooseFrom = Point_Image::IMAGE2;

			cv::circle(imageToDisplay1, cv::Point2f(currentImage1PointX, currentImage1PointY), MARK_RADIUS, MARKER_COLOR, -1);
			imageToDisplay1WithText = imageToDisplay1.clone();
		}
		else if (x >= image1.cols && imageToChooseFrom == Point_Image::IMAGE2) {
			currentImage2PointX = x - image1.cols;
			currentImage2PointY = y;
			imageToChooseFrom = Point_Image::IMAGE1;

			float* currentCorrespondence = new float[4];
			currentCorrespondence[0] = currentImage1PointX;
			currentCorrespondence[1] = currentImage1PointY;
			currentCorrespondence[2] = currentImage2PointX;
			currentCorrespondence[3] = currentImage2PointY;

			pointCorrepondences.push_back(currentCorrespondence);

			numCorrespondences++;

			cv::circle(imageToDisplay2, cv::Point2f(currentImage2PointX, currentImage2PointY), MARK_RADIUS, MARKER_COLOR, -1);
			imageToDisplay2WithText = imageToDisplay2.clone();
		}
	}
	else if (event == cv::EVENT_RBUTTONDOWN) {
		isWindowClosed = true;
	}
	else if (event == cv::EVENT_MOUSEMOVE) {
		if (x < image1.cols) {			
			std::string coordsString = "x: " + std::to_string(x) + " y: " + std::to_string(y) + " #p: " + std::to_string(numCorrespondences);
			imageToDisplay1WithText = imageToDisplay1.clone();
			cv::putText(imageToDisplay1WithText, coordsString, cv::Point2f(0, image1.rows - TXT_OFFSET_Y), cv::FONT_HERSHEY_DUPLEX, 1, TXT_COLOR);
		}
		else {
			std::string coordsString = "x: " + std::to_string(x - image1.cols) + " y: " + std::to_string(y) + " #p: " + std::to_string(numCorrespondences);
			imageToDisplay2WithText = imageToDisplay2.clone();
			cv::putText(imageToDisplay2WithText, coordsString, cv::Point2f(0, image2.rows - TXT_OFFSET_Y), cv::FONT_HERSHEY_DUPLEX, 1, TXT_COLOR);
		}
	}
	if (imageToDisplay1WithText.dims != 0 && imageToDisplay2WithText.dims != 0) {
		cv::Mat matArray[] = { imageToDisplay1WithText, imageToDisplay2WithText };
		cv::hconcat(matArray, 2, resultImageToDisplay);
	}
}

float** CorrespondenceChooser::getPointCorrespondences() {
	if (pointCorrepondences.size() == 0) {
		return NULL;
	}
	float** correspondences = new float* [pointCorrepondences.size()];
	for (unsigned int i = 0; i < pointCorrepondences.size(); i++) {
		correspondences[i] = new float[4];
		correspondences[i][0] = pointCorrepondences.at(i)[0];
		correspondences[i][1] = pointCorrepondences.at(i)[1];
		correspondences[i][2] = pointCorrepondences.at(i)[2];
		correspondences[i][3] = pointCorrepondences.at(i)[3];
	}
	for (unsigned int i = 0; i < pointCorrepondences.size(); i++) {
		delete[] pointCorrepondences.at(i);
	}
	return correspondences;
}

cv::Mat CorrespondenceChooser::getResultImageToDisplay() {
	return resultImageToDisplay;
}