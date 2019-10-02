#pragma once

#include <vector>

#include"opencv2/opencv.hpp"

#define MARK_RADIUS 5
#define TXT_OFFSET_Y 20		// will be subtracted from image height
#define MARKER_COLOR cv::Scalar(0,0,255)
#define TXT_COLOR cv::Scalar(0,0,255)

enum Point_Image { IMAGE1, IMAGE2 };

class CorrespondenceChooser {
	/*
	* Class for GUI that shows two images (that are possibly related by some perspective transformation)
	* and allows user to select the corresponding points between the images. It then returns the coordinates
	* of the points.
	*/
private:
	static Point_Image imageToChooseFrom;
	static std::vector<float*> pointCorrepondences;
	static cv::Mat image1, image2;
	static cv::Mat imageToDisplay1, imageToDisplay1WithText, imageToDisplay2, imageToDisplay2WithText;
	static cv::Mat resultImageToDisplay;
	static float currentImage1PointX, currentImage1PointY, currentImage2PointX, currentImage2PointY;
public:
	static bool isWindowClosed;
	static int numCorrespondences;

	CorrespondenceChooser() = delete;
	static void initializeImages(cv::Mat image1, cv::Mat image2);
	static void CallBackFunc(int event, int x, int y, int, void*);
	static float** getPointCorrespondences();
	static cv::Mat getResultImageToDisplay();
};
