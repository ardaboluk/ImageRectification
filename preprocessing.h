#pragma once

class Preprocessing {

private:
	static double** getNormalizationMat(float imageWidth, float imageHeight);
public:
	/*
	* pointCorresponces shoud be in format (X,Y,X',Y')
	*/
	static float** normalizeCoordinates(float** pointCorresponces, float image1Width, float image1Height, float image2Width, float image2Height, int numPoints);

	static double** denormalizeFundamentalMatrix(double** fmat, float image1Width, float image1Height, float image2Width, float image2Height);
};
