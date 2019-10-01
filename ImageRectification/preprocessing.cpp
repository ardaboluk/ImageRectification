
#include <cmath>
#include <iostream>

#include "preprocessing.h"
#include "util.h"

double** Preprocessing::getNormalizationMat(float imageWidth, float imageHeight) {

	double** transformMat1 = new double* [3];
	for (int i = 0; i < 3; i++) {
		transformMat1[i] = new double[3];
	}

	transformMat1[0][0] = 2.0 / imageWidth;
	transformMat1[0][1] = 0;
	transformMat1[0][2] = -1.0;
	transformMat1[1][0] = 0;
	transformMat1[1][1] = 2.0 / imageHeight;
	transformMat1[1][2] = -1.0;
	transformMat1[2][0] = 0;
	transformMat1[2][1] = 0;
	transformMat1[2][2] = 1;

	return transformMat1;
}

// TODO: should re-write using matMul
float** Preprocessing::normalizeCoordinates(float** pointCorrespondences, float image1Width, float image1Height, float image2Width, float image2Height, int numPoints) {

	float** normalizedCorrespondences = new float* [numPoints];
	for (int i = 0; i < numPoints; i++) {
		normalizedCorrespondences[i] = new float[4];
		normalizedCorrespondences[i][0] = 2 * pointCorrespondences[i][0] / image1Width - 1;
		normalizedCorrespondences[i][1] = 2 * pointCorrespondences[i][1] / image1Height - 1;
		normalizedCorrespondences[i][2] = 2 * pointCorrespondences[i][2] / image2Width - 1;
		normalizedCorrespondences[i][3] = 2 * pointCorrespondences[i][3] / image2Height - 1;
	}
	return normalizedCorrespondences;
}

double** Preprocessing::denormalizeFundamentalMatrix(double** fmat, float image1Width, float image1Height, float image2Width, float image2Height) {

	double** normalizationMat1 = getNormalizationMat(image1Width, image1Height);
	double** normalizationMat2 = getNormalizationMat(image2Width, image2Height);

	double** tMat2Tr = Util::transpose(normalizationMat2, 3, 3);
	double** tmpMul = Util::matMul(tMat2Tr, fmat, 3, 3, 3, 3);
	double** denormalizedFMat = Util::matMul(tmpMul, normalizationMat1, 3, 3, 3, 3);

	for (int i = 0; i < 3; i++) {
		delete[] normalizationMat1[i];
		delete[] normalizationMat2[i];
		delete[] tMat2Tr[i];
		delete[] tmpMul[i];
	}

	delete[] normalizationMat1;
	delete[] normalizationMat2;
	delete[] tMat2Tr;
	delete[] tmpMul;

	return denormalizedFMat;
}