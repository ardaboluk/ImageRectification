#pragma once

class Estimator {
private:
	// 8x9 homogeneous linear system
	float hms[8][9];

	void buildHMSMatrix(float**);
public:
	// it's argument is corresponding points in the form {{x_1, y_1, x'_1, y'_1}, {x_2, y_2, x'_2, y'_2}, ...}
	float** estimateFundamentalMatrix(float**);
};