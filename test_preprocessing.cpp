

#include <vector>
#include <random>
#include <cassert>
#include "preprocessing.h"

#define NUM_TEST_POINTS 100
#define X_COORD_MIN 0
#define Y_COORD_MIN 0
#define X_COORD_MAX 700
#define Y_COORD_MAX 700

float randomFloat(float a, float b);
std::vector<cv::Point2f> getRandomPoints2f();

void test_transformPointsToHomogen();
void test_getNormalizationMat();

int main(){

    test_transformPointsToHomogen();

    return 0;
}

float randomFloat(float a, float b){
    float random = ((float)rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

std::vector<cv::Point2f> getRandomPoints2f(){
    std::vector<cv::Point2f> randPoints;
    for(int i = 0; i < NUM_TEST_POINTS; i++){
        
    }
    return randPoints;
}

void test_transformPointsToHomogen(){
    std::vector<cv::Point2f> testPoints;
    for(int i = 0; i < NUM_TEST_POINTS; i++){
        cv::Point2f tmpTestPoint;
        tmpTestPoint.x = randomFloat(X_COORD_MIN, X_COORD_MAX);
        tmpTestPoint.y = randomFloat(Y_COORD_MIN, Y_COORD_MAX);
        testPoints.push_back(tmpTestPoint);
    }

    std::cout << "Initial points: " << std::endl << testPoints << std::endl;

    std::vector<cv::Point3f> testPointsHomogen =  Preprocessing::transformPointsToHomogen(testPoints);

    std::cout << "Homogen points: " << std::endl << testPointsHomogen << std::endl;

    for(int i = 0; i < NUM_TEST_POINTS; i++){
        assert(testPoints[i].x == testPointsHomogen[i].x && testPoints[i].y == testPointsHomogen[i].y && testPointsHomogen[i].z == 1);
    }
}

void test_getNormalizationMat(){

}