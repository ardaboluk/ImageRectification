

#include <vector>
#include <random>
#include <cassert>
#include <limits>
#include "preprocessing.h"

#define NUM_TEST_CASES 10
#define NUM_TEST_POINTS 100
#define X_COORD_MIN 0
#define Y_COORD_MIN 0
#define X_COORD_MAX 700
#define Y_COORD_MAX 700

float randomFloat(float a, float b);
std::vector<cv::Point2f> getRandomPoints2f();

void test_transformPointsToHomogen();
void test_normalizeCoordinates();

int main(){

    test_transformPointsToHomogen();
    std::cout << std::endl;
    test_normalizeCoordinates();

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
        cv::Point2f tmpTestPoint;
        tmpTestPoint.x = randomFloat(X_COORD_MIN, X_COORD_MAX);
        tmpTestPoint.y = randomFloat(Y_COORD_MIN, Y_COORD_MAX);
        randPoints.push_back(tmpTestPoint);
    }
    return randPoints;
}

void test_transformPointsToHomogen(){
    std::cout << "TESTING Preprocessing::transformPointsToHomogen" << std::endl;
    std::vector<cv::Point2f> testPoints = getRandomPoints2f();
    std::vector<cv::Point3f> testPointsHomogen =  Preprocessing::transformPointsToHomogen(testPoints);
    for(int i = 0; i < NUM_TEST_POINTS; i++){
        assert(testPoints[i].x == testPointsHomogen[i].x && testPoints[i].y == testPointsHomogen[i].y && testPointsHomogen[i].z == 1);
    }
    std::cout << "TEST SUCCESSFUL" << std::endl;
}

void test_normalizeCoordinates(){
    std::cout << "TESTING Preprocessing::normalizeCoordinates" << std::endl;
    for(int i = 0; i < NUM_TEST_CASES; i++){
        std::vector<cv::Point2f> points2f = getRandomPoints2f();
        std::vector<cv::Point3f> points3f = Preprocessing::transformPointsToHomogen(points2f);
        cv::Mat normalizationMat = Preprocessing::getNormalizationMat(points3f);
        std::vector<cv::Point3f> normalizedPoints3f = Preprocessing::normalizeCoordinates(points3f, normalizationMat);
        
        cv::Point3f normCenter;
        float normAvgDistance = 0;
        for(auto it = normalizedPoints3f.begin(); it != normalizedPoints3f.end(); ++it){
            normCenter += (*it);
        }
        normCenter.x /= normalizedPoints3f.size();
        normCenter.y /= normalizedPoints3f.size();
        normCenter.z /= normalizedPoints3f.size();

        float avgDist = 0;
        for(auto it = normalizedPoints3f.begin(); it != normalizedPoints3f.end(); ++it){
            cv::Point3f diffPoint = (*it) - normCenter;
            avgDist += sqrtf(powf(diffPoint.x, 2.0f) + powf(diffPoint.y, 2.0f) + powf(diffPoint.z, 2.0f));
        }
        avgDist /= normalizedPoints3f.size();

        float epsilonf = std::numeric_limits<float>::epsilon() * 10;
        float sqrt2f = sqrtf(2.0f);
        assert((normCenter.x < epsilonf) && (normCenter.y < epsilonf) && (abs(normCenter.z - 1) < epsilonf));
        assert((avgDist - sqrt2f < epsilonf));
    }
    std::cout << "TEST SUCCESSFUL" << std::endl;
}