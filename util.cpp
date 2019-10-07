
#include <iostream>
#include "util.h"

void Util::displayMat(cv::Mat& cvMatrix, std::string explanation) {
	std::cout << explanation << std::endl;
	std::cout << cvMatrix << std::endl;
	std::cout << std::endl;
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> Util::extractMatches(std::string image1Filename, std::string image2Filename){
	cv::Mat image1 = cv::imread(image1Filename);
	cv::Mat image2 = cv::imread(image2Filename);

	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ("BruteForce-Hamming");

    detector->detect ( image1,keypoints_1 );
    detector->detect ( image2,keypoints_2 );

    descriptor->compute ( image1, keypoints_1, descriptors_1 );
    descriptor->compute ( image2, keypoints_2, descriptors_2 );

    cv::Mat outimg1;
    drawKeypoints( image1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
    imshow("ORB Keypoints",outimg1);

    std::vector<cv::DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match (descriptors_1, descriptors_2, matches);

    double min_dist=10000, max_dist=0;

    for ( int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist);
    printf ( "-- Min dist : %f \n", min_dist);

    std::vector<cv::DMatch> good_matches;
    for ( int i = 0; (unsigned int)i < descriptors_1.rows; i++)
    {
        if ( matches[i].distance <= cv::max(2*min_dist, 30.0))
        {
            good_matches.push_back (matches[i]);
        }
    }

    cv::Mat img_match;
    cv::Mat img_goodmatch;
    cv::drawMatches (image1, keypoints_1, image2, keypoints_2, matches, img_match);
    cv::drawMatches (image1, keypoints_1, image2, keypoints_2, good_matches, img_goodmatch);
    cv::imshow ("Matches", img_match);
    cv::imshow ("Good Matches", img_goodmatch);
    cv::waitKey(0);

	std::vector<cv::Point2f> correspondingPoints1;
	std::vector<cv::Point2f> correspondingPoints2;
	for(int i = 0; i < good_matches.size(); i++){
		int idx1 = good_matches[i].queryIdx;
		int idx2 = good_matches[i].trainIdx;
		correspondingPoints1.push_back(keypoints_1[idx1].pt);
		correspondingPoints2.push_back(keypoints_2[idx2].pt);
	}

	std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> correspondingPointsList;
	correspondingPointsList.first = correspondingPoints1;
	correspondingPointsList.second = correspondingPoints2;

	return correspondingPointsList;
}