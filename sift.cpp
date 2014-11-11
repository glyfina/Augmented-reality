#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <cstdio>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <vector>
 #include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
using namespace cv;
using namespace std;


int main(int argc, const char* argv[])
{

   
    cv::Mat input1 = cv::imread("image.jpg",1); //Load as grayscale
    //cv::cvtColor(input0,input1,CV_BGR2GRAY);

    //second input load as grayscale
   cv::Mat input2 = cv::imread("image1.jpg",1);


    cv::SiftFeatureDetector detector;
    //cv::SiftFeatureDetector detector(1,1,cv::SIFT::CommonParams::DEFAULT_NOCTAVES,cv::SIFT::CommonParams::DEFAULT_NOCTAVE_LAYERS,cv::SIFT::CommonParams::DEFAULT_FIRST_OCTAVE,cv::SIFT::CommonParams::FIRST_ANGLE);
    std::vector<cv::KeyPoint> keypoints1;
    detector.detect(input1, keypoints1); // keypoints of input1  are stored 

    // Add results to image and save.
    cv::Mat output1;
    cv::drawKeypoints(input1, keypoints1, output1);
    cv::imshow("Sift_result1.jpg", output1);
    
    //keypoints array for input 2
    std::vector<cv::KeyPoint> keypoints2;

    //output array for ouput 2
    cv::Mat output2;

    //Sift extractor of opencv

    cv::SiftDescriptorExtractor extractor;

    cv::Mat descriptors1,descriptors2;

   
   cv::BFMatcher matcher(cv::NORM_L2);

    cv::vector<cv::DMatch> matches;
    cv::Mat img_matches;
        detector.detect(input2,keypoints2);

        cv::drawKeypoints(input2,keypoints2,output2);

        cv::imshow("Sift_result2.jpg",output2);
        

        extractor.compute(input1,keypoints1,descriptors1);
        extractor.compute(input2,keypoints2,descriptors2);


        matcher.match(descriptors1,descriptors2,matches);


        //show result
        cv::drawMatches(input1,keypoints1,input2,keypoints2,matches,img_matches);
        cv::imshow("matches",img_matches);
        

     cv::waitKey();

    return 0;
}
