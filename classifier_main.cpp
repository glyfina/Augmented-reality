#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>
#include <iostream>
#include <cstdio>
#include<stdio.h>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <vector>
#include "GL/glut.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <ctime>
#include <queue>
#include <vector>
#include <stdlib.h>
#include <sys/stat.h>
#include <math.h>
#include "opencv2/ml/ml.hpp"
#include <limits>

using namespace std;
using namespace cv;
using std::vector;
using std::iostream;

int main(){
cv::DescriptorExtractor *descriptor = new cv::SiftDescriptorExtractor();
cv::FeatureDetector *detector = new cv::SiftFeatureDetector;
cv::DescriptorMatcher *matcher = new cv::FlannBasedMatcher;

//defining terms for bowkmeans trainer
    TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
   
    int dictionarySize = 50;
   
    int retries = 1;
    int flags = KMEANS_PP_CENTERS;
    BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);

    //BOWImgDescriptorExtractor bowDE(descriptor, matcher);
    Ptr<BOWImgDescriptorExtractor> bowide(new BOWImgDescriptorExtractor(descriptor, matcher));
    Mat dictionary;

////////////////////////////////////////////

Mat labels_kmeans;
int x;
char y;

FILE * inp = fopen ("labels_kmeans.txt" , "r");
for(int i=0;i<31;i++){
fscanf ( inp , "%d %c" , &x , &y );
labels_kmeans.push_back(x);

}

///////////////////////////////////////////////////
  string n;
  char filename[255];

     for(int i=1;i<32;i++){
      n=sprintf(filename, "./dictionary/dict_%d.yaml",i); 
      FileStorage fs(filename,FileStorage::READ);
     
     if(fs.isOpened()==true){
      fs["dictionary"]>>dictionary;
      
      
      fs.release();
     }
    bowide->setVocabulary(dictionary);
    }


/////////////////////////////////////////////////////
NormalBayesClassifier classifier;
NormalBayesClassifier classifier_new;
classifier.load("classifier_50.xml");
//////////////////////////////////////////////////////////
  Mat tryme;
  Mat tryDescriptor;
  Mat img3 = imread("IMG_13.jpg");
  vector<KeyPoint> keypoints3;
  detector->detect(img3, keypoints3);


  bowide->compute(img3, keypoints3, tryDescriptor);
  tryme.push_back(tryDescriptor);
  
  cout<<"ok"<<endl;
  int f=classifier.predict(tryme);
  cout<<"prediction1="<<f<<endl;

///////////////////////////////////////////////////////  

 n=sprintf(filename, "./classifier/classifier_%d.xml",f); 
classifier.load(filename);
int s=classifier.predict(tryme);
cout<<s<<endl;


///////////////////////////////////////////////////////////

Mat show1;
Mat show2;

n=sprintf(filename, "IMG_%d.jpg",s); 
Mat img4=imread(filename);

Size s1( img3.size().width/2, img3.size().height/2 );
resize( img3,show1 , s1, 0, 0, CV_INTER_AREA );

Size s2( img4.size().width /2, img4.size().height /2 );
resize( img4,show2, s2, 0, 0, CV_INTER_AREA );

imshow("query image",show1);
imshow("predicted image",show2);

waitKey(0);
///////////////////////////////////////////////////////////
return 0;
}
