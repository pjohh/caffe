#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>

#if CV_VERSION_MAJOR == 3
#include <opencv2/imgcodecs/imgcodecs.hpp>
#define CV_GRAY2BGR cv::COLOR_GRAY2BGR
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#define CV_BGR2YCrCb cv::COLOR_BGR2YCrCb
#define CV_YCrCb2BGR cv::COLOR_YCrCb2BGR
#define CV_IMWRITE_JPEG_QUALITY cv::IMWRITE_JPEG_QUALITY
#define CV_LOAD_IMAGE_COLOR cv::IMREAD_COLOR
#define CV_THRESH_BINARY_INV cv::THRESH_BINARY_INV
#define CV_THRESH_OTSU cv::THRESH_OTSU
#endif

#include <algorithm>
#include <numeric>
#include <vector>

#include "caffe/util/im_transforms.hpp"
#include "caffe/util/math_functions.hpp"
#include <iomanip>

using namespace std;

int main( int argc, char** argv ) {
  //google::InitGoogleLogging(argv[0]);
  
  
  string open_path = "/home/pjoh/data/robot_dataset/nhg/Images/";
  string save_path = "/home/pjoh/data/robot_dataset_extended/nhg/Images/";
  string anno_src = "/home/pjoh/data/robot_dataset/nhg/Annotations/";
  string anno_dst = "/home/pjoh/data/robot_dataset_extended/nhg/Annotations/";
  string img_name;
  cv::Mat in_img, out_img;
  float brightness_noise = 0, saturation_noise = 0, color_noise = 0, desaturation = 0, prob = 0;
  
  for(int i = 0; i < 10; ++i){
    if(i == 0){
      brightness_noise = 0;
      color_noise = 0;
      saturation_noise = 0;
      desaturation = 0;
      prob = 0;
    }
    else {
      brightness_noise = 0.2;
      color_noise = 0.02;
      saturation_noise = 0.07;
      prob = 0.4;
    }
    
    ifstream img_file("/home/pjoh/data/robot_dataset/nhg/Images/train.txt");
    
    while(img_file >> img_name){
      in_img = cv::imread(open_path + img_name);
      
      if (brightness_noise > 0 || color_noise > 0 || saturation_noise > 0) {
	float prob_brightness, prob_color, prob_saturation;
	caffe::caffe_rng_uniform(1, static_cast<float>(0), static_cast<float>(1), &prob_brightness);
	caffe::caffe_rng_uniform(1, static_cast<float>(0), static_cast<float>(1), &prob_color);
	caffe::caffe_rng_uniform(1, static_cast<float>(0), static_cast<float>(1), &prob_saturation);
	
	CHECK(brightness_noise < 1) << "brightness noise needs to be < 1 !!!";
	float b_noise = 0;
	if (prob_brightness <= prob) {
	  caffe::caffe_rng_uniform(1, static_cast<float>(0), brightness_noise, &b_noise);
	  if (int(b_noise*1000)%2 == 1) b_noise *= -1; 
	}
	
	CHECK(saturation_noise < 1) << "saturation noise needs to be < 1 !!!";
	float s_noise = 0;
	if (prob_saturation <= prob) {
	  caffe::caffe_rng_uniform(1, static_cast<float>(0), saturation_noise, &s_noise);
	  if (int(s_noise*1000)%2 == 1) s_noise *= -1; 
	}
	
	CHECK(color_noise < 1) << "color noise needs to be < 1 !!!";
	float c_noise = 0;
	if (prob_color <= prob) {
	  caffe::caffe_rng_uniform(1, static_cast<float>(0), color_noise, &c_noise);
	  if (int(c_noise*1000)%2 == 1) c_noise *= -1;
	}
	//caffe_rng_uniform(1, 1-color_noise, 1+color_noise, &color_noise);
	
	cout << "brightness noise: " << fixed << setprecision(3) << setw(6) << setfill(' ') << b_noise << " | color noise: " << fixed << setprecision(3) << setw(6) << setfill(' ') << c_noise << " | saturation noise: " << fixed << setprecision(3) << setw(6) << setfill(' ') << s_noise << endl;
	//out_img = cv::Mat::zeros(in_img.size(), in_img.type());
	cv::Mat hsv_img;
	cv::cvtColor(in_img, hsv_img, CV_BGR2HSV);
	
	for(int y = 0; y < hsv_img.rows; y++ ) { 
	  for( int x = 0; x < hsv_img.cols; x++ ) { 
	    for( int c = 0; c < 3; c++ ) { 
	      if (c == 0) hsv_img.at<cv::Vec3b>(y,x)[c] = cv::saturate_cast<uchar>(hsv_img.at<cv::Vec3b>(y,x)[c] + c_noise*100);
	      if (c == 1) hsv_img.at<cv::Vec3b>(y,x)[c] = cv::saturate_cast<uchar>(hsv_img.at<cv::Vec3b>(y,x)[c] + s_noise*100);
	      if (c == 2) hsv_img.at<cv::Vec3b>(y,x)[c] = cv::saturate_cast<uchar>(hsv_img.at<cv::Vec3b>(y,x)[c] + b_noise*100);
	      //if (int(c*1000*color_noise)%2 == 0) out_img.at<cv::Vec3b>(y,x)[c] = cv::saturate_cast<uchar>(color_noise*(in_img.at<cv::Vec3b>(y,x)[c] + brightness_noise*100));
	      //else out_img.at<cv::Vec3b>(y,x)[c] = cv::saturate_cast<uchar>(1*(in_img.at<cv::Vec3b>(y,x)[c] + brightness_noise*100));
	    }
	  }
	}
	//stringstream ss;
	//ss << "New: b: " << (int)(brightness_noise*100) << " c: " << (int)(color_noise*100) << " s: " << (int)(saturation_noise*100);
	
	cv::cvtColor(hsv_img, out_img, CV_HSV2BGR);
	//cv::namedWindow("Original Image", 1);
	//cv::namedWindow(ss.str(), 1);
	//cv::imshow("Original Image", in_img);
	//cv::imshow(ss.str(), out_img);
	//cv::waitKey(0);
      }
      else out_img = in_img;
      
      if (desaturation > 0 ) {
	cv::Mat hsv_img;
	cv::cvtColor(in_img, hsv_img, CV_BGR2HSV);
	
	for(int y = 0; y < hsv_img.rows; y++ ) { 
	  for( int x = 0; x < hsv_img.cols; x++ ) {
	    for( int c = 0; c < 3; c++ ) if (c == 1) hsv_img.at<cv::Vec3b>(y,x)[c] = cv::saturate_cast<uchar>(hsv_img.at<cv::Vec3b>(y,x)[c] - desaturation*100);
	  }
	}
	cv::cvtColor(hsv_img, out_img, CV_HSV2BGR);
	//cv::imshow("Original Image", in_img);
	//cv::imshow("New Image", out_img);
	//cv::waitKey(0);
      }
      
      // split <file_name> from .jpg
      string delimiter = ".";
      string name = img_name.substr(0, img_name.find(delimiter));
      
      ostringstream full_name;
      full_name << name << "_da_" << i;
      
      cv::imwrite(save_path + full_name.str() + ".jpg", out_img);
      
      // copy anno files
      string anno_load = anno_src + name + ".xml";
      string anno_save = anno_dst + full_name.str() + ".xml";
      ifstream  anno_src_(anno_load.c_str(), ios::binary);
      ofstream  anno_dst_(anno_save.c_str(), ios::binary);
      anno_dst_ << anno_src_.rdbuf();
    }
  }
}

#endif  // USE_OPENCV