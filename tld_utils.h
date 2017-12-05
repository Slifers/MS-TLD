#ifndef TLD_UNTILS_H
#define TLD_UNTILS_H


#include <opencv2/opencv.hpp>
#pragma once



void drawBox(cv::Mat& image, CvRect box, cv::Scalar color, int thick=1); 

void drawPoints(cv::Mat& image, std::vector<cv::Point2f> points,cv::Scalar color=cv::Scalar::all(255));

cv::Mat createMask(const cv::Mat& image, CvRect box);

float median(std::vector<float> v);

std::vector<int> index_shuffle(int begin,int end);

#endif // !TLD_UNTILS_H