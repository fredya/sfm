/*
Create Date: 11052017
Content: Moving from Matlab to C++ for adaptive structure from motion
Created by: Yazhe
*/

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv; 

int main()
{
	//load cameraMatrix, distCoeffs
	FileStorage fs("out_camera.xml", FileStorage::READ);
	fs["camera_matrix"] >> cameraMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	cout << "camera_matrix" << cameraMatrix;

	//read images from source
	Mat im1 = imread("f1.jpg", CV_LOAD_IMAGE_COLOR);
	imshow("a", im1);
	Mat im2 = imread("f2.jpg", CV_LOAD_IMAGE_COLOR);
	imshow("b", im2);

	//undistort images
	Mat temp, view;
	temp = im1;
	undistort(temp, view, cameraMatrix, distCoeffs);


	waitKey(0);
	return 0;
}

