/*
Create Date: 11052017
Content: Moving from Matlab to C++ for adaptive structure from motion
Created by: Yazhe
*/

#define CERES_FOUND true
#include <iostream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/sfm.hpp>

using namespace std;
using namespace cv; 

struct compclass{
	bool operator() (DMatch matches1, DMatch matches2) { return (matches1.distance < matches2.distance);}
}mycomp;

int main()
{
	//Time start
	auto start = std::chrono::system_clock::now();

	//load cameraMatrix, distCoeffs
	FileStorage fs("out_camera.xml", FileStorage::READ);
	Mat cameraMatrix, distCoeffs;
	fs["camera_matrix"] >> cameraMatrix;
	fs["distortion_coefficients"] >> distCoeffs;

	//read images from source
	Mat im1 = imread("f1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat im2 = imread("f2.jpg", CV_LOAD_IMAGE_COLOR);

	//undistort images
	Mat temp1, im1_undistort, temp2, im2_undistort;
	temp1 = im1;
	undistort(temp1, im1_undistort, cameraMatrix, distCoeffs);
	temp2 = im2;
	undistort(temp2, im2_undistort, cameraMatrix, distCoeffs);

	//SIFT descriptor
	Ptr<xfeatures2d::SIFT> sift_im1 = xfeatures2d::SIFT::create();
	Ptr<xfeatures2d::SIFT> sift_im2 = xfeatures2d::SIFT::create();	
	Mat mask;
	mask = Mat::ones(im1.rows, im1.cols, CV_8U);
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2; 
	sift_im1 -> detectAndCompute(im1_undistort, mask, keypoints1, descriptors1);
	sift_im2 -> detectAndCompute(im2_undistort, mask, keypoints2, descriptors2);

	//FlannBasedMatcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors1, descriptors2, matches );

	//-- Quick calculation of max and min distances between keypoints
	double max_dist = 0; double min_dist = 100;
	for( int i = 0; i < descriptors1.rows; i++ )
	{ 
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );

	//-- Sort matches
	sort(matches.begin(), matches.end(), mycomp);

	//------display sorted matches
	// for (int i = 0; i < 20; i++)
	// {
	// 	cout << "distance " << matches[i].distance << ' ';
	// 	cout << "imgIdx " << matches[i].imgIdx << ' ';
	// 	cout << "queryIdx " << matches[i].queryIdx << ' ';
	// 	cout << "trainIdx " << matches[i].trainIdx << ' ';
	// 	cout << "keypoints1.pt " << keypoints1[matches[i].queryIdx].pt << endl;
	// }

	//-- The best n matches
	int n = 20;
	vector<DMatch> good_matches(matches.begin(), matches.begin()+n);
	vector<KeyPoint> keypoints1_sort, keypoints2_sort;
	for (int i = 0; i < n; i++){
		keypoints1_sort.push_back(keypoints1[good_matches[i].queryIdx]);
		keypoints2_sort.push_back(keypoints2[good_matches[i].trainIdx]);
	}

	Mat_<float> frame1(2, n);
	Mat_<float> frame2(2, n);
	for (int i = 0; i < n; i++)
	{
		// cout << "keypoints1_sort " << keypoints1_sort[i].pt<< ' ';
		// cout << "keypoints2_sort " << keypoints2_sort[i].pt<< endl;
		frame1(0,i) = keypoints1_sort[i].pt.x;
		frame1(1,i) = keypoints1_sort[i].pt.y;
		frame2(0,i) = keypoints2_sort[i].pt.x;
		frame2(1,i) = keypoints2_sort[i].pt.y;
	}
	cout << "frame1 " << frame1<< endl << endl;
	cout << "frame2 " << frame2<< endl << endl;
	//-- Draw matches
	Mat img_matches;
    drawMatches( im1_undistort, keypoints1, im2_undistort, keypoints2, good_matches, img_matches, 
    			Scalar::all(-1), Scalar::all(-1),
               	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    namedWindow("Good Matches", WINDOW_NORMAL);
    imshow( "Good Matches", img_matches );

    //Reconstructing 3D points
    vector<Mat> points2d_match, Rs, ts, points3d_match;

	points2d_match.push_back(Mat(frame1));
	points2d_match.push_back(Mat(frame2));
	// cout << "points2d_match " << points2d_match[1] << endl;

	Mat F, E, F_norm;
	sfm::normalizedEightPointSolver(frame1, frame2, F);
	cout << "F " << F << endl;
	sfm::essentialFromFundamental(F, cameraMatrix, cameraMatrix, E);
	cout << "E " << E << endl;
	sfm::normalizeFundamental(F, F_norm);
	cout << "F_norm " << F_norm << endl;




	// cout << "size " << points2d_match.size() << endl;
 //    sfm::reconstruct(points2d_match, Rs, ts, cameraMatrix, points3d_match, true);
 //    cout << "points3d " << points3d_match.size() << endl;
 //    cout << "size " << points2d_match.size() << endl;
 //    cout << "Rs " << Rs.size() << endl;
    cout << "cameraMatrix " << cameraMatrix << endl;

    // cout << "3D points:" << points3d.size() << endl;





    //Time end
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	cout << "It took " <<elapsed.count() << " msec" << '\n';
	waitKey(0);
	return 0;
}

