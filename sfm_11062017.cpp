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
	cout << endl << "matches size " << matches.size() << endl;

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
	int n = 6000;
	vector<DMatch> good_matches(matches.begin(), matches.begin()+n);
	vector<KeyPoint> keypoints1_sort, keypoints2_sort;
	for (int i = 0; i < n; i++){
		keypoints1_sort.push_back(keypoints1[good_matches[i].queryIdx]);
		keypoints2_sort.push_back(keypoints2[good_matches[i].trainIdx]);
	}

	vector<Point2f> frame1(n);
	vector<Point2f> frame2(n);
	Mat_<float> frame1_Mat(2, n);
  	Mat_<float> frame2_Mat(2, n);	
  	for (int i = 0; i < n; i++)
	{
		// cout << "keypoints1_sort " << keypoints1_sort[i].pt<< ' ';
		// cout << "keypoints2_sort " << keypoints2_sort[i].pt<< endl;
		frame1[i] = Point2f((float)keypoints1_sort[i].pt.x, (float)keypoints1_sort[i].pt.y);
		frame2[i] = Point2f((float)keypoints2_sort[i].pt.x, (float)keypoints2_sort[i].pt.y);
		frame1_Mat(0,i) = keypoints1_sort[i].pt.x;
		frame1_Mat(1,i) = keypoints1_sort[i].pt.y;
		frame2_Mat(0,i) = keypoints2_sort[i].pt.x;
		frame2_Mat(1,i) = keypoints2_sort[i].pt.y;
	}
	cout << "frame1 " << frame1[0] << endl << endl;
	cout << "frame2 " << frame2[0] << endl << endl;
	//-- Draw matches
	Mat img_matches;
    drawMatches( im1_undistort, keypoints1, im2_undistort, keypoints2, good_matches, img_matches, 
    			Scalar::all(-1), Scalar::all(-1),
               	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    namedWindow("Good Matches", WINDOW_NORMAL);
    imshow( "Good Matches", img_matches );

    //Reconstructing 3D points
    Mat K = cameraMatrix;
    Mat F = findFundamentalMat(frame1, frame2, FM_RANSAC, 0.1, 0.99);
	// Mat E = K.t() * F * K; 







    Mat E;
	sfm::essentialFromFundamental(F, K, K, E);
	cout << "F" << F << endl;
	cout << "E" << E << endl;
	vector<Mat> Rs, ts;
	sfm::motionFromEssential(E, Rs, ts);
	Mat P2;	
	sfm::projectionFromKRt(K, Rs[0], ts[0], P2);
	// SVD svd(E);
	// Matx33d W(0,-1,0,   //HZ 9.13
	//       1,0,0,
	//       0,0,1);
	// Matx33d Winv(0,1,0,
	//      -1,0,0,
	//      0,0,1);
	// Mat R = svd.u * Mat(W) * svd.vt; //HZ 9.19
	// Mat t = svd.u.col(2); //u3
	Matx34d P1= Matx34d(K.at<double>(0,0), K.at<double>(0,1), K.at<double>(0,2), 0.0,
			 K.at<double>(1,0), K.at<double>(1,1), K.at<double>(1,2), 0.0,
			 K.at<double>(2,0), K.at<double>(2,1), K.at<double>(2,2), 0.0);

	// Matx34d P2 = Matx34d(R.at<double>(0,0),    R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
	//          R.at<double>(1,0),    R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
	//          R.at<double>(2,0),    R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2));







	cout << "Rs" << Rs[0] << endl;
	cout << "ts" << ts[0] << endl;
	cout << "P1" << P1 << endl;
	cout << "P2" << P2 << endl;

	// ----Triangulate points
	Mat points4D;
	triangulatePoints(P1, P2, frame2_Mat, frame1_Mat, points4D);
	cout << endl << "points4D " << points4D << endl;
	cout << endl << "points4D_30 " << points4D.at<float>(0,0) << endl;



	Mat pt_3d; convertPointsFromHomogeneous(Mat(points4D.t()).reshape(4, 1),pt_3d);
	cout << endl << "pt_3d" << pt_3d << endl;


	// //----Convert points4D to 3D
	// Mat_<float> points4D_to3D(3, n);
	// for (int i = 0; i < n; i++){
	// 	points4D_to3D.at<float>(0,i) = points4D.at<float>(0,i)/points4D.at<float>(3,i);
	// 	points4D_to3D.at<float>(1,i) = points4D.at<float>(1,i)/points4D.at<float>(3,i);
	// 	points4D_to3D.at<float>(2,i) = points4D.at<float>(2,i)/points4D.at<float>(3,i);
	// }
	// cout << endl << "points4D_to3D" << points4D_to3D << endl;

	//----Show 3d points
	viz::Viz3d window("Coordinate Frame");
	window.setWindowSize(Size(500,500));
	window.setWindowPosition(Point(150,150));
	window.setBackgroundColor(); // black by default

	vector<Vec3f> point_cloud;
	Mat_<float> temp_pt(3,1);
	for (int i = 0; i < n; i++){
		temp_pt.at<float>(0,0) = pt_3d.at<float>(i,0);
		temp_pt.at<float>(1,0) = pt_3d.at<float>(i,1);
		temp_pt.at<float>(2,0) = pt_3d.at<float>(i,2);
		point_cloud.push_back(Vec3f(temp_pt));
	}

	for (int i = 0; i < n; i++){
		cout << endl << "point_cloud" << point_cloud[i] << endl;
	}
	cout << "Rendering points   ... ";
    viz::WCloud cloud_widget(point_cloud, viz::Color::green());
    window.showWidget("point_cloud", cloud_widget);
    cout << "[DONE]" << endl;

    window.spin();

    //Time end
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	cout << "It took " <<elapsed.count() << " msec" << '\n';
	waitKey(0);
	return 0;
}

