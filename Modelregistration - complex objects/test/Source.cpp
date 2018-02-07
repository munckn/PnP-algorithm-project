// C++
#include <iostream>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

//useful include
#include "Model.h"
#include "Utils.h"

using namespace cv;
using namespace std;


/**  GLOBAL VARIABLES  **/
													   

// Intrinsic camera parameters: UVC WEBCAM
double params_CANON[] = { 716,   // fx
716,  // fy
320,      // cx
240 };    // cy

//number of pictures to take
int nbpict = 4;

//boolean to check if we take new pictures
bool picCheck;

string write_path = "ymlFiles/";
string read_path = "Pictures/";

// Some basic colors
Scalar red(0, 0, 255);
Scalar green(0, 255, 0);
Scalar blue(255, 0, 0);
Scalar yellow(0, 255, 255);



//function to ask if the user need pictures of an object
bool needNewPicture() {

	printf("do you need pictures for sfm?(y/n)");
	char r;
	scanf_s("%c", &r);
	if ((r == 'y') || (r == 'Y'))
		return true;
	else
		return false;
}

//function to take pictures and store them in a file
void TakePicures(int nb) {

	VideoCapture cap;
	cap.open(1);
	Mat img;
	int i = 0;
	while (i < nb)
	{
		if (i == 0)
			printf("take a picture from the origin point\n");
		else 
			cout << "picture " + IntToString(i + 1) + " of " + IntToString(nb) + " \n";
		while (waitKey(1) < 0)
		{
			cap >> img;
			resize(img, img, Size(640, 480), 0, 0, INTER_LANCZOS4);
			if (img.empty())
			{
				printf("pas d'images trouvées \n");
			}
			
			
			namedWindow("caméra", CV_WINDOW_AUTOSIZE);
			imshow("caméra", img);
		}
		String titre = "pictures/image" + to_string(i + 1) + ".jpg";
		imwrite(titre, img);
		i++;
	}

}

bool CheckCoherentRotation(const cv::Mat_<double>& R) {

	if (fabsf(determinant(R)) - 1.0 > 1e-07) {

		cerr << "rotation matrix is invalid" << endl;

		return false;

	}
	return true;
}

/**  Main program  **/
int main(int argc, char *argv[])
{
	//create the model we will obtain by sfm
	Model model1, model2; 
	//variables used for storing 2D points
	vector<Point2f> list_points2dOrigin, list_points2dSecond, OriginPts,SecondPts;
	
	Mat imOrigin,imSecond,imOriginKey, imSecondKey;
	//get the focal point
	Point2f C = (params_CANON[2], params_CANON[3]);
	
	//get the focal length
	double f = params_CANON[1];

	picCheck = needNewPicture();

	if (picCheck)
	{
		TakePicures(nbpict);
	}
	
	//read image of origin point of view
	string path = read_path + "image1.jpg";
	imOrigin = imread(path, 1);

	//initialize the number of wanted keypoints in an image
	int numKeyPoints = 10000;
	//variables useful for keypoints computation
	vector<KeyPoint> keypts1, keypts2;

	Mat desc1, desc2, desccorrct;

	//creation of ORB classes
	Ptr<Feature2D> orb = ORB::create(numKeyPoints);

	orb->detectAndCompute(imOrigin, noArray(), keypts1, desc1);

	for (int j = 0; j < nbpict-1; j++)
	{
		string sec_path = read_path + "image"+ IntToString(j+2) +".jpg";
		imSecond = imread(sec_path, 1);

		orb->detectAndCompute(imSecond, noArray(), keypts2, desc2);

		if (j == 0) {

			for (int i = 0; i < keypts1.size() - 1; i++)
			{
				KeyPoint Key = keypts1.at(i);
				Point2f X = Key.pt;
				list_points2dOrigin.push_back(X);
			}
			
				namedWindow("image1 without keypoint", CV_WINDOW_AUTOSIZE);
				namedWindow("image1 with keypoint", CV_WINDOW_AUTOSIZE);

				imshow("image1 without keypoint", imOrigin);
				waitKey(0);
				destroyWindow("image1 without keypoint");

				draw2DPoints(imOrigin, list_points2dOrigin, green);
				imshow("image1 with keypoint", imOrigin);
				waitKey(0);
			
			for (int i = 0; i < keypts2.size() - 1; i++)
			{
				KeyPoint Key = keypts2.at(i);
				Point2f X = Key.pt;
				list_points2dSecond.push_back(X);
			}
			
				namedWindow("image2 without keypoint", CV_WINDOW_AUTOSIZE);
				namedWindow("image2 with keypoint", CV_WINDOW_AUTOSIZE);

				//display images without and with keypoints 
				imshow("image2 without keypoint", imSecond);
				waitKey(0);
				destroyWindow("image2 without keypoint");

				draw2DPoints(imSecond, list_points2dSecond, green);
				imshow("image2 with keypoint", imSecond);
				waitKey(0);
			
		}
		//intansiate the matching descriptor and get matches between two images

		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

		vector<DMatch> matches;
		if(desc1.rows<=desc2.rows)
			matcher->match(desc1, desc2, matches);
		else
			matcher->match(desc2, desc1, matches);

		for (size_t j = 0; j < matches.size(); j++) {

			if (desc1.rows <= desc2.rows) {

				

				SecondPts.push_back(keypts2[matches[j].trainIdx].pt);

				

				OriginPts.push_back(keypts1[matches[j].queryIdx].pt);
			}
			else {

				SecondPts.push_back(keypts2[matches[j].queryIdx].pt);

				

				OriginPts.push_back(keypts1[matches[j].trainIdx].pt);

			}
		}
		//display images with common keypoints
		if (j == 0) {
			draw2DPoints(imSecond, SecondPts, blue);

			draw2DPoints(imOrigin, OriginPts, blue);
			namedWindow("image1 with matches", CV_WINDOW_AUTOSIZE);
			imshow("image1 with matches", imOrigin);
			waitKey(0);

			namedWindow("image2 with matches", CV_WINDOW_AUTOSIZE);
			imshow("image2 with matches", imSecond);
			waitKey(0);
			destroyAllWindows;
		}
		
		// essential matrix to recover the pose
		Mat E, status;
		cout << "finding Essential Matrix...\n";
		E = findEssentialMat(OriginPts, SecondPts, f, C, RANSAC, 0.999, 1.0, status);

		//matrices for camera extrinsic parameters
		Mat R, t;
		cout << "recovering pose...\n";
		int check = recoverPose(E, OriginPts, SecondPts, R, t, f, C, noArray());
		cout << "\nt.x : " << t.at<double>(0) << "\nt.y : " << t.at<double>(1) << "\nt.z : " << t.at<double>(2);

		//checking the rotation matrice
		bool testR = CheckCoherentRotation(R);


		//get projections matrices 
		Mat Pright = Mat::eye(3, 4, CV_32F);

		Mat Pleft;
		hconcat(R, t, Pleft);

		//list of 3D points for triangulation
		Mat list_4Dpoints;
		vector<Point3d>list_3Dpoints;

		//triangulation from rwo views of the object
		//for the left image
		cout << "triangulation processing...\n";
		triangulatePoints(Pright, Pleft, OriginPts, SecondPts, list_4Dpoints);
		Size size_list = list_4Dpoints.size();
		// turn 4D points into 3D points
		for ( int i = 0; i < list_4Dpoints.size().width; ++i) {

			Point3d point3d;
			point3d.x = list_4Dpoints.at<float>(0, i) / list_4Dpoints.at<float>(3, i);
			point3d.y = list_4Dpoints.at<float>(1, i) / list_4Dpoints.at<float>(3, i);
			point3d.z = list_4Dpoints.at<float>(2, i) / list_4Dpoints.at<float>(3, i);
			list_3Dpoints.push_back(point3d);

		}
		//adding descriptors, 2D/3D points correspondences and keypoints in models

		for (  int i = 0; i < min(desc1.size().height, desc2.size().height); ++i) {
			Point2d point2dRight(OriginPts[i]);
			Point2d point2dLeft(SecondPts[i]);
			Point3d point3d;
			point3d = list_3Dpoints.at(i);
			if (j == 0) {
				model1.add_correspondence(point2dRight, point3d);
				model1.add_descriptor(desc1.row(i));
				model1.add_keypoint(keypts1[i]);
			}
			if (testR) {
				model2.add_correspondence(point2dLeft, point3d);
				model2.add_descriptor(desc2.row(i));
				model2.add_keypoint(keypts2[i]);
			}
		}
		// save the model into a *.yaml file
		if (j == 0) {
			string W_path = write_path+"Object" + IntToString(j + 1) + ".yml";
			model1.save(W_path);
		}
		string w_secPath = write_path+"Object" + IntToString(j + 2) + ".yml";
		model2.save(w_secPath);
		cout << "step " + IntToString(j + 1) + "done\n";
	}
	return 0;
}


