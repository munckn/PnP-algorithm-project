// C++
#include <iostream>
#include <time.h>
// OpenCV
#include <opencv2//core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
// PnP Tutorial
#include "Mesh.h"
#include "Model.h"
#include "PnPProblem.h"
#include "RobustMatcher.h"
#include "ModelRegistration.h"
#include "Utils.h"

//useful libraries
#include<math.h>

/**  GLOBAL VARIABLES  **/

using namespace cv;
using namespace std;


double params_WEBCAM[] = { 716,   // fx
                           716,  // fy
                           320,      // cx
                           240};    // cy

// Some basic colors
Scalar red(0, 0, 255);
Scalar green(0,255,0);
Scalar blue(255,0,0);
Scalar yellow(0,255,255);

//number of yml files
int nbyml = 4;
// Robust Matcher parameters
int numKeyPoints = 1000;      // number of detected keypoints
float ratioTest = 0.70f;          // ratio test
bool fast_match = true;       // fastRobustMatch() or robustMatch()
int matchesTreshold = 60;

// RANSAC parameters
int iterationsCount = 300;      // number of Ransac iterations.
float reprojectionError = 4.0;  // maximum allowed distance to consider it an inlier.
double confidence = 0.80;        // ransac successful confidence.

// Kalman Filter parameters
int minInliersKalman = 30;    // Kalman threshold updating

//vector used to get an average of distance 
vector<double> list_dist;
int lim_list = 10;
double dist_swift = 0.0;

// PnP parameters
int pnpMethod = SOLVEPNP_ITERATIVE;

/**  Functions headers  **/
void help();
void initKalmanFilter( KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt);
void updateKalmanFilter( KalmanFilter &KF, Mat &measurements,
                         Mat &translation_estimated, Mat &rotation_estimated );
void fillMeasurements( Mat &measurements,
                       const Mat &translation_measured, const Mat &rotation_measured);


/**  Main program  **/
int main(int argc, char *argv[])
{

  help();

  const String keys =
      "{help h        |      | print this message                   }"
      "{video v       |      | path to recorded video               }"
      "{model         |      | path to yml model                    }"
      "{mesh          |      | path to ply mesh                     }"
      "{keypoints k   |2000  | number of keypoints to detect        }"
      "{ratio r       |0.7   | threshold for ratio test             }"
      "{iterations it |500   | RANSAC maximum iterations count      }"
      "{error e       |2.0   | RANSAC reprojection errror           }"
      "{confidence c  |0.95  | RANSAC confidence                    }"
      "{inliers in    |30    | minimum inliers for Kalman update    }"
      "{method  pnp   |0     | PnP method: (0) ITERATIVE - (1) EPNP - (2) P3P - (3) DLS}"
      "{fast f        |true  | use of robust fast match             }"
      ;
  CommandLineParser parser(argc, argv, keys);

  if (parser.has("help"))
  {
      parser.printMessage();
      return 0;
  }
  else
  {
    numKeyPoints = !parser.has("keypoints") ? parser.get<int>("keypoints") : numKeyPoints;
    ratioTest = !parser.has("ratio") ? parser.get<float>("ratio") : ratioTest;
    fast_match = !parser.has("fast") ? parser.get<bool>("fast") : fast_match;
    iterationsCount = !parser.has("iterations") ? parser.get<int>("iterations") : iterationsCount;
    reprojectionError = !parser.has("error") ? parser.get<float>("error") : reprojectionError;
    confidence = !parser.has("confidence") ? parser.get<float>("confidence") : confidence;
    minInliersKalman = !parser.has("inliers") ? parser.get<int>("inliers") : minInliersKalman;
    pnpMethod = !parser.has("method") ? parser.get<int>("method") : pnpMethod;
  }

  PnPProblem pnp_detection(params_WEBCAM);
  PnPProblem pnp_detection_est(params_WEBCAM);
  // get list of models to detection
  vector<Model> list_models;
  Model model;               // instantiate Model object
  //load yml files to model
  for (int i = 0; i < nbyml; i++) {

	  string l_path ="Object" + IntToString(i + 1) + ".yml";
	  model.load(l_path); // load a 3D textured object model
	  list_models.push_back(model);
  }
  //variable used to get model info
	vector<Point3f> list_points3d_model;
	Mat descriptors_model;


  RobustMatcher rmatcher;                                                     // instantiate RobustMatcher

  Ptr<FeatureDetector> orb = ORB::create(numKeyPoints);

  rmatcher.setFeatureDetector(orb);                                      // set feature detector
  rmatcher.setDescriptorExtractor(orb);                                 // set descriptor extractor

  Ptr<flann::IndexParams> indexParams = makePtr<flann::LshIndexParams>(6, 12, 1); // instantiate LSH index parameters
  Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);       // instantiate flann search parameters

  // instantiate FlannBased matcher
  Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
  rmatcher.setDescriptorMatcher(matcher);                                                         // set matcher
  rmatcher.setRatio(ratioTest); // set ratio test parameter

  KalmanFilter KF;         // instantiate Kalman Filter
  int nStates = 18;            // the number of states
  int nMeasurements = 6;       // the number of measured states
  int nInputs = 0;             // the number of control actions
  double dt = 0.125;           // time between measurements (1/FPS)

  initKalmanFilter(KF, nStates, nMeasurements, nInputs, dt);    // init function
  Mat measurements(nMeasurements, 1, CV_64F); measurements.setTo(Scalar(0));
  bool good_measurement = false;

  


  
  // Create & Open Window
  namedWindow("REAL TIME DEMO", CV_WINDOW_AUTOSIZE);


  VideoCapture cap;                           // instantiate VideoCapture
  cap.open(1);                      // open a recorded video

  if(!cap.isOpened())   // check if we succeeded
  {
    cout << "Could not open the camera device" << endl;
    return -1;
  }

  // start and end times
  time_t start, end;

  // fps calculated using number of frames / seconds
  // floating point seconds elapsed since start
  double fps, sec;

  // frame counter
  int counter = 0;

  double dist;

  // start the clock
  time(&start);

  Mat frame, frame_vis;


	  while (waitKey(1)<0) // capture frame until ESC is pressed
	  {
		  
		  cap >> frame ;
		  resize(frame, frame, Size(640, 480), 0, 0, INTER_LANCZOS4);
		  frame_vis = frame.clone();    // refresh visualisation frame


		  // -- Step 1: Robust matching between model descriptors and scene descriptors

		  vector<DMatch> good_matches;       // to obtain the 3D points of the model
		  vector<KeyPoint> keypoints_scene;  // to obtain the 2D points of the scene

		  for (int i = 0; i < list_models.size(); i++) {
			  list_points3d_model = list_models.at(i).get_points3d();  // list with model 3D coordinates
			  descriptors_model = list_models.at(i).get_descriptors();            // list with descriptors of each 3D coordinate

			  if (fast_match)
			  {
				  rmatcher.fastRobustMatch(frame, good_matches, keypoints_scene, descriptors_model);
			  }
			  else
			  {
				  rmatcher.robustMatch(frame, good_matches, keypoints_scene, descriptors_model);
			  }

			  if (good_matches.size() > matchesTreshold) {
				  break;
			  }
		  }
		  // -- Step 2: Find out the 2D/3D correspondences

		  vector<Point3f> list_points3d_model_match; // container for the model 3D coordinates found in the scene
		  vector<Point2f> list_points2d_scene_match; // container for the model 2D coordinates found in the scene

		  for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index)
		  {
			  Point3f point3d_model = list_points3d_model[good_matches[match_index].trainIdx];  // 3D point from model
			  Point2f point2d_scene = keypoints_scene[good_matches[match_index].queryIdx].pt; // 2D point from the scene
			  list_points3d_model_match.push_back(point3d_model);         // add 3D point
			  list_points2d_scene_match.push_back(point2d_scene);         // add 2D point
		  }

		  // Draw outliers
		   draw2DPoints(frame_vis, list_points2d_scene_match, red);

		  vector<Point2f> list_points2d_inliers;

		  if (good_matches.size() > matchesTreshold) // None matches, then RANSAC crashes
		  {


			  // -- Step 3: Estimate the pose using RANSAC approach
			  pnp_detection.estimatePose(list_points3d_model_match, list_points2d_scene_match, pnpMethod);
			 
			  good_measurement = false;

			  // GOOD MEASUREMENT

			  if (list_points2d_scene_match.size() >= matchesTreshold)
			  {

				  // Get the measured translation
				  Mat translation_measured(3, 1, CV_64F);
				  translation_measured = pnp_detection.get_t_matrix();

				  // Get the measured rotation
				  Mat rotation_measured(3, 3, CV_64F);
				  rotation_measured = pnp_detection.get_R_matrix();

				  // fill the measurements vector
				  fillMeasurements(measurements, translation_measured, rotation_measured);
				  double px = 0;
				  double py = 0;
				  double pz = 0;

				  for (int i = 0; i < good_matches.size(); i++) {

					  px = px + list_points3d_model_match.at(i).x;
					  py = py + list_points3d_model_match.at(i).y;
					  pz = pz + list_points3d_model_match.at(i).z;
				  }
				  Mat Mw(3, 1, CV_64F);
				  Mw = (px / list_points3d_model_match.size(), py / list_points3d_model_match.size(), pz / list_points3d_model_match.size());
				  Mat Mc(3, 1, CV_64F);
				  projectToCameraCoords(rotation_measured, translation_measured, Mw, Mc);
				  dist_swift = sqrt(pow(Mc.at<double>(0),2) + pow(Mc.at<double>(1), 2) + pow(Mc.at<double>(2), 2));
				  
				  // Instantiate estimated translation and rotation
				  Mat translation_estimated(3, 1, CV_64F);
				  Mat rotation_estimated(3, 3, CV_64F);

				  // -- Step 4: Set estimated projection matrix
				  pnp_detection_est.set_P_matrix(rotation_measured, translation_measured);
			  }

		  }
		  else
			  dist_swift = 0.0;


		  
		  float l = 5;
		  vector<Point2f> pose_points2d;
		  pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(0, 0, 0)));  // axis center
		  pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(l, 0, 0)));  // axis x
		  pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(0, l, 0)));  // axis y
		  pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(0, 0, l)));  // axis z
		  draw3DCoordinateAxes(frame_vis, pose_points2d);           // draw axes

		  // FRAME RATE

		  // see how much time has elapsed
		  time(&end);

		  // calculate current FPS
		  ++counter;
		  sec = difftime(end, start);

		  fps = counter / sec;

		  drawFPS(frame_vis, fps, yellow); // frame ratio
		 

		  // -- Step 5: Draw some debugging text

		  // Draw some debug text

		  string n = IntToString((int)list_points2d_scene_match.size());
		  string n1 = to_string(dist_swift);
		  string text = "Found " + n + " matches";
		  string text2 = "distance obtenue: " + n1;
		  drawText(frame_vis, text, green);
		  drawText2(frame_vis, text2, red);
		  

		  imshow("REAL TIME DEMO", frame_vis);

	  }
  
  // Close and Destroy Window
  destroyWindow("REAL TIME DEMO");

  cout << "GOODBYE ..." << endl;

}

/**********************************************************************************************************/
void help()
{
cout
<< "--------------------------------------------------------------------------"   << endl
<< "This program shows how to detect an object given its 3D textured model. You can choose to "
<< "use a recorded video or the webcam."                                          << endl
<< "Usage:"                                                                       << endl
<< "./cpp-tutorial-pnp_detection -help"                                           << endl
<< "Keys:"                                                                        << endl
<< "'esc' - to quit."                                                             << endl
<< "--------------------------------------------------------------------------"   << endl
<< endl;
}

/**********************************************************************************************************/
void initKalmanFilter(KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt)
{

  KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter

  setIdentity(KF.processNoiseCov, Scalar::all(1e-5));       // set process noise
  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-2));   // set measurement noise
  setIdentity(KF.errorCovPost, Scalar::all(1));             // error covariance


                     /** DYNAMIC MODEL **/

  //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
  //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
  //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
  //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
  //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]

  // position
  KF.transitionMatrix.at<double>(0,3) = dt;
  KF.transitionMatrix.at<double>(1,4) = dt;
  KF.transitionMatrix.at<double>(2,5) = dt;
  KF.transitionMatrix.at<double>(3,6) = dt;
  KF.transitionMatrix.at<double>(4,7) = dt;
  KF.transitionMatrix.at<double>(5,8) = dt;
  KF.transitionMatrix.at<double>(0,6) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(1,7) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(2,8) = 0.5*pow(dt,2);

  // orientation
  KF.transitionMatrix.at<double>(9,12) = dt;
  KF.transitionMatrix.at<double>(10,13) = dt;
  KF.transitionMatrix.at<double>(11,14) = dt;
  KF.transitionMatrix.at<double>(12,15) = dt;
  KF.transitionMatrix.at<double>(13,16) = dt;
  KF.transitionMatrix.at<double>(14,17) = dt;
  KF.transitionMatrix.at<double>(9,15) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(10,16) = 0.5*pow(dt,2);
  KF.transitionMatrix.at<double>(11,17) = 0.5*pow(dt,2);


           /** MEASUREMENT MODEL **/

  //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
  //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
  //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
  //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
  //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
  //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]

  KF.measurementMatrix.at<double>(0,0) = 1;  // x
  KF.measurementMatrix.at<double>(1,1) = 1;  // y
  KF.measurementMatrix.at<double>(2,2) = 1;  // z
  KF.measurementMatrix.at<double>(3,9) = 1;  // roll
  KF.measurementMatrix.at<double>(4,10) = 1; // pitch
  KF.measurementMatrix.at<double>(5,11) = 1; // yaw

}

/**********************************************************************************************************/
void updateKalmanFilter( KalmanFilter &KF, Mat &measurement,
                         Mat &translation_estimated, Mat &rotation_estimated )
{

  // First predict, to update the internal statePre variable
  Mat prediction = KF.predict();

  // The "correct" phase that is going to use the predicted value and our measurement
  Mat estimated = KF.correct(measurement);

  // Estimated translation
  translation_estimated.at<double>(0) = estimated.at<double>(0);
  translation_estimated.at<double>(1) = estimated.at<double>(1);
  translation_estimated.at<double>(2) = estimated.at<double>(2);

  // Estimated euler angles
  Mat eulers_estimated(3, 1, CV_64F);
  eulers_estimated.at<double>(0) = estimated.at<double>(9);
  eulers_estimated.at<double>(1) = estimated.at<double>(10);
  eulers_estimated.at<double>(2) = estimated.at<double>(11);

  // Convert estimated quaternion to rotation matrix
  rotation_estimated = euler2rot(eulers_estimated);

}

/**********************************************************************************************************/
void fillMeasurements( Mat &measurements,
                       const Mat &translation_measured, const Mat &rotation_measured)
{
  // Convert rotation matrix to euler angles
  Mat measured_eulers(3, 1, CV_64F);
  measured_eulers = rot2euler(rotation_measured);

  // Set measurement to predict
  measurements.at<double>(0) = translation_measured.at<double>(0); // x
  measurements.at<double>(1) = translation_measured.at<double>(1); // y
  measurements.at<double>(2) = translation_measured.at<double>(2); // z
  measurements.at<double>(3) = measured_eulers.at<double>(0);      // roll
  measurements.at<double>(4) = measured_eulers.at<double>(1);      // pitch
  measurements.at<double>(5) = measured_eulers.at<double>(2);      // yaw
}
