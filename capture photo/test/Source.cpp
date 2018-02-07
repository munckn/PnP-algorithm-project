#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<sstream>
using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
	VideoCapture cap;
	cap.open(1);
	Mat img;
	int test = 5, i = 0;
	while ( i < test)
	{
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
		String titre = "Image"+to_string(i+1)+".jpg";
		imwrite(titre, img);
		i++;
	}
	return 0;
}

