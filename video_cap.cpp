#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <thread>

using namespace cv;
using namespace std;
using std::thread;

const int CAMERA_NUM = 4;

string command;

void keboardListener()
{
	while(1)
    {
        cin>>command;
        cout<<"command:"<<command<<endl;
    }
}

int main(int argc, char* argv[])
{
	VideoCapture caps[CAMERA_NUM];
	if(strcmp(argv[1], "a") == 0)
	{
		printf("open all\n");
		
		for(int i=0;i<CAMERA_NUM;i++)
		{
			caps[i].open(i);
			if(!caps[i].isOpened())
			{
				printf("camera %d open failed\n",i);
				return 0;
			}
			// caps[i].set(cv::CAP_PROP_FRAME_WIDTH,1280);
			// caps[i].set(cv::CAP_PROP_FRAME_HEIGHT,720);
		}
	}
	else
	{
		printf("open camera :%d\n", atoi(argv[1]));
		caps[0].open(atoi(argv[1]));
		caps[0].set(cv::CAP_PROP_FRAME_WIDTH,960);
		caps[0].set(cv::CAP_PROP_FRAME_HEIGHT,640);
	}

	//gst_testsink.open("appsrc ! queue ! videoconvert ! video/x-raw, format=RGBA ! nvvidconv ! nvoverlaysink ", cv::CAP_GSTREAMER, 0, 3, cv::Size(1280, 720));
	cv::Mat img[CAMERA_NUM];

	double fps=0;
	
	//cap.set(cv::CAP_PROP_FPS,30);

	thread keyboardListenerTh(keboardListener);
	int cnt = 0;
	int screenshotCnt = 1;
	while(true)
	{
		if(argv[1] == "a")
		{
			for(int i=0;i<CAMERA_NUM;i++)
			{
				if (!caps[i].read(img[i])) 
				{
       				std::cout<<"Capture read error:"<<i<<std::endl;
       				break;
				}
   			}

			if(command == "p")
			{
				imwrite("front.png", img[0]);
				imwrite("left.png", img[1]);
				imwrite("back.png", img[2]);
				imwrite("right.png", img[3]);
				command = "";
			}
		}
		else
		{
			if (!caps[0].read(img[0])) 
			{
				std::cout<<"Capture read error"<<std::endl;
				break;
   			}
			imshow("CamShow",img[0]);
			printf("img width:%d, img height:%d\n", img[0].cols, img[0].rows);
			if(command == "p")
			{
				string filename;
				std::stringstream StrStm;
				StrStm<<screenshotCnt++;
				StrStm>>filename;
				imwrite("img"+filename+".png",img[0]);
				command = "";
			}
		}
		
		cout<<"cnt:"<<cnt++<<endl;
		
		waitKey(33);
	}

	return 0;
}

