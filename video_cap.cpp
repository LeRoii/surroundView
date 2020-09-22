#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <thread>

using namespace cv;
using namespace std;
using std::thread;


string command;

void keboardListener()
{
	while(1)
    {
        cin>>command;
        cout<<"command:"<<command<<endl;
    }
}

int main()
{
	VideoCapture cap(0);
	cap.release();
	cap = VideoCapture(0);
	//gst_testsink.open("appsrc ! queue ! videoconvert ! video/x-raw, format=RGBA ! nvvidconv ! nvoverlaysink ", cv::CAP_GSTREAMER, 0, 3, cv::Size(1280, 720));
	cv::Mat img;

	double fps=0;

	cap.set(cv::CAP_PROP_FRAME_WIDTH,1280);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT,720);
	//cap.set(cv::CAP_PROP_FPS,30);

	thread keyboardListenerTh(keboardListener);
	int cnt = 0;
	int screenshotCnt = 1;
	while(true)
	{
	
		if (!cap.read(img)) {
       		std::cout<<"Capture read error"<<std::endl;
       		break;
   		}
		imshow("CamShow",img);
		imwrite("frame.png",img);
		if(command == "p")
		{
			string filename;
			std::stringstream StrStm;
			StrStm<<screenshotCnt++;
			StrStm>>filename;
			imwrite("img"+filename+".png",img);
			command = "";
		}
		
		fps = cap.get(cv::CAP_PROP_FPS);
		cout<<"FPS:"<<fps<<"cnt:"<<cnt++<<endl;
		

		waitKey(33);
	}

	cap.release();
	
	return 0;
}

