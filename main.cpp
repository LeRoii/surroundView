#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stdio.h>
#include <string>
#include <vector>

#include <chrono>
#include <thread>

using namespace cv;
using std::cout;
using std::endl;
using std::vector;
using std::thread;
//using namespace std;

const int CAMERA_NUM = 4;
const int CAMERA_FRAME_WIDTH = 1280;
const int CAMERA_FRAME_HEIGHT = 720;

const int SURROUND_VIEW_IMG_WIDTH = 1280;
const int SURROUND_VIEW_IMG_HEIGHT = 1625;

const int PERSPECTIVE_IMT_WIDTH = 1280;
const int PERSPECTIVE_IMT_HEIGHT = 720;

const int CAR_IMG_WIDTH = 640;
const int CAR_IMG_HEIGHT = 622;
const int CAR_IMG_START_X = SURROUND_VIEW_IMG_WIDTH/2 - CAR_IMG_WIDTH/2;
const int CAR_IMG_START_Y = 500;//SURROUND_VIEW_IMG_HEIGHT/2 - CAR_IMG_HEIGHT/2;

//pixel on surround view img
const int FRONT_VIEW_DIST = 500;//in pixel

const int FRONT_CROPED_START_X = 0;
const int FRONT_CROPED_START_Y = 0;
const int RIGHT_CROPED_START_X = SURROUND_VIEW_IMG_WIDTH/2 + CAR_IMG_WIDTH/2 - 20;
const int RIGHT_CROPED_START_Y = 43;
const int LEFT_CROPED_START_X = 0;
const int LEFT_CROPED_START_Y = 260;
const int BACK_CROPED_START_X = 0;
const int BACK_CROPED_START_Y = FRONT_VIEW_DIST + CAR_IMG_HEIGHT;

const int BACK_VIEW_DIST = 500;

//pixel on perspective img
const int FRONT_IMG_CROP_START_X = 0;
const int FRONT_IMG_CROP_START_Y = 674 - FRONT_VIEW_DIST;
const int FRONT_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH;
const int FRONT_IMG_CROP_HEIGHT = FRONT_VIEW_DIST;

const int BACK_IMG_CROP_HEIGHT = BACK_VIEW_DIST;
const int BACK_IMG_CROP_START_X = 0;
const int BACK_IMG_CROP_START_Y = 680 - BACK_VIEW_DIST;
const int BACK_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH;

const int RIGHT_IMG_CROP_START_X = 28;
const int RIGHT_IMG_CROP_START_Y = 0;
const int RIGHT_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH - RIGHT_CROPED_START_X;
const int RIGHT_IMG_CROP_HEIGHT = 1280;

const int LEFT_IMG_CROP_WIDTH = (SURROUND_VIEW_IMG_WIDTH - CAR_IMG_WIDTH)/2;
const int LEFT_IMG_CROP_HEIGHT = RIGHT_IMG_CROP_HEIGHT;
const int LEFT_IMG_CROP_START_X = 642 - LEFT_IMG_CROP_WIDTH;//right edge on rotated left img
const int LEFT_IMG_CROP_START_Y = 0;

const int FRONT_RIGHT_MERGE_ROW_DIFF = RIGHT_CROPED_START_Y - RIGHT_IMG_CROP_START_Y;
const int FRONT_RIGHT_MERGE_COL_DIFF = RIGHT_CROPED_START_X - RIGHT_IMG_CROP_START_X;
const int FRONT_LEFT_MERGE_ROW_DIFF = LEFT_CROPED_START_Y - LEFT_IMG_CROP_START_Y;
const int FRONT_LEFT_MERGE_COL_DIFF = LEFT_CROPED_START_X - LEFT_IMG_CROP_START_X;

const int TOP_MERGE_START_Y = 470;//280;
const int BOT_MERGE_END_Y = 1440;//1322;


// cv::Mat mapx = cv::Mat(imgSize,CV_32FC1);
// cv::Mat mapy = cv::Mat(imgSize,CV_32FC1);
cv::Mat R = cv::Mat::eye(3,3,CV_32F);

cv::Mat mapx[CAMERA_NUM];
cv::Mat mapy[CAMERA_NUM];

cv::VideoCapture cameras[CAMERA_NUM];
cv::Mat frames[CAMERA_NUM];
cv::Mat undistoredFrames[CAMERA_NUM];
cv::Mat persprctiveImgs[CAMERA_NUM];
cv::Mat perspectiveHomography[CAMERA_NUM];
// cv::cuda::GpuMat gpuInput[CAMERA_NUM];
// cv::cuda::GpuMat gpuOutput[CAMERA_NUM];

void frameProc(int camId)
{
	cameras[camId] >> frames[camId];
	cv::remap(frames[camId], undistoredFrames[camId], mapx[camId], mapy[camId], cv::INTER_LINEAR);
	cv::warpPerspective(undistoredFrames[camId], persprctiveImgs[camId], perspectiveHomography[camId], Size(PERSPECTIVE_IMT_WIDTH,PERSPECTIVE_IMT_HEIGHT));
	// gpuInput[camId].upload(undistoredFrames[camId]);
	// cv::cuda::warpPerspective( gpuInput[camId], gpuOutput[camId], perspectiveHomography[camId], Size(PERSPECTIVE_IMT_WIDTH,PERSPECTIVE_IMT_HEIGHT));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	// gpuOutput[camId].download(persprctiveImgs[camId]);
}

int main()
{
	// int num_devices = cv::cuda::getCudaEnabledDeviceCount();

    // if (num_devices <= 0) {
    //     std::cerr << "There is no device." << std::endl;
    //     return -1;
    // }
    // int enable_device_id = -1;
    // for (int i = 0; i < num_devices; i++) {
    //     cv::cuda::DeviceInfo dev_info(i);
    //     if (dev_info.isCompatible()) {
    //         enable_device_id = i;
    //     }
    // }
    // if (enable_device_id < 0) {
    //     std::cerr << "GPU module isn't built for GPU" << std::endl;
    //     return -1;
    // }
    // cv::cuda::setDevice(enable_device_id);

    // std::cout << "GPU is ready, device ID is " << num_devices << "\n";


	cv::Size imgSize = Size(CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT);
	float scale = 1.0;
	cv::Size undistorSize = Size(imgSize.width*scale,imgSize.height*scale);
    
	cv::Matx33d intrinsic_matrix[CAMERA_NUM];
	cv::Vec4d distortion_coeffs[CAMERA_NUM];
	cv::Mat newMatrix[CAMERA_NUM];

	for(int i=0;i<CAMERA_NUM;i++)
	{
		newMatrix[i] = cv::Mat::eye(3,3,CV_32F);
		mapx[i] = cv::Mat(undistorSize,CV_32FC1);
		mapy[i] = cv::Mat(undistorSize,CV_32FC1);
	}

    //front
    intrinsic_matrix[0] << 499.2256225154061, 0, 685.0325527895111,
                        0, 499.6344093018186, 288.8632118906361,
                        0, 0, 1;

    distortion_coeffs[0] << -0.0230412, 0.00631978, -0.00455568, 0.000311248;

    //left
    intrinsic_matrix[1] << 512.0799991633208, 0, 681.3682183385124,
                        0, 511.931977341321, 348.725565495493,
                        0, 0, 1;

    distortion_coeffs[1] << -0.0309463, 0.00392602, -0.00515291, 0.00102781;

	//back
    intrinsic_matrix[2] << 500.8548704340391, 0, 644.1812130625166,
                        0, 499.9234264350891, 391.6005802176933,
                        0, 0, 1;

    distortion_coeffs[2] << -0.0136425, -0.0220779, 0.0208222, -0.00740363;

	//right
    intrinsic_matrix[3] << 499.9046978644982, 0, 612.955400120308,
                        0, 500.02613225669, 357.855947068545,
                        0, 0, 1;

    distortion_coeffs[3] << -0.0248636, 0.0124981, -0.0126063, 0.00352282;

	newMatrix[0] = (cv::Mat_<float>(3,3)<< 136.20689, 0, 869, 0, 158.6207, 363, 0, 0, 1);//front
	newMatrix[1] = (cv::Mat_<float>(3,3)<< 165.0714, 0, 719, 0, 183.35963, 411, 0, 0, 1);//left
	newMatrix[2] = (cv::Mat_<float>(3,3)<< 272.40869, 0, 640, 0, 209.15546, 360, 0, 0, 1);//back
	newMatrix[3] = (cv::Mat_<float>(3,3)<< 165.0714, 0, 737, 0, 183.35963, 451, 0, 0, 1);//right


	for(int i=0;i<CAMERA_NUM;i++)
	{
		cout<<"camera:"<<i<<endl<<intrinsic_matrix[i]<<endl<<distortion_coeffs[i]<<endl;
		cv::fisheye::initUndistortRectifyMap(intrinsic_matrix[i], distortion_coeffs[i], R, newMatrix[i], imgSize, CV_32FC1, mapx[i], mapy[i]);
	}

	// for(int i=0;i<CAMERA_NUM;i++)
	for(int i=0;i<2;i++)
	{
		cameras[i].open(i*2);
		// cameras[i].release();
		// cameras[i].open(i*2);
		if(!cameras[i].isOpened())
		{
			printf("camera %d open failed\n",i);
			return 0;
		}
		cameras[i].set(cv::CAP_PROP_FRAME_WIDTH,CAMERA_FRAME_WIDTH);
		cameras[i].set(cv::CAP_PROP_FRAME_HEIGHT,CAMERA_FRAME_HEIGHT);
	}

	double fps,width,height;

	cv::Mat perspectiveFront, perspectiveLeft, perspectiveRight, perspectiveBack;
	cv::Mat frontCroped, leftCroped, rightCroped, backCroped, ret;
	
	//for every vector, 4 points are stored in order of leftbottom, rightbottom, righttop, lefttop
	vector<Point2f> srcPts[CAMERA_NUM] = {
		{cv::Point2f(186,517), cv::Point2f(615, 357), cv::Point2f(1059,367), cv::Point2f(1226,461)},	//front camera
		{cv::Point2f(399, 579), cv::Point2f(509, 494), cv::Point2f(938, 495), cv::Point2f(1025, 558)},	//left camera
		{cv::Point2f(174,422), cv::Point2f(523, 278), cv::Point2f(871,302), cv::Point2f(1177,399)},	//back camera
		{cv::Point2f(54, 681), cv::Point2f(199, 607), cv::Point2f(1048, 600), cv::Point2f(1107, 648)},	//right camera
	};

	vector<Point2f> dstPts[CAMERA_NUM] = {
		{cv::Point2f(218,665), cv::Point2f(222, 454), cv::Point2f(998, 464), cv::Point2f(997,615)},
		{cv::Point2f(317, 634), cv::Point2f(314, 510), cv::Point2f(1022, 502), cv::Point2f(1022, 608)},
		{cv::Point2f(342,639), cv::Point2f(348, 266), cv::Point2f(1083,432), cv::Point2f(1082,615)},
		{cv::Point2f(281, 701), cv::Point2f(284, 650), cv::Point2f(1145, 643), cv::Point2f(1144, 680)},
	};

	for(int i=0;i<CAMERA_NUM;i++)
	{
		perspectiveHomography[i] = findHomography(srcPts[i], dstPts[i]);
	}

	ret = cv::Mat(Size(SURROUND_VIEW_IMG_WIDTH, SURROUND_VIEW_IMG_HEIGHT),CV_8UC3,Scalar(0, 0, 0));
	cv::Mat carImg = cv::imread("car.png");
	cv::resize(carImg, carImg, cv::Size(CAR_IMG_WIDTH, CAR_IMG_HEIGHT+80));
	
	int frameCnt = 0;
	while(true)
	{
		auto startTime = std::chrono::steady_clock::now();

		thread frontFrameProcThread(frameProc,0);
		thread leftFrameProcThread(frameProc,1);
		thread backFrameProcThread(frameProc,2);
		thread rightFrameProcThread(frameProc,3);

		frontFrameProcThread.join();
		leftFrameProcThread.join();
		backFrameProcThread.join();
		rightFrameProcThread.join();

		auto endTime = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
		cout<<"before stich SpendTime = "<<  duration.count() <<"ms"<<endl;
		continue;

		perspectiveFront = persprctiveImgs[0];
		perspectiveLeft = persprctiveImgs[1];
		perspectiveBack = persprctiveImgs[2];
		perspectiveRight = persprctiveImgs[3];

		frontCroped = perspectiveFront(Rect(FRONT_IMG_CROP_START_X, FRONT_IMG_CROP_START_Y, FRONT_IMG_CROP_WIDTH, FRONT_IMG_CROP_HEIGHT)).clone();
		backCroped = perspectiveBack(Rect(BACK_IMG_CROP_START_X, BACK_IMG_CROP_START_Y, BACK_IMG_CROP_WIDTH, BACK_IMG_CROP_HEIGHT)).clone();
		
		flip(backCroped,backCroped,-1);

		transpose(perspectiveRight, perspectiveRight);
		flip(perspectiveRight, perspectiveRight, 1);
		
		transpose(perspectiveLeft, perspectiveLeft);
		flip(perspectiveLeft, perspectiveLeft, 0);

		rightCroped = perspectiveRight(Rect(RIGHT_IMG_CROP_START_X, RIGHT_IMG_CROP_START_Y, RIGHT_IMG_CROP_WIDTH, RIGHT_IMG_CROP_HEIGHT)).clone();
		leftCroped = perspectiveLeft(Rect(LEFT_IMG_CROP_START_X, LEFT_IMG_CROP_START_Y, LEFT_IMG_CROP_WIDTH, LEFT_IMG_CROP_HEIGHT)).clone();

		rightCroped.copyTo(ret(Rect(RIGHT_CROPED_START_X, RIGHT_CROPED_START_Y, RIGHT_IMG_CROP_WIDTH,RIGHT_IMG_CROP_HEIGHT)));
		leftCroped.copyTo(ret(Rect(LEFT_CROPED_START_X, LEFT_CROPED_START_Y, LEFT_IMG_CROP_WIDTH, LEFT_IMG_CROP_HEIGHT)));
		frontCroped.copyTo(ret(Rect(FRONT_CROPED_START_X, FRONT_CROPED_START_Y, FRONT_IMG_CROP_WIDTH, FRONT_IMG_CROP_HEIGHT)));
		backCroped.copyTo(ret(Rect(BACK_CROPED_START_X, BACK_CROPED_START_Y, BACK_IMG_CROP_WIDTH, BACK_IMG_CROP_HEIGHT)));
		carImg.copyTo(ret(Rect(CAR_IMG_START_X, CAR_IMG_START_Y, CAR_IMG_WIDTH, CAR_IMG_HEIGHT+80)));


		//front right merge
		int x1 = PERSPECTIVE_IMT_WIDTH;
		int y1 = TOP_MERGE_START_Y;
		int x2 = RIGHT_CROPED_START_X;
		int y2 = FRONT_VIEW_DIST;

		int mergeColStart;// = PERSPECTIVE_IMT_WIDTH;

		for(int i = y1;i<=y2;i++)
		{
			mergeColStart = (i-y1)*(x2-x1)/(y2-y1)+x1;
			for(;mergeColStart<=SURROUND_VIEW_IMG_WIDTH;mergeColStart++)
			{
				ret.at<Vec3b>(i,mergeColStart)[0] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[0];
				ret.at<Vec3b>(i,mergeColStart)[1] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[1];
				ret.at<Vec3b>(i,mergeColStart)[2] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[2];

				// cout<<"	front right merge area:"<<mergeColStart<<",,";
			}
		}

		//front left merge
		x1 = LEFT_CROPED_START_X;
		y1 = TOP_MERGE_START_Y;
		x2 = LEFT_IMG_CROP_WIDTH;
		y2 = FRONT_VIEW_DIST;

		for(int i = y1;i<=y2;i++)
		{
			int j_limt = (i-y1)*(x2-x1)/(y2-y1)+x1;
			for(int j=0;j<=j_limt;j++)
			{
				ret.at<Vec3b>(i,j)[0] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[0];
				ret.at<Vec3b>(i,j)[1] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[1];
				ret.at<Vec3b>(i,j)[2] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[2];
			}
		}

		//back left merge
		x1 = LEFT_CROPED_START_X+208;
		y1 = BOT_MERGE_END_Y;
		x2 = LEFT_IMG_CROP_WIDTH;
		y2 = BACK_CROPED_START_Y;

		for(int i=y2;i<y1;i++)
		{
			int j_limt = (i-y1)*(x2-x1)/(y2-y1)+x1;
			for(int j=0;j<=j_limt;j++)
			{
				ret.at<Vec3b>(i,j)[0] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[0];
				ret.at<Vec3b>(i,j)[1] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[1];
				ret.at<Vec3b>(i,j)[2] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[2];
			}
		}

		//back right merge
		x1 = SURROUND_VIEW_IMG_WIDTH;
		y1 = BOT_MERGE_END_Y;
		x2 = RIGHT_CROPED_START_X;
		y2 = BACK_CROPED_START_Y;

		for(int i=y2;i<y1;i++)
		{
			mergeColStart = (i-y1)*(x2-x1)/(y2-y1)+x1;
			for(;mergeColStart<=x1;mergeColStart++)
			{
				ret.at<Vec3b>(i,mergeColStart)[0] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[0];
				ret.at<Vec3b>(i,mergeColStart)[1] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[1];
				ret.at<Vec3b>(i,mergeColStart)[2] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[2];

				// cout<<"	back right merge area:"<<mergeColStart<<",,";
			}
		}
		endTime = std::chrono::steady_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
		cout<<"one frame SpendTime = "<<  duration.count() <<"ms"<<endl;

		cv::Mat rett;
		cv::resize(ret,rett,Size(1024,1024));
		cv::imshow("ret",rett);
		cv::waitKey(1);
	}

		//gpuOutput.release();
		//gpuInput.release();
	
	return 0;
}

