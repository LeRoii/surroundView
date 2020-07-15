#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stdio.h>
#include <string>
#include <vector>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

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

const int PERSPECTIVE_IMT_WIDTH = 2030;
const int PERSPECTIVE_IMT_HEIGHT = 1200;

const int SURROUND_VIEW_IMG_WIDTH = 2030;
const int SURROUND_VIEW_IMG_HEIGHT = 2400;

const int X_EXPAND = 750;
const int Y_EXPAND = 480;

//pixel on surround view img
const int FRONT_VIEW_DIST = 500;//in pixel
const int LEFT_VIEW_DIST = 480;
const int BACK_VIEW_DIST = 500;
const int RIGHT_VIEW_DIST = 480;

const int FRONT_CROPED_START_X = 0;
const int FRONT_CROPED_START_Y = 0;
const int RIGHT_CROPED_START_X =1450;
const int RIGHT_CROPED_START_Y = FRONT_VIEW_DIST;
const int LEFT_CROPED_START_X = 0;
const int LEFT_CROPED_START_Y = FRONT_VIEW_DIST;
const int BACK_CROPED_START_X = 0;
const int BACK_CROPED_START_Y = 1900;

const int TOP_MERGE_START_Y = 140;
const int BOT_MERGE_END_Y = 2100;


//pixel on perspective img
const int FRONT_IMG_CROP_START_X = 0;
const int FRONT_IMG_CROP_START_Y = 290;
const int FRONT_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH;
const int FRONT_IMG_CROP_HEIGHT = FRONT_VIEW_DIST;

const int BACK_IMG_CROP_START_X = 0;
const int BACK_IMG_CROP_START_Y = 470;
const int BACK_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH;
const int BACK_IMG_CROP_HEIGHT = BACK_VIEW_DIST;

const int RIGHT_IMG_CROP_START_X = 278;
const int RIGHT_IMG_CROP_START_Y = 440;
const int RIGHT_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH - RIGHT_CROPED_START_X;
const int RIGHT_IMG_CROP_HEIGHT = PERSPECTIVE_IMT_HEIGHT * 2 - FRONT_VIEW_DIST - BACK_VIEW_DIST;

const int LEFT_IMG_CROP_START_X = 300;
const int LEFT_IMG_CROP_START_Y = 270;
const int LEFT_IMG_CROP_WIDTH = 500;
const int LEFT_IMG_CROP_HEIGHT = PERSPECTIVE_IMT_HEIGHT * 2 - FRONT_VIEW_DIST - BACK_VIEW_DIST;

const int FRONT_RIGHT_MERGE_ROW_DIFF = RIGHT_CROPED_START_Y - RIGHT_IMG_CROP_START_Y;
const int FRONT_RIGHT_MERGE_COL_DIFF = RIGHT_CROPED_START_X - RIGHT_IMG_CROP_START_X;
const int FRONT_LEFT_MERGE_ROW_DIFF = LEFT_CROPED_START_Y - LEFT_IMG_CROP_START_Y;
const int FRONT_LEFT_MERGE_COL_DIFF = LEFT_CROPED_START_X - LEFT_IMG_CROP_START_X;

cv::Size imgSize((CAMERA_FRAME_WIDTH+X_EXPAND)*1.0,(CAMERA_FRAME_HEIGHT+Y_EXPAND)*1.0);

cv::Mat mapx = cv::Mat(imgSize,CV_32FC1);
cv::Mat mapy = cv::Mat(imgSize,CV_32FC1);
cv::Mat R = cv::Mat::eye(3,3,CV_32F);

cv::VideoCapture cameras[CAMERA_NUM];
cv::Mat frames[CAMERA_NUM];
cv::Mat undistoredFrames[CAMERA_NUM];
cv::Mat persprctiveImgs[CAMERA_NUM];
cv::Mat perspectiveHomography[CAMERA_NUM];
cv::cuda::GpuMat gpuInput[CAMERA_NUM];
cv::cuda::GpuMat gpuOutput[CAMERA_NUM];

void frameProc(int camId)
{
	cameras[camId] >> frames[camId];
	cv::Mat borderDistoredImage = frames[camId].clone();
	copyMakeBorder(frames[camId],borderDistoredImage,(int)(Y_EXPAND/2),(int)(Y_EXPAND/2),(int)(X_EXPAND/2),(int)(X_EXPAND/2),BORDER_CONSTANT);

	//cv::imwrite("borderDistoredImage"+imageFileName,borderDistoredImage);
	cv::remap(borderDistoredImage,undistoredFrames[camId],mapx, mapy, cv::INTER_LINEAR);

	//cv::imwrite("undistored"+imageFileName,undistoredFrames[i]);

	//warpPerspective(result, img1, h, result.size(), 2);
	gpuInput[camId].upload(undistoredFrames[camId]);
	cv::cuda::warpPerspective( gpuInput[camId], gpuOutput[camId], perspectiveHomography[camId], Size(PERSPECTIVE_IMT_WIDTH,PERSPECTIVE_IMT_HEIGHT));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	gpuOutput[camId].download(persprctiveImgs[camId]);
}

int main()
{
	int num_devices = cv::cuda::getCudaEnabledDeviceCount();

    if (num_devices <= 0) {
        std::cerr << "There is no device." << std::endl;
        return -1;
    }
    int enable_device_id = -1;
    for (int i = 0; i < num_devices; i++) {
        cv::cuda::DeviceInfo dev_info(i);
        if (dev_info.isCompatible()) {
            enable_device_id = i;
        }
    }
    if (enable_device_id < 0) {
        std::cerr << "GPU module isn't built for GPU" << std::endl;
        return -1;
    }
    cv::cuda::setDevice(enable_device_id);

    std::cout << "GPU is ready, device ID is " << num_devices << "\n";


	

	cv::Matx33d intrinsic_matrix;
	cv::Vec4d distortion_coeffs;
	//int x_expand = 750,y_expand = 480;		//x,y方向的扩展(x横向，y纵向)，适当增大可以不损失原图像信息	//1.8mm front ok
	//int x_expand = 0,y_expand = 200;
	//cv::Size imgSize = cv::Size(640,480);

	//1.8mm
	intrinsic_matrix << 765.4423103128532, 0, 958.402420837329,
 						0, 765.6356214447392, 687.4306063863424,
 						0, 0, 1;
	distortion_coeffs << -0.0316929, 0.0360623, -0.0623949, 0.0348068;

	//1280x720 200fov
	// intrinsic_matrix << 244.7879600344495, 0, 724.9293172169554,
 	// 					0, 244.8360344210398, 510.1527621243182,
	// 					0, 0, 1;
	// distortion_coeffs << -0.018679, -0.0218638, 0.0288774, -0.0139819;


	cout<<intrinsic_matrix<<endl;   
    cout<<distortion_coeffs<<endl; 

	float mmPerPiexl = 2.05f;

	cv::fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,intrinsic_matrix,imgSize,CV_32FC1,mapx,mapy);
/*
	//find H

	//right
	vector<Point2f> srcPts = {Point2f(313,751), Point2f(533,527), Point2f(1921,722), Point2f(1750,550)};
	vector<Point2f> dstPts = {Point2f(315,804), Point2f(448,530), Point2f(1918,764), Point2f(1819,555)};
	Mat hRight = findHomography(srcPts, dstPts);

	//front
	srcPts = {Point2f(291,771), Point2f(464, 845), Point2f(620, 468), Point2f(1362,384), Point2f(1652,703), Point2f(1919,568)};
	dstPts = {Point2f(1015-1560/mmPerPiexl, 771), Point2f(1015-870/mmPerPiexl, 771), Point2f(1015-870/mmPerPiexl, 771-850/mmPerPiexl),
				Point2f(1015+870/mmPerPiexl, 771-860/mmPerPiexl), Point2f(1015+870/mmPerPiexl, 771), Point2f(1015+1560/mmPerPiexl,771)};
	Mat hFront = findHomography(srcPts, dstPts);

	//left
	// srcPts = {Point2f(230,868), Point2f(390,975), Point2f(655,639), Point2f(1376,617), Point2f(1727,967), Point2f(1984,830)};
	// dstPts = {Point2f(227,984+0), Point2f(390,975), Point2f(389+100,605), Point2f(1727-100,590), Point2f(1727,967), Point2f(2008,962+0)};
	srcPts = {Point2f(282,785), Point2f(359,831), Point2f(474,515), Point2f(1644,583), Point2f(1768,817), Point2f(1955,741)};
	dstPts = {Point2f(270,828), Point2f(340,828), Point2f(340,522), Point2f(1780,550), Point2f(1780,828), Point2f(1960,828)};
	
	Mat hLeft = findHomography(srcPts, dstPts);

	//back
	srcPts = {Point2f(357, 805), Point2f(575, 892), Point2f(724, 447), Point2f(1297, 382), Point2f(1494, 894), Point2f(1926, 780)};
	dstPts = {Point2f(300, 878), Point2f(520,878+300), Point2f(520,450), Point2f(1442, 361), Point2f(1442,878), Point2f(1939, 878)};
	dstPts = {Point2f(1015-1680/mmPerPiexl, 878), Point2f(1015-800/mmPerPiexl,878), Point2f(1015-800/mmPerPiexl,878-1160/mmPerPiexl), 
			Point2f(1015+800/mmPerPiexl, 878-1380/mmPerPiexl), Point2f(1015+800/mmPerPiexl,878), Point2f(1015+2410/mmPerPiexl, 878)};
	Mat hBack = findHomography(srcPts, dstPts);


	srcPts = {Point2f(357, 805), Point2f(724, 447), Point2f(1297, 382), Point2f(1926, 780)};
	dstPts = {Point2f(260, 898), Point2f(520,450), Point2f(1442, 361), Point2f(1939, 878)};
	hBack = getPerspectiveTransform(srcPts, dstPts);

	cv::Mat distoredFront = cv::imread("front.png");
	cv::Mat distoredLeft = cv::imread("left.png");
	cv::Mat distoredRight = cv::imread("right.png");
	cv::Mat distoredBack = cv::imread("back.png");
	//imshow("111",distoredImg);
	
	cv::cuda::GpuMat gpuInput;
	cv::cuda::GpuMat gpuOutput;

	//cv::Size undistoredImgSize(CAMERA_FRAME_WIDTH*1,CAMERA_FRAME_HEIGHT*1);
	cv::Size imgSize((CAMERA_FRAME_WIDTH+x_expand)*1.0,(CAMERA_FRAME_HEIGHT+y_expand)*1.0);

	cv::Mat borderDistoredFront, borderDistoredLeft, borderDistoredRight, borderDistoredBack;
	copyMakeBorder(distoredFront, borderDistoredFront,(int)(y_expand/2), (int)(y_expand/2), (int)(x_expand/2), (int)(x_expand/2), BORDER_CONSTANT);
	copyMakeBorder(distoredLeft, borderDistoredLeft, (int)(y_expand/2), (int)(y_expand/2), (int)(x_expand/2), (int)(x_expand/2), BORDER_CONSTANT);
	copyMakeBorder(distoredRight, borderDistoredRight, (int)(y_expand/2), (int)(y_expand/2), (int)(x_expand/2), (int)(x_expand/2), BORDER_CONSTANT);
	copyMakeBorder(distoredBack, borderDistoredBack, (int)(y_expand/2), (int)(y_expand/2), (int)(x_expand/2), (int)(x_expand/2), BORDER_CONSTANT);

	cv::Mat mapx = cv::Mat(imgSize,CV_32FC1);
	cv::Mat mapy = cv::Mat(imgSize,CV_32FC1);
	cv::Mat R = cv::Mat::eye(3,3,CV_32F);
	
	//double alpha = 1.0; 
	//cv::Mat newCamMatrix = getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, Size(CAMERA_FRAME_WIDTH,CAMERA_FRAME_HEIGHT), alpha, undistoredImgSize);
	cv::fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, imgSize, CV_32FC1,mapx,mapy);
	//cv::imwrite("borderDistoredImage.png",borderDistoredImage);

	cv::Mat undistoredFront, undistoredLeft, undistoredRight, undistoredBack;
	cv::remap(borderDistoredFront, undistoredFront, mapx, mapy, cv::INTER_CUBIC);
	cv::remap(borderDistoredLeft, undistoredLeft, mapx, mapy, cv::INTER_CUBIC);
	cv::remap(borderDistoredRight, undistoredRight, mapx, mapy, cv::INTER_CUBIC);
	cv::remap(borderDistoredBack, undistoredBack, mapx, mapy, cv::INTER_CUBIC);

	cv::imwrite("undistoredFront.png",undistoredFront);
	cv::imwrite("undistoredLeft.png",undistoredLeft);
	cv::imwrite("undistoredRight.png",undistoredRight);
	cv::imwrite("undistoredBack.png",undistoredBack);


	cv::Mat perspectiveFront, perspectiveLeft, perspectiveRight, perspectiveBack;
	gpuInput.upload(undistoredFront);
	cv::cuda::warpPerspective( gpuInput, gpuOutput, hFront, Size(undistoredFront.size().width,undistoredFront.size().height+0));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	//cv::cuda::warpPerspective( gpuInput, gpuOutput, hFront, Size(1280,1280));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  

	gpuOutput.download(perspectiveFront);

	gpuInput.upload(undistoredLeft);
	cv::cuda::warpPerspective( gpuInput, gpuOutput, hLeft, Size(undistoredFront.size().width,undistoredFront.size().height+0));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	gpuOutput.download(perspectiveLeft);

	gpuInput.upload(undistoredRight);
	cv::cuda::warpPerspective( gpuInput, gpuOutput, hRight, Size(undistoredFront.size().width,undistoredFront.size().height+0));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	gpuOutput.download(perspectiveRight);

	gpuInput.upload(undistoredBack);
	cv::cuda::warpPerspective( gpuInput, gpuOutput, hBack, Size(undistoredFront.size().width,undistoredFront.size().height+0));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	gpuOutput.download(perspectiveBack);

	//resize(perspectiveImg,perspectiveImg,Size(1000,1000));

	cv::imwrite("perspectiveFront.png",perspectiveFront);
	cv::imwrite("perspectiveLeft.png",perspectiveLeft);
	cv::imwrite("perspectiveRight.png",perspectiveRight);
	cv::imwrite("perspectiveBack.png",perspectiveBack);

	const int PERSPECTIVE_IMT_WIDTH = 2030;
	const int PERSPECTIVE_IMT_HEIGHT = 1200;

	const int SURROUND_VIEW_IMG_WIDTH = 2030;
	const int SURROUND_VIEW_IMG_HEIGHT = 2400;

	//pixel on surround view img
	const int FRONT_VIEW_DIST = 500;//in pixel
	const int LEFT_VIEW_DIST = 480;
	const int BACK_VIEW_DIST = 500;
	const int RIGHT_VIEW_DIST = 480;

	const int FRONT_CROPED_START_X = 0;
	const int FRONT_CROPED_START_Y = 0;
	const int RIGHT_CROPED_START_X =1450;
	const int RIGHT_CROPED_START_Y = FRONT_VIEW_DIST;
	const int LEFT_CROPED_START_X = 0;
	const int LEFT_CROPED_START_Y = FRONT_VIEW_DIST;
	const int BACK_CROPED_START_X = 0;
	const int BACK_CROPED_START_Y = 1900;

	const int TOP_MERGE_START_Y = 140;
	const int BOT_MERGE_END_Y = 2100;


	//pixel on perspective img
	const int FRONT_IMG_CROP_START_X = 0;
	const int FRONT_IMG_CROP_START_Y = 290;
	const int FRONT_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH;
	const int FRONT_IMG_CROP_HEIGHT = FRONT_VIEW_DIST;

	const int BACK_IMG_CROP_START_X = 0;
	const int BACK_IMG_CROP_START_Y = 470;
	const int BACK_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH;
	const int BACK_IMG_CROP_HEIGHT = BACK_VIEW_DIST;

	const int RIGHT_IMG_CROP_START_X = 278;
	const int RIGHT_IMG_CROP_START_Y = 440;
	const int RIGHT_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH - RIGHT_CROPED_START_X;
	const int RIGHT_IMG_CROP_HEIGHT = PERSPECTIVE_IMT_HEIGHT * 2 - FRONT_VIEW_DIST - BACK_VIEW_DIST;

	const int LEFT_IMG_CROP_START_X = 300;
	const int LEFT_IMG_CROP_START_Y = 270;
	const int LEFT_IMG_CROP_WIDTH = 500;
	const int LEFT_IMG_CROP_HEIGHT = PERSPECTIVE_IMT_HEIGHT * 2 - FRONT_VIEW_DIST - BACK_VIEW_DIST;

	const int FRONT_RIGHT_MERGE_ROW_DIFF = RIGHT_CROPED_START_Y - RIGHT_IMG_CROP_START_Y;
	const int FRONT_RIGHT_MERGE_COL_DIFF = RIGHT_CROPED_START_X - RIGHT_IMG_CROP_START_X;
	const int FRONT_LEFT_MERGE_ROW_DIFF = LEFT_CROPED_START_Y - LEFT_IMG_CROP_START_Y;
	const int FRONT_LEFT_MERGE_COL_DIFF = LEFT_CROPED_START_X - LEFT_IMG_CROP_START_X;

	cv::Mat frontCroped, leftCroped, rightCroped, backCroped, ret;
	//right = cv::imread("rightperspectiveImg.png");
	ret = cv::Mat(Size(perspectiveFront.size().width, perspectiveFront.size().height*2),CV_8UC3,Scalar(255, 0, 0));
	//ret = cv::imread("rett.png");
	
	frontCroped = perspectiveFront(Rect(FRONT_IMG_CROP_START_X, FRONT_IMG_CROP_START_Y, FRONT_IMG_CROP_WIDTH, FRONT_IMG_CROP_HEIGHT)).clone();
	backCroped = perspectiveBack(Rect(BACK_IMG_CROP_START_X, BACK_IMG_CROP_START_Y, BACK_IMG_CROP_WIDTH, BACK_IMG_CROP_HEIGHT)).clone();

	flip(backCroped,backCroped,-1);

	transpose(perspectiveRight, perspectiveRight);
	flip(perspectiveRight, perspectiveRight, 1);
	cv::imwrite("rotatedright.png",perspectiveRight);

	transpose(perspectiveLeft, perspectiveLeft);
	flip(perspectiveLeft, perspectiveLeft, 0);
	cv::imwrite("rotatedleft.png",perspectiveLeft);

	// cout<<"llllllllllllll:"<<right.size()<<endl;
	// cout<<ret.size()<<endl;
	rightCroped = perspectiveRight(Rect(RIGHT_IMG_CROP_START_X,RIGHT_IMG_CROP_START_Y,RIGHT_IMG_CROP_WIDTH,RIGHT_IMG_CROP_HEIGHT)).clone();
	leftCroped = perspectiveLeft(Rect(LEFT_IMG_CROP_START_X, LEFT_IMG_CROP_START_Y, LEFT_IMG_CROP_WIDTH, LEFT_IMG_CROP_HEIGHT)).clone();

	cv::imwrite("rightCroped.png", rightCroped);
	cv::imwrite("leftCroped.png", leftCroped);
	cv::imwrite("frontCroped.png", frontCroped);
	cv::imwrite("backCroped.png", backCroped);

	rightCroped.copyTo(ret(Rect(RIGHT_CROPED_START_X, RIGHT_CROPED_START_Y, RIGHT_IMG_CROP_WIDTH,RIGHT_IMG_CROP_HEIGHT)));
	frontCroped.copyTo(ret(Rect(FRONT_CROPED_START_X, FRONT_CROPED_START_Y, FRONT_IMG_CROP_WIDTH,FRONT_IMG_CROP_HEIGHT)));
	backCroped.copyTo(ret(Rect(BACK_CROPED_START_X, BACK_CROPED_START_Y, BACK_IMG_CROP_WIDTH, BACK_IMG_CROP_HEIGHT)));
	leftCroped.copyTo(ret(Rect(LEFT_CROPED_START_X, LEFT_CROPED_START_Y, LEFT_IMG_CROP_WIDTH, LEFT_IMG_CROP_HEIGHT)));

	printf("251:::::%d,,%d\n",(251-TOP_MERGE_START_Y)/(FRONT_VIEW_DIST-TOP_MERGE_START_Y)*(RIGHT_CROPED_START_X-PERSPECTIVE_IMT_WIDTH)+PERSPECTIVE_IMT_WIDTH,
	(251-250)*(1650-2130)/(500-250)+2130);

	int mergeColStart;// = PERSPECTIVE_IMT_WIDTH;
	for(int i = TOP_MERGE_START_Y;i<=FRONT_VIEW_DIST;i++)
	//for(int i = TOP_MERGE_START_Y;i<=TOP_MERGE_START_Y+3;i++)
	{
		//mergeColStart = 2130 - (i-250)*48/25;//	front right merge area
		mergeColStart = (i-TOP_MERGE_START_Y)*(RIGHT_CROPED_START_X-PERSPECTIVE_IMT_WIDTH)/(FRONT_VIEW_DIST-TOP_MERGE_START_Y)+PERSPECTIVE_IMT_WIDTH;
		for(;mergeColStart<=SURROUND_VIEW_IMG_WIDTH;mergeColStart++)
		{
			ret.at<Vec3b>(i,mergeColStart)[0] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[0];
			ret.at<Vec3b>(i,mergeColStart)[1] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[1];
			ret.at<Vec3b>(i,mergeColStart)[2] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[2];

			//cout<<"	front right merge area:"<<mergeColStart<<",,";
		}
		//cout<<"i=="<<i<<endl;

		//front left merge area
		int j_limt = (i-TOP_MERGE_START_Y)*(LEFT_IMG_CROP_WIDTH-LEFT_CROPED_START_X)/(FRONT_VIEW_DIST-TOP_MERGE_START_Y)+LEFT_CROPED_START_X;
		for(int j=0;j<=j_limt;j++)
		{
			ret.at<Vec3b>(i,j)[0] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[0];
			ret.at<Vec3b>(i,j)[1] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[1];
			ret.at<Vec3b>(i,j)[2] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[2];
		}
	}

	for(int i=FRONT_VIEW_DIST+LEFT_IMG_CROP_HEIGHT;i<BOT_MERGE_END_Y;i++)
	{
		//back left merge
		int j_limt = (i-BOT_MERGE_END_Y)*(LEFT_IMG_CROP_WIDTH-LEFT_CROPED_START_X)/(FRONT_VIEW_DIST+LEFT_IMG_CROP_HEIGHT-BOT_MERGE_END_Y)+LEFT_CROPED_START_X;
		for(int j=0;j<=j_limt;j++)
		{
			ret.at<Vec3b>(i,j)[0] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[0];
			ret.at<Vec3b>(i,j)[1] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[1];
			ret.at<Vec3b>(i,j)[2] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[2];
		}

		//back right merge
		mergeColStart = (i-BOT_MERGE_END_Y)*(RIGHT_CROPED_START_X-SURROUND_VIEW_IMG_HEIGHT)/(FRONT_VIEW_DIST+LEFT_IMG_CROP_HEIGHT-BOT_MERGE_END_Y)+SURROUND_VIEW_IMG_HEIGHT;
		for(;mergeColStart<=SURROUND_VIEW_IMG_WIDTH;mergeColStart++)
		{
			ret.at<Vec3b>(i,mergeColStart)[0] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[0];
			ret.at<Vec3b>(i,mergeColStart)[1] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[1];
			ret.at<Vec3b>(i,mergeColStart)[2] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[2];

			//cout<<"	back right merge area:"<<mergeColStart<<",,";
		}
	}

	cv::imwrite("ret2.png",ret);


return 0;
	
*/

	for(int i=0;i<CAMERA_NUM;i++)
	{
		cameras[i].open(i);
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
	cv::Size undistoredImgSize((CAMERA_FRAME_WIDTH+X_EXPAND)*1.5,(CAMERA_FRAME_HEIGHT+Y_EXPAND)*1.5);
	
	//for every vector, 4 points are stored in order of leftbottom, rightbottom, righttop, lefttop
	vector<Point2f> srcPts[CAMERA_NUM] = {
		{Point2f(291,771), Point2f(464, 845), Point2f(620, 468), Point2f(1362,384), Point2f(1652,703), Point2f(1919,568)},	//front camera
		{Point2f(282,785), Point2f(359,831), Point2f(474,515), Point2f(1644,583), Point2f(1768,817), Point2f(1955,741)},	//left camera
		{Point2f(357, 805), Point2f(575, 892), Point2f(724, 447), Point2f(1297, 382), Point2f(1494, 894), Point2f(1926, 780)},	//back camera
		{Point2f(313,751), Point2f(533,527), Point2f(1921,722), Point2f(1750,550)},	//right camera
	};

	vector<Point2f> dstPts[CAMERA_NUM] = {
		{Point2f(1015-1560/mmPerPiexl, 771), Point2f(1015-870/mmPerPiexl, 771), Point2f(1015-870/mmPerPiexl, 771-850/mmPerPiexl),
		Point2f(1015+870/mmPerPiexl, 771-860/mmPerPiexl), Point2f(1015+870/mmPerPiexl, 771), Point2f(1015+1560/mmPerPiexl,771)},
		{Point2f(270,828), Point2f(340,828), Point2f(340,522), Point2f(1780,550), Point2f(1780,828), Point2f(1960,828)},
		{Point2f(1015-1680/mmPerPiexl, 878), Point2f(1015-800/mmPerPiexl,878), Point2f(1015-800/mmPerPiexl,878-1160/mmPerPiexl), 
			Point2f(1015+800/mmPerPiexl, 878-1380/mmPerPiexl), Point2f(1015+800/mmPerPiexl,878), Point2f(1015+2410/mmPerPiexl, 878)},
		{Point2f(315,804), Point2f(448,530), Point2f(1918,764), Point2f(1819,555)},
	};

	for(int i=0;i<CAMERA_NUM;i++)
	{
		perspectiveHomography[i] = findHomography(srcPts[i], dstPts[i]);
	}

	ret = cv::Mat(Size(PERSPECTIVE_IMT_WIDTH, PERSPECTIVE_IMT_HEIGHT*2),CV_8UC3,Scalar(255, 0, 0));


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
		// for(int i=0;i<CAMERA_NUM;i++)
		// {
		// 	cameras[i] >> frames[i];
		// 	std::string imageFileName;
        // 	std::stringstream StrStm;
		// 	StrStm<<"frame"<<i<<".png";
		// 	StrStm>>imageFileName;
		// 	//cv::imwrite(imageFileName,frames[i]);
		// 	//cv::imshow("ret",frames[i]);

		// 	fps = cameras[0].get(cv::CAP_PROP_FPS);
		// 	width = cameras[0].get(cv::CAP_PROP_FRAME_WIDTH);
		// 	height = cameras[0].get(cv::CAP_PROP_FRAME_HEIGHT);
		// 	printf("frameCnt:%d, fps:%f,width:%f,height:%f\n",frameCnt++,fps,width,height);

		// 	cv::Mat borderDistoredImage = frames[i].clone();
		// 	copyMakeBorder(frames[i],borderDistoredImage,(int)(y_expand/2),(int)(y_expand/2),(int)(x_expand/2),(int)(x_expand/2),BORDER_CONSTANT);

		// 	cv::Mat mapx = cv::Mat(imgSize,CV_32FC1);
		// 	cv::Mat mapy = cv::Mat(imgSize,CV_32FC1);
		// 	cv::Mat R = cv::Mat::eye(3,3,CV_32F);

			
		// 	cv::fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,intrinsic_matrix,imgSize,CV_32FC1,mapx,mapy);
		// 	//cv::imwrite("borderDistoredImage"+imageFileName,borderDistoredImage);
		// 	cv::remap(borderDistoredImage,undistoredFrames[i],mapx, mapy, cv::INTER_LINEAR);

		// 	//cv::imwrite("undistored"+imageFileName,undistoredFrames[i]);


		// 	gpuInput.upload(undistoredFrames[i]);
		// 	cv::cuda::warpPerspective( gpuInput, gpuOutput, perspectiveHomography[i], Size(PERSPECTIVE_IMT_WIDTH,PERSPECTIVE_IMT_HEIGHT));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
		// 	gpuOutput.download(persprctiveImgs[i]);

		// 	cv::imwrite("persprctiveImgs"+imageFileName,persprctiveImgs[i]);

		// 	//resize(frames[i],frames[i],Size(640,480));
			
		// }

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
		//cv::imwrite("rotatedright.png",perspectiveRight);

		transpose(perspectiveLeft, perspectiveLeft);
		flip(perspectiveLeft, perspectiveLeft, 0);
		//cv::imwrite("rotatedleft.png",perspectiveLeft);

		// cout<<"llllllllllllll:"<<right.size()<<endl;
		// cout<<ret.size()<<endl;
		rightCroped = perspectiveRight(Rect(RIGHT_IMG_CROP_START_X,RIGHT_IMG_CROP_START_Y,RIGHT_IMG_CROP_WIDTH,RIGHT_IMG_CROP_HEIGHT)).clone();
		leftCroped = perspectiveLeft(Rect(LEFT_IMG_CROP_START_X, LEFT_IMG_CROP_START_Y, LEFT_IMG_CROP_WIDTH, LEFT_IMG_CROP_HEIGHT)).clone();

		// cv::imwrite("rightCroped.png", rightCroped);
		// cv::imwrite("leftCroped.png", leftCroped);
		// cv::imwrite("frontCroped.png", frontCroped);
		// cv::imwrite("backCroped.png", backCroped);

		rightCroped.copyTo(ret(Rect(RIGHT_CROPED_START_X, RIGHT_CROPED_START_Y, RIGHT_IMG_CROP_WIDTH,RIGHT_IMG_CROP_HEIGHT)));
		frontCroped.copyTo(ret(Rect(FRONT_CROPED_START_X, FRONT_CROPED_START_Y, FRONT_IMG_CROP_WIDTH,FRONT_IMG_CROP_HEIGHT)));
		backCroped.copyTo(ret(Rect(BACK_CROPED_START_X, BACK_CROPED_START_Y, BACK_IMG_CROP_WIDTH, BACK_IMG_CROP_HEIGHT)));
		leftCroped.copyTo(ret(Rect(LEFT_CROPED_START_X, LEFT_CROPED_START_Y, LEFT_IMG_CROP_WIDTH, LEFT_IMG_CROP_HEIGHT)));

		int mergeColStart;// = PERSPECTIVE_IMT_WIDTH;
		for(int i = TOP_MERGE_START_Y;i<=FRONT_VIEW_DIST;i++)
		//for(int i = TOP_MERGE_START_Y;i<=TOP_MERGE_START_Y+3;i++)
		{
			//mergeColStart = 2130 - (i-250)*48/25;//	front right merge area
			mergeColStart = (i-TOP_MERGE_START_Y)*(RIGHT_CROPED_START_X-PERSPECTIVE_IMT_WIDTH)/(FRONT_VIEW_DIST-TOP_MERGE_START_Y)+PERSPECTIVE_IMT_WIDTH;
			for(;mergeColStart<=SURROUND_VIEW_IMG_WIDTH;mergeColStart++)
			{
				ret.at<Vec3b>(i,mergeColStart)[0] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[0];
				ret.at<Vec3b>(i,mergeColStart)[1] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[1];
				ret.at<Vec3b>(i,mergeColStart)[2] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[2];

				//cout<<"	front right merge area:"<<mergeColStart<<",,";
			}
			//cout<<"i=="<<i<<endl;

			//front left merge area
			int j_limt = (i-TOP_MERGE_START_Y)*(LEFT_IMG_CROP_WIDTH-LEFT_CROPED_START_X)/(FRONT_VIEW_DIST-TOP_MERGE_START_Y)+LEFT_CROPED_START_X;
			for(int j=0;j<=j_limt;j++)
			{
				ret.at<Vec3b>(i,j)[0] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[0];
				ret.at<Vec3b>(i,j)[1] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[1];
				ret.at<Vec3b>(i,j)[2] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[2];
			}
		}

		for(int i=FRONT_VIEW_DIST+LEFT_IMG_CROP_HEIGHT;i<BOT_MERGE_END_Y;i++)
		{
			//back left merge
			int j_limt = (i-BOT_MERGE_END_Y)*(LEFT_IMG_CROP_WIDTH-LEFT_CROPED_START_X)/(FRONT_VIEW_DIST+LEFT_IMG_CROP_HEIGHT-BOT_MERGE_END_Y)+LEFT_CROPED_START_X;
			for(int j=0;j<=j_limt;j++)
			{
				ret.at<Vec3b>(i,j)[0] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[0];
				ret.at<Vec3b>(i,j)[1] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[1];
				ret.at<Vec3b>(i,j)[2] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[2];
			}

			//back right merge
			mergeColStart = (i-BOT_MERGE_END_Y)*(RIGHT_CROPED_START_X-SURROUND_VIEW_IMG_HEIGHT)/(FRONT_VIEW_DIST+LEFT_IMG_CROP_HEIGHT-BOT_MERGE_END_Y)+SURROUND_VIEW_IMG_HEIGHT;
			for(;mergeColStart<=SURROUND_VIEW_IMG_WIDTH;mergeColStart++)
			{
				ret.at<Vec3b>(i,mergeColStart)[0] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[0];
				ret.at<Vec3b>(i,mergeColStart)[1] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[1];
				ret.at<Vec3b>(i,mergeColStart)[2] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[2];

				//cout<<"	back right merge area:"<<mergeColStart<<",,";
			}
		}
			endTime = std::chrono::steady_clock::now();
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
			cout<<"one frame SpendTime = "<<  duration.count() <<"ms"<<endl;

			cv::Mat rett;
			cv::resize(ret,rett,Size(1024,1024));
			cv::imshow("ret",rett);
			//cv::waitKey(333);
		}

		//gpuOutput.release();
		//gpuInput.release();
	
	return 0;
}

