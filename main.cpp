#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stdio.h>
#include <string>
#include <vector>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

using namespace cv;
using std::cout;
using std::endl;
using std::vector;
//using namespace std;

const int CAMERA_NUM = 4;
const int CAMERA_FRAME_WIDTH = 1280;
const int CAMERA_FRAME_HEIGHT = 720;

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


	cv::VideoCapture cameras[CAMERA_NUM];

	cv::Matx33d intrinsic_matrix;
	cv::Vec4d distortion_coeffs;
	int x_expand = 750,y_expand = 480;		//x,y方向的扩展(x横向，y纵向)，适当增大可以不损失原图像信息	//1.8mm
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

	//find H

	//right
	// vector<Point2f> srcPts = {Point2f(313,751), Point2f(533,527), Point2f(1921,722), Point2f(1750,550)};
	// vector<Point2f> dstPts = {Point2f(315,804), Point2f(448,530), Point2f(1918,764), Point2f(1819,555)};

	//front
	vector<Point2f> srcPts = {Point2f(230,868), Point2f(390,975), Point2f(655,639), Point2f(1376,617), Point2f(1727,967), Point2f(1984,830)};
	vector<Point2f> dstPts = {Point2f(227,984+0), Point2f(390,975), Point2f(389+100,605), Point2f(1727-100,590), Point2f(1727,967), Point2f(2008,962+0)};
	
	Mat h = findHomography(srcPts, dstPts);

	cv::Mat distoredImg = cv::imread("front.png");
	imshow("111",distoredImg);
	
	cv::Mat borderDistoredImage;
	cv::Mat undistoredImg;
	cv::Mat perspectiveImg;
	cv::cuda::GpuMat gpuInput;
	cv::cuda::GpuMat gpuOutput;

	cv::Size undistoredImgSize(CAMERA_FRAME_WIDTH*1,CAMERA_FRAME_HEIGHT*1);
	cv::Size imgSize((CAMERA_FRAME_WIDTH+x_expand)*1.0,(CAMERA_FRAME_HEIGHT+y_expand)*1.0);

	copyMakeBorder(distoredImg,borderDistoredImage,(int)(y_expand/2),(int)(y_expand/2),(int)(x_expand/2),(int)(x_expand/2),BORDER_CONSTANT);

	cv::Mat mapx = cv::Mat(imgSize,CV_32FC1);
	cv::Mat mapy = cv::Mat(imgSize,CV_32FC1);
	cv::Mat R = cv::Mat::eye(3,3,CV_32F);
	

	

	double alpha = 1.0; 
	cv::Mat newCamMatrix = getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, Size(CAMERA_FRAME_WIDTH,CAMERA_FRAME_HEIGHT), alpha, undistoredImgSize);
	cv::fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, imgSize, CV_32FC1,mapx,mapy);
	cv::imwrite("borderDistoredImage.png",borderDistoredImage);
	cv::remap(borderDistoredImage,undistoredImg,mapx, mapy, cv::INTER_CUBIC);

	cv::imwrite("undistored.png",undistoredImg);


	gpuInput.upload(undistoredImg);
	cv::cuda::warpPerspective( gpuInput, gpuOutput, h, Size(undistoredImg.size().width+100,undistoredImg.size().height+0));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	gpuOutput.download(perspectiveImg);

	//resize(perspectiveImg,perspectiveImg,Size(1000,1000));

	cv::imwrite("perspectiveImg.png",perspectiveImg);

	const int PERSPECTIVE_IMT_WIDTH = 2130;
	const int PERSPECTIVE_IMT_HEIGHT = 1200;

	const int SURROUND_VIEW_IMG_WIDTH = 2130;
	const int SURROUND_VIEW_IMG_HEIGHT = 2400;

	//pixel on surround view img
	const int FRONT_VIEW_DIST = 500;//in pixel
	const int LEFT_VIEW_DIST = 480;
	const int BACK_VIEW_DIST = 500;
	const int RIGHT_VIEW_DIST = 480;

	const int FRONT_CROPED_START_X = 0;
	const int FRONT_CROPED_START_Y = 0;
	const int RIGHT_CROPED_START_X =1650;
	const int RIGHT_CROPED_START_Y = FRONT_VIEW_DIST;
	const int LEFT_CROPED_START_X = 0;
	const int LEFT_CROPED_START_Y = FRONT_VIEW_DIST;
	const int BACK_CROPED_START_X = 0;
	const int BACK_CROPED_START_Y = 1900;

	const int TOP_MERGE_START_Y = 250;
	const int BOT_MERGE_END_Y = 2200;


	//pixel on perspective img
	const int FRONT_IMG_CROP_START_X = 0;
	const int FRONT_IMG_CROP_START_Y = 500;
	const int FRONT_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH;
	const int FRONT_IMG_CROP_HEIGHT = FRONT_VIEW_DIST;

	const int BACK_IMG_CROP_START_X = 0;
	const int BACK_IMG_CROP_START_Y = 500;
	const int BACK_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH;
	const int BACK_IMG_CROP_HEIGHT = BACK_VIEW_DIST;

	const int RIGHT_IMG_CROP_START_X = 370;
	const int RIGHT_IMG_CROP_START_Y = 430;
	const int RIGHT_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH - RIGHT_CROPED_START_X;
	const int RIGHT_IMG_CROP_HEIGHT = PERSPECTIVE_IMT_HEIGHT * 2 - FRONT_VIEW_DIST - BACK_VIEW_DIST;

	const int LEFT_IMG_CROP_START_X = 370;
	const int LEFT_IMG_CROP_START_Y = 430;
	const int LEFT_IMG_CROP_WIDTH = 490;
	const int LEFT_IMG_CROP_HEIGHT = PERSPECTIVE_IMT_HEIGHT * 2 - FRONT_VIEW_DIST - BACK_VIEW_DIST;

	const int FRONT_RIGHT_MERGE_ROW_DIFF = RIGHT_CROPED_START_Y - RIGHT_IMG_CROP_START_Y;
	const int FRONT_RIGHT_MERGE_COL_DIFF = RIGHT_CROPED_START_X - RIGHT_IMG_CROP_START_X;
	const int FRONT_LEFT_MERGE_ROW_DIFF = LEFT_CROPED_START_Y - LEFT_IMG_CROP_START_Y;
	const int FRONT_LEFT_MERGE_COL_DIFF = LEFT_CROPED_START_X - LEFT_IMG_CROP_START_X;

	cv::Mat front,left,right,back,ret;
	right = cv::imread("rightperspectiveImg.png");
	//ret = cv::Mat(Size(perspectiveImg.size().width, perspectiveImg.size().height*2),CV_8UC3,Scalar(255, 0, 0));
	ret = cv::imread("rett.png");
	
	front = perspectiveImg(Rect(FRONT_IMG_CROP_START_X,FRONT_IMG_CROP_START_Y,FRONT_IMG_CROP_WIDTH,FRONT_IMG_CROP_HEIGHT)).clone();
	back = front.clone();
	flip(back,back,-1);
	transpose(right, right);
	flip(right, right, 1);
	cv::imwrite("rotatedright.png",right);
		cout<<"llllllllllllll:"<<right.size()<<endl;
	cout<<ret.size()<<endl;
	auto rightCroped = right(Rect(RIGHT_IMG_CROP_START_X,RIGHT_IMG_CROP_START_Y,RIGHT_IMG_CROP_WIDTH,RIGHT_IMG_CROP_HEIGHT)).clone();

	cv::imwrite("rightCroped.png",rightCroped);

	rightCroped.copyTo(ret(Rect(RIGHT_CROPED_START_X,RIGHT_CROPED_START_Y,RIGHT_IMG_CROP_WIDTH,RIGHT_IMG_CROP_HEIGHT)));
	
	
	front.copyTo(ret(Rect(FRONT_CROPED_START_X,FRONT_CROPED_START_Y,FRONT_IMG_CROP_WIDTH,FRONT_IMG_CROP_HEIGHT)));
	back.copyTo(ret(Rect(BACK_CROPED_START_X,BACK_CROPED_START_Y,back.cols,back.rows)));

	//printf("251:::::%d,,%d\n",(251-TOP_MERGE_START_Y)/(FRONT_VIEW_DIST-TOP_MERGE_START_Y)*(RIGHT_CROPED_START_X-PERSPECTIVE_IMT_WIDTH)+PERSPECTIVE_IMT_WIDTH,
	//(251-250)*(1650-2130)/(500-250)+2130);

	int mergeColStart;// = PERSPECTIVE_IMT_WIDTH;
	for(int i = TOP_MERGE_START_Y;i<=FRONT_VIEW_DIST;i++)
	//for(int i = TOP_MERGE_START_Y;i<=TOP_MERGE_START_Y+3;i++)
	{
		//mergeColStart = 2130 - (i-250)*48/25;//	front right merge area
		mergeColStart = (i-TOP_MERGE_START_Y)*(RIGHT_CROPED_START_X-PERSPECTIVE_IMT_WIDTH)/(FRONT_VIEW_DIST-TOP_MERGE_START_Y)+PERSPECTIVE_IMT_WIDTH;
		for(;mergeColStart<=SURROUND_VIEW_IMG_WIDTH;mergeColStart++)
		{
			ret.at<Vec3b>(i,mergeColStart)[0] = right.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[0];
			ret.at<Vec3b>(i,mergeColStart)[1] = right.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[1];
			ret.at<Vec3b>(i,mergeColStart)[2] = right.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[2];

			//cout<<"	front right merge area:"<<mergeColStart<<",,";
		}
		//cout<<"i=="<<i<<endl;

		//front left merge area
		int j_limt = (i-TOP_MERGE_START_Y)*(LEFT_IMG_CROP_WIDTH-LEFT_CROPED_START_X)/(FRONT_VIEW_DIST-TOP_MERGE_START_Y)+LEFT_CROPED_START_X;
		for(int j=0;j<=j_limt;j++)
		{
			// ret.at<Vec3b>(i,j)[0] = left.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[0];
			// ret.at<Vec3b>(i,j)[1] = left.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[1];
			// ret.at<Vec3b>(i,j)[2] = left.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[2];
		}
	}

	for(int i=FRONT_VIEW_DIST+LEFT_IMG_CROP_HEIGHT;i<BOT_MERGE_END_Y;i++)
	{
		//back left merge
		int j_limt = (i-BOT_MERGE_END_Y)*(LEFT_IMG_CROP_WIDTH-LEFT_CROPED_START_X)/(FRONT_VIEW_DIST+LEFT_IMG_CROP_HEIGHT-BOT_MERGE_END_Y)+LEFT_CROPED_START_X;
		for(int j=0;j<=j_limt;j++)
		{
			// ret.at<Vec3b>(i,j)[0] = left.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[0];
			// ret.at<Vec3b>(i,j)[1] = left.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[1];
			// ret.at<Vec3b>(i,j)[2] = left.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[2];
		}

		//back right merge
		mergeColStart = (i-BOT_MERGE_END_Y)*(RIGHT_CROPED_START_X-SURROUND_VIEW_IMG_HEIGHT)/(FRONT_VIEW_DIST+LEFT_IMG_CROP_HEIGHT-BOT_MERGE_END_Y)+SURROUND_VIEW_IMG_HEIGHT;
		for(;mergeColStart<=SURROUND_VIEW_IMG_HEIGHT;mergeColStart++)
		{
			ret.at<Vec3b>(i,mergeColStart)[0] = right.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[0];
			ret.at<Vec3b>(i,mergeColStart)[1] = right.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[1];
			ret.at<Vec3b>(i,mergeColStart)[2] = right.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[2];
		}
	}


	cout<<ret.at<Vec3b>(250,2130)<<endl;
	cout<<right.at<Vec3b>(250-70,2130-1280)<<endl;
// imshow("111",ret);
// 	waitKey(0);
	cv::imwrite("ret2.png",ret);


return 0;
	

/*
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

	cv::Mat frames[CAMERA_NUM];
	cv::Mat undistoredFrames[CAMERA_NUM];
	cv::Mat persprctiveImgs[CAMERA_NUM];
	cv::Mat perspectiveHomogtaphy[CAMERA_NUM];
	cv::cuda::GpuMat gpuInput;
	cv::cuda::GpuMat gpuOutput;
	cv::Mat ret;
	cv::Size undistoredImgSize((CAMERA_FRAME_WIDTH+x_expand)*1.5,(CAMERA_FRAME_HEIGHT+y_expand)*1.5);

	//for every vector, 4 points are stored in order of leftbottom, rightbottom, righttop, lefttop
	vector<Point2f> srcPts[CAMERA_NUM] = {
		{Point2f(672,544), Point2f(838,547), Point2f(847,488), Point2f(697,485)},	//front camera
		{Point2f(672,544), Point2f(838,547), Point2f(847,488), Point2f(697,485)},	//left camera
		{Point2f(672,544), Point2f(838,547), Point2f(847,488), Point2f(697,485)},	//back camera
		{Point2f(672,544), Point2f(838,547), Point2f(847,488), Point2f(697,485)},	//right camera
	};

	vector<Point2f> dstPts[CAMERA_NUM] = {
		{Point2f(672,544), Point2f(838,547), Point2f(847,488), Point2f(697,485)},
		{Point2f(672,544), Point2f(838,547), Point2f(847,488), Point2f(697,485)},
		{Point2f(672,544), Point2f(838,547), Point2f(847,488), Point2f(697,485)},
		{Point2f(672,544), Point2f(838,547), Point2f(847,488), Point2f(697,485)},
	};

	for(int i=0;i<CAMERA_NUM;i++)
	{
		perspectiveHomogtaphy[i] = findHomography(srcPts[i], dstPts[i]);
	}


	int frameCnt = 0;
	while(true)
	{
		for(int i=0;i<CAMERA_NUM;i++)
		{
			cameras[i] >> frames[i];
			std::string imageFileName;
        	std::stringstream StrStm;
			StrStm<<"frame"<<i<<".png";
			StrStm>>imageFileName;
			//cv::imwrite(imageFileName,frames[i]);
			//cv::imshow("ret",frames[i]);

			fps = cameras[0].get(cv::CAP_PROP_FPS);
			width = cameras[0].get(cv::CAP_PROP_FRAME_WIDTH);
			height = cameras[0].get(cv::CAP_PROP_FRAME_HEIGHT);
			printf("frameCnt:%d, fps:%f,width:%f,height:%f\n",frameCnt++,fps,width,height);

			cv::Mat borderDistoredImage = frames[i].clone();
			copyMakeBorder(frames[i],borderDistoredImage,(int)(y_expand/2),(int)(y_expand/2),(int)(x_expand/2),(int)(x_expand/2),BORDER_CONSTANT);

			cv::Mat mapx = cv::Mat(undistoredImgSize,CV_32FC1);
			cv::Mat mapy = cv::Mat(undistoredImgSize,CV_32FC1);
			cv::Mat R = cv::Mat::eye(3,3,CV_32F);

			
			cv::fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,intrinsic_matrix,undistoredImgSize,CV_32FC1,mapx,mapy);
			//cv::imwrite("borderDistoredImage"+imageFileName,borderDistoredImage);
			cv::remap(borderDistoredImage,undistoredFrames[i],mapx, mapy, cv::INTER_LINEAR);

			cv::imwrite("undistored"+imageFileName,undistoredFrames[i]);


			gpuInput.upload(undistoredFrames[i]);
			cv::cuda::warpPerspective( gpuInput, gpuOutput, perspectiveHomogtaphy[i], Size(CAMERA_FRAME_WIDTH,CAMERA_FRAME_HEIGHT));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
			gpuOutput.download(persprctiveImgs[i]);

			cv::imwrite("persprctiveImgs"+imageFileName,persprctiveImgs[i]);

			resize(frames[i],frames[i],Size(640,480));
			
		}

		cv::Mat left,right;
		cv::hconcat(frames[0],frames[1],left);
		cv::hconcat(frames[2],frames[3],right);
		cv::vconcat(left,right,ret);
		cv::resize(ret,ret,cv::Size(1024,1024));
		cv::imshow("ret",ret);

		cv::waitKey(333);
	}

	gpuOutput.release();
	gpuInput.release();*/
	
	return 0;
}

