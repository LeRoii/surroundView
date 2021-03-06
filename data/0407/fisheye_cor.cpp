#include<opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
// #include <opencv2/cudawarping.hpp>

#include <vector>
#include <string>

using std::cout;
using std::endl;
using std::vector;
using namespace cv;

const int CAMERA_NUM = 4;
const int CAMERA_FRAME_WIDTH = 1280;
const int CAMERA_FRAME_HEIGHT = 720;

int main()
{
    cv::Matx33d intrinsic_matrix[4];
	cv::Vec4d distortion_coeffs[4];

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

	
    // cout<<intrinsic_matrix<<endl;   
    // cout<<distortion_coeffs<<endl;

	// auto newintrinsic_matrix = intrinsic_matrix;
	// newintrinsic_matrix(0,0) = intrinsic_matrix(0,0)/2;
	// newintrinsic_matrix(1,1) = intrinsic_matrix(1,1)/2;

	// cout<<newintrinsic_matrix<<endl;

	std::string foldername = "/home/zpwang/panaimg/";

	cv::Mat distoredFront = cv::imread(foldername + "front.png");
	cv::Mat distoredLeft = cv::imread(foldername + "left.png");
	cv::Mat distoredBack = cv::imread(foldername + "back.png");
	cv::Mat distoredRight = cv::imread(foldername + "right.png");

	cv::Mat undistoredFront, undistoredLeft, undistoredRight, undistoredBack;

	cv::Size imgSize = distoredFront.size();
	float scale = 1.0;
	Size undistorSize = Size(imgSize.width*scale,imgSize.height*scale);
	scale = 1.0;
	Size perspectiveSize = Size(undistorSize.width*scale,undistorSize.height*scale);
    cv::Mat undistorImg;
    cv::Mat mapx[CAMERA_NUM];// = cv::Mat(undistorSize,CV_32FC1);
    cv::Mat mapy[CAMERA_NUM];// = cv::Mat(undistorSize,CV_32FC1);
    cv::Mat R = cv::Mat::eye(3,3,CV_32FC1);

	// Mat newMatrix = Mat::eye(3,3,CV_32F);
	// Mat optMatrix = Mat::eye(3,3,CV_32F);

	cv::Mat newMatrix[CAMERA_NUM];

	for(int i=0;i<CAMERA_NUM;i++)
	{
		newMatrix[i] = cv::Mat::eye(3,3,CV_32F);
		mapx[i] = cv::Mat(undistorSize,CV_32FC1);
		mapy[i] = cv::Mat(undistorSize,CV_32FC1);
	}

	newMatrix[0] = (cv::Mat_<float>(3,3)<< 136.20689, 0, 869, 0, 158.6207, 363, 0, 0, 1);//front
	newMatrix[1] = (cv::Mat_<float>(3,3)<< 184.48276, 0, 692, 0, 194.82761, 360, 0, 0, 1);//left
	newMatrix[2] = (cv::Mat_<float>(3,3)<< 272.40869, 0, 640, 0, 209.15546, 360, 0, 0, 1);//back
	newMatrix[3] = (cv::Mat_<float>(3,3)<< 165.0714, 0, 737, 0, 183.35963, 451, 0, 0, 1);//right
	Mat newMatrixxxx = Mat::eye(3,3,CV_32F);
	// auto newmatrix = intrinsic_matrix;

	for(int i=0;i<4;i++)
	{
		// cout<<"camera:"<<i<<endl<<intrinsic_matrix[i]<<endl<<distortion_coeffs[i]<<endl<<endl;
		fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsic_matrix[i], distortion_coeffs[i], imgSize, R, newMatrix[i], 0.9f, undistorSize, 1.0);
		newMatrix[i].at<float>(0,2) = CAMERA_FRAME_WIDTH/2;
    	newMatrix[i].at<float>(1,2) = CAMERA_FRAME_HEIGHT/2;
		cout<<"estimateNewCameraMatrixForUndistortRectify"<<endl<<newMatrix[i]<<endl<<endl;
		// cv::fisheye::initUndistortRectifyMap(intrinsic_matrix[i], distortion_coeffs[i], R, newMatrix[i], imgSize, CV_32FC1, mapx[i], mapy[i]);
		cv::fisheye::initUndistortRectifyMap(intrinsic_matrix[i], distortion_coeffs[i], R, newMatrix[i], undistorSize, CV_32FC1, mapx[i], mapy[i]);
	}

	// newMatrix[3].at<float>(0,2) = CAMERA_FRAME_WIDTH/2 - 150;
	// newMatrix[3].at<float>(1,2) = CAMERA_FRAME_HEIGHT/2;
	// cv::fisheye::initUndistortRectifyMap(intrinsic_matrix[3], distortion_coeffs[3], R, newMatrix[3], imgSize, CV_32FC1, mapx[3], mapy[3]);

	cv::remap(distoredFront, undistoredFront, mapx[0], mapy[0], cv::INTER_CUBIC);
	cv::remap(distoredLeft, undistoredLeft, mapx[1], mapy[1], cv::INTER_CUBIC);
	cv::remap(distoredBack, undistoredBack, mapx[2], mapy[2], cv::INTER_CUBIC);
	cv::remap(distoredRight, undistoredRight, mapx[3], mapy[3], cv::INTER_CUBIC);

	cv::imwrite("undistoredFront.png",undistoredFront);
	cv::imwrite("undistoredLeft.png",undistoredLeft);
	cv::imwrite("undistoredRight.png",undistoredRight);
	cv::imwrite("undistoredBack.png",undistoredBack);

	// //find H

	//right
	vector<Point2f> srcPts = {cv::Point2f(289,516), cv::Point2f(392, 392), cv::Point2f(1038,383), cv::Point2f(1134,448)};
	vector<Point2f> dstPts = {cv::Point2f(283,563), cv::Point2f(283, 364), cv::Point2f(1132,399), cv::Point2f(1150,566)};
	Mat hRight = findHomography(srcPts, dstPts);
	

	//front
	srcPts = {cv::Point2f(202,520), cv::Point2f(333, 445), cv::Point2f(843,444), cv::Point2f(908,502)};
	dstPts = {cv::Point2f(156,636), cv::Point2f(154, 559), cv::Point2f(962, 561), cv::Point2f(962,627)};
	Mat hFront = findHomography(srcPts, dstPts);
	// Mat hFront = (cv::Mat_<float>(3,3)<<0.1011, 0.5896, -772.2538, -0.0708, 1.5019, -1759.8974, -0.0000, 0.0009, -1.0000);
	// Mat hFront = (cv::Mat_<double>(3,3)<<0.4048, 2.7428, -1023.2658, -0.0322, 3.0527, -895.9248, -0.0000, 0.0039, -1.0000);

	cout<<"hFront:"<<hFront<<endl;

	//left
	srcPts = {cv::Point2f(217,456), cv::Point2f(295, 406), cv::Point2f(870,395), cv::Point2f(940,460)};
	dstPts = {cv::Point2f(224,507), cv::Point2f(223, 378), cv::Point2f(973,395), cv::Point2f(973,505)};
	
	Mat hLeft = findHomography(srcPts, dstPts);

	//back
	srcPts = {cv::Point2f(187,484), cv::Point2f(407, 356), cv::Point2f(796,372), cv::Point2f(926,500)};
	dstPts = {cv::Point2f(246,546), cv::Point2f(242, 347), cv::Point2f(905,349), cv::Point2f(904,541)};
	Mat hBack = findHomography(srcPts, dstPts);

	cv::Mat perspectiveFront, perspectiveLeft, perspectiveRight, perspectiveBack;
	// cv::cuda::GpuMat gpuInput;
	// cv::cuda::GpuMat gpuOutput;

	// gpuInput.upload(undistoredFront);
	// cv::cuda::warpPerspective( gpuInput, gpuOutput, hFront, perspectiveSize, INTER_CUBIC);//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	// //cv::cuda::warpPerspective( gpuInput, gpuOutput, hFront, Size(1280,1280));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  

	// gpuOutput.download(perspectiveFront);

	cv::warpPerspective(undistoredFront,perspectiveFront, hFront, perspectiveSize, INTER_CUBIC);

	// gpuInput.upload(undistoredLeft);
	// cv::cuda::warpPerspective( gpuInput, gpuOutput, hLeft, perspectiveSize, INTER_CUBIC);//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	// gpuOutput.download(perspectiveLeft);

	cv::warpPerspective(undistoredLeft,perspectiveLeft, hLeft, perspectiveSize, INTER_CUBIC);


	// gpuInput.upload(undistoredRight);
	// cv::cuda::warpPerspective( gpuInput, gpuOutput, hRight, perspectiveSize, INTER_CUBIC);//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	// gpuOutput.download(perspectiveRight);

	cv::warpPerspective(undistoredRight,perspectiveRight, hRight, perspectiveSize, INTER_CUBIC);


	// gpuInput.upload(undistoredBack);
	// cv::cuda::warpPerspective( gpuInput, gpuOutput, hBack, perspectiveSize, INTER_CUBIC);//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	// gpuOutput.download(perspectiveBack);

	cv::warpPerspective(undistoredBack,perspectiveBack, hBack, perspectiveSize, INTER_CUBIC);


	resize(perspectiveFront,perspectiveFront,Size(1280,720));
	resize(perspectiveLeft,perspectiveLeft,Size(1280,720));
	resize(perspectiveRight,perspectiveRight,Size(1280,720));
	resize(perspectiveBack,perspectiveBack,Size(1280,720));

	cv::imwrite("perspectiveFront.png",perspectiveFront);
	cv::imwrite("perspectiveLeft.png",perspectiveLeft);
	cv::imwrite("perspectiveRight.png",perspectiveRight);
	cv::imwrite("perspectiveBack.png",perspectiveBack);

	cv::Mat convertedLeft = perspectiveLeft;
	cv::Mat convertedRight = perspectiveRight;
	// perspectiveLeft.convertTo(convertedLeft, perspectiveLeft.type(), 0.74, -14);
	// perspectiveRight.convertTo(convertedRight, perspectiveLeft.type(), 0.63, 20);
	// perspectiveBack.convertTo(perspectiveBack, perspectiveLeft.type(), 0.68, -3);

	cv::imwrite("perspectiveLeftconverted.png",convertedLeft);


	const int SURROUND_VIEW_IMG_WIDTH = 1500;
	const int SURROUND_VIEW_IMG_HEIGHT = 1500;

	const int PERSPECTIVE_IMT_WIDTH = 1280;
	const int PERSPECTIVE_IMT_HEIGHT = 720;

	const int CAR_IMG_WIDTH = 530;
	const int CAR_IMG_HEIGHT = 550;
	const int CAR_IMG_START_X = 440;//SURROUND_VIEW_IMG_WIDTH/2 - CAR_IMG_WIDTH/2;
	const int CAR_IMG_START_Y = 498;//SURROUND_VIEW_IMG_HEIGHT/2 - CAR_IMG_HEIGHT/2;

	//pixel on surround view img
	const int FRONT_VIEW_DIST = 500;//in pixel

	const int FRONT_CROPED_START_X = 0;
	const int FRONT_CROPED_START_Y = 0;
	const int RIGHT_CROPED_START_X = 900;//SURROUND_VIEW_IMG_WIDTH/2 + CAR_IMG_WIDTH/2;
	const int RIGHT_CROPED_START_Y = 340;
	const int LEFT_CROPED_START_X = 0;
	const int LEFT_CROPED_START_Y = 340;
	const int BACK_CROPED_START_X = 0;
	const int BACK_CROPED_START_Y = FRONT_VIEW_DIST + CAR_IMG_HEIGHT;

	const int BACK_VIEW_DIST = 500;

	//pixel on perspective img
	const int FRONT_IMG_CROP_START_X = 0;
	const int FRONT_IMG_CROP_START_Y = 630 - FRONT_VIEW_DIST;
	const int FRONT_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH;
	const int FRONT_IMG_CROP_HEIGHT = FRONT_VIEW_DIST;

	const int BACK_IMG_CROP_HEIGHT = BACK_VIEW_DIST;
	const int BACK_IMG_CROP_START_X = BACK_CROPED_START_X;
	const int BACK_IMG_CROP_START_Y = 540 - BACK_VIEW_DIST;
	const int BACK_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH - BACK_CROPED_START_X;

	const int RIGHT_IMG_CROP_START_X = 142;
	const int RIGHT_IMG_CROP_START_Y = 0;
	const int RIGHT_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH - RIGHT_CROPED_START_X;
	const int RIGHT_IMG_CROP_HEIGHT = 1280;
	
	const int LEFT_IMG_CROP_WIDTH = 370;//(SURROUND_VIEW_IMG_WIDTH - CAR_IMG_WIDTH)/2;
	const int LEFT_IMG_CROP_HEIGHT = RIGHT_IMG_CROP_HEIGHT;
	const int LEFT_IMG_CROP_START_X = 540 - LEFT_IMG_CROP_WIDTH;//right edge on rotated left img
	const int LEFT_IMG_CROP_START_Y = 132;

	const int FRONT_RIGHT_MERGE_ROW_DIFF = RIGHT_CROPED_START_Y - RIGHT_IMG_CROP_START_Y;
	const int FRONT_RIGHT_MERGE_COL_DIFF = RIGHT_CROPED_START_X - RIGHT_IMG_CROP_START_X;
	const int FRONT_LEFT_MERGE_ROW_DIFF = LEFT_CROPED_START_Y - LEFT_IMG_CROP_START_Y;
	const int FRONT_LEFT_MERGE_COL_DIFF = LEFT_CROPED_START_X - LEFT_IMG_CROP_START_X;

	const int TOP_MERGE_START_Y = 330;//280;
	const int BOT_MERGE_END_Y = 1066;//1322;

	cv::Mat frontCroped, leftCroped, rightCroped, backCroped, ret;
	ret = cv::Mat(Size(SURROUND_VIEW_IMG_WIDTH, SURROUND_VIEW_IMG_HEIGHT),CV_8UC3,Scalar(255, 255, 255));

	cv::Mat carImg = cv::imread("car.png");
	cv::resize(carImg, carImg, cv::Size(CAR_IMG_WIDTH, CAR_IMG_HEIGHT));
	
	frontCroped = perspectiveFront(Rect(FRONT_IMG_CROP_START_X, FRONT_IMG_CROP_START_Y, FRONT_IMG_CROP_WIDTH, FRONT_IMG_CROP_HEIGHT)).clone();
	backCroped = perspectiveBack(Rect(BACK_IMG_CROP_START_X, BACK_IMG_CROP_START_Y, BACK_IMG_CROP_WIDTH, BACK_IMG_CROP_HEIGHT)).clone();
	
	flip(backCroped,backCroped,-1);

	transpose(convertedRight, convertedRight);
	flip(convertedRight, convertedRight, 1);
	
	transpose(convertedLeft, convertedLeft);
	flip(convertedLeft, convertedLeft, 0);

	rightCroped = convertedRight(Rect(RIGHT_IMG_CROP_START_X, RIGHT_IMG_CROP_START_Y, RIGHT_IMG_CROP_WIDTH, RIGHT_IMG_CROP_HEIGHT)).clone();
	leftCroped = convertedLeft(Rect(LEFT_IMG_CROP_START_X, LEFT_IMG_CROP_START_Y, LEFT_IMG_CROP_WIDTH, LEFT_IMG_CROP_HEIGHT)).clone();

	cv::imwrite("cropFront.png",frontCroped);
	cv::imwrite("cropback.png",backCroped);
	cv::imwrite("rotatedright.png",convertedRight);
	cv::imwrite("rotatedleft.png",convertedLeft);
	cv::imwrite("cropright.png",rightCroped);
	cv::imwrite("cropleft.png",leftCroped);

	leftCroped.copyTo(ret(Rect(LEFT_CROPED_START_X, LEFT_CROPED_START_Y, LEFT_IMG_CROP_WIDTH, LEFT_IMG_CROP_HEIGHT)));
	rightCroped.copyTo(ret(Rect(RIGHT_CROPED_START_X, RIGHT_CROPED_START_Y, RIGHT_IMG_CROP_WIDTH,RIGHT_IMG_CROP_HEIGHT)));
	frontCroped.copyTo(ret(Rect(FRONT_CROPED_START_X, FRONT_CROPED_START_Y, FRONT_IMG_CROP_WIDTH, FRONT_IMG_CROP_HEIGHT)));
	backCroped.copyTo(ret(Rect(BACK_CROPED_START_X, BACK_CROPED_START_Y, BACK_IMG_CROP_WIDTH, BACK_IMG_CROP_HEIGHT)));

		// carImg.copyTo(ret(Rect(CAR_IMG_START_X, CAR_IMG_START_Y, CAR_IMG_WIDTH, CAR_IMG_HEIGHT+80)));

	// carImg.copyTo(ret(Rect(CAR_IMG_START_X, CAR_IMG_START_Y, CAR_IMG_WIDTH, CAR_IMG_HEIGHT)));

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
			ret.at<Vec3b>(i,mergeColStart)[0] = convertedRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[0];
			ret.at<Vec3b>(i,mergeColStart)[1] = convertedRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[1];
			ret.at<Vec3b>(i,mergeColStart)[2] = convertedRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[2];

			// cout<<"	front right merge area:"<<mergeColStart<<",,";
		}
	}

	// front left merge
	x1 = LEFT_CROPED_START_X;
	y1 = TOP_MERGE_START_Y;
	x2 = LEFT_IMG_CROP_WIDTH;
	y2 = FRONT_VIEW_DIST;

	for(int i = y1;i<=y2;i++)
	{
		int j_limt = (i-y1)*(x2-x1)/(y2-y1)+x1;
		for(int j=0;j<=j_limt;j++)
		{
			ret.at<Vec3b>(i,j)[0] = convertedLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[0];
			ret.at<Vec3b>(i,j)[1] = convertedLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[1];
			ret.at<Vec3b>(i,j)[2] = convertedLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[2];
		}
	}

	//back left merge
	x1 = LEFT_CROPED_START_X;
	y1 = BOT_MERGE_END_Y;
	x2 = LEFT_IMG_CROP_WIDTH;
	y2 = BACK_CROPED_START_Y;

	for(int i=y2;i<y1;i++)
	{
		int j_limt = (i-y1)*(x2-x1)/(y2-y1)+x1;
		for(int j=0;j<=j_limt;j++)
		{
			ret.at<Vec3b>(i,j)[0] = convertedLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[0];
			ret.at<Vec3b>(i,j)[1] = convertedLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[1];
			ret.at<Vec3b>(i,j)[2] = convertedLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[2];
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
			ret.at<Vec3b>(i,mergeColStart)[0] = convertedRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[0];
			ret.at<Vec3b>(i,mergeColStart)[1] = convertedRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[1];
			ret.at<Vec3b>(i,mergeColStart)[2] = convertedRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[2];

			// cout<<"	back right merge area:"<<mergeColStart<<",,";
		}
	}

	// // resize(ret,ret,Size(1000,1000));
	// // ret = ret(Rect(0,0,SURROUND_VIEW_IMG_WIDTH, 1450));
	// // cv::Mat tmp;
	// // // cv::blur(ret(Rect(0, 300, 300, 140)), tmp, cv::Size(20, 20));

	// // cv::GaussianBlur(ret(Rect(0, 300, 300, 140)), tmp, cv::Size(19, 19), 15);

	// // tmp.copyTo(ret(Rect(0, 300, 300, 140)));

	// // cv::imwrite("tmp.png",tmp);

	// ret = ret(Rect(10,0,SURROUND_VIEW_IMG_WIDTH-10, 1590));
	// resize(ret,ret,Size(600,1100));


	cv::imwrite("ret.png",ret);

	return 0;

}