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

	std::string foldername = "./bk/0923/";

	cv::Mat distoredFront = cv::imread(foldername + "front.png");
	cv::Mat distoredLeft = cv::imread(foldername + "left.png");
	cv::Mat distoredBack = cv::imread(foldername + "back.png");
	cv::Mat distoredRight = cv::imread(foldername + "right.png");

	cv::Mat undistoredFront, undistoredLeft, undistoredRight, undistoredBack;

	cv::Size imgSize = distoredFront.size();
	float scale = 1.0;
	Size undistorSize = Size(imgSize.width*scale,imgSize.height*scale);
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
		// fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsic_matrix[i], distortion_coeffs[i], imgSize, R, newMatrix[i], 0.75f, undistorSize, 1.0);
		// newMatrix[i].at<float>(0,2) = CAMERA_FRAME_WIDTH/2;
    	// newMatrix[i].at<float>(1,2) = CAMERA_FRAME_HEIGHT/2;
		cout<<"estimateNewCameraMatrixForUndistortRectify"<<endl<<newMatrix[i]<<endl<<endl;
		cv::fisheye::initUndistortRectifyMap(intrinsic_matrix[i], distortion_coeffs[i], R, newMatrix[i], imgSize, CV_32FC1, mapx[i], mapy[i]);
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
	vector<Point2f> srcPts = {cv::Point2f(434,506), cv::Point2f(532, 442), cv::Point2f(939,459), cv::Point2f(1106,618)};
	vector<Point2f> dstPts = {cv::Point2f(446,591), cv::Point2f(445, 458), cv::Point2f(1055,468), cv::Point2f(1057,638)};
	Mat hRight = findHomography(srcPts, dstPts);
	

	//front
	srcPts = {cv::Point2f(102,550), cv::Point2f(652, 342), cv::Point2f(1038,354), cv::Point2f(1237,466)};
	dstPts = {cv::Point2f(285,657), cv::Point2f(282, 361), cv::Point2f(1110, 402), cv::Point2f(1111,619)};
	Mat hFront = findHomography(srcPts, dstPts);
	// Mat hFront = (cv::Mat_<float>(3,3)<<0.1011, 0.5896, -772.2538, -0.0708, 1.5019, -1759.8974, -0.0000, 0.0009, -1.0000);
	// Mat hFront = (cv::Mat_<double>(3,3)<<0.4048, 2.7428, -1023.2658, -0.0322, 3.0527, -895.9248, -0.0000, 0.0039, -1.0000);

	cout<<"hFront:"<<hFront<<endl;

	//left
	srcPts = {cv::Point2f(399,579), cv::Point2f(509, 494), cv::Point2f(938,495), cv::Point2f(1025,558)};
	dstPts = {cv::Point2f(317,634), cv::Point2f(314, 510), cv::Point2f(1022,502), cv::Point2f(1022,608)};
	
	Mat hLeft = findHomography(srcPts, dstPts);

	//back
	srcPts = {cv::Point2f(174,422), cv::Point2f(523, 278), cv::Point2f(871,302), cv::Point2f(1177,399)};
	dstPts = {cv::Point2f(342,639), cv::Point2f(348, 266), cv::Point2f(1083,432), cv::Point2f(1082,615)};
	Mat hBack = findHomography(srcPts, dstPts);

	cv::Mat perspectiveFront, perspectiveLeft, perspectiveRight, perspectiveBack;
	cv::cuda::GpuMat gpuInput;
	cv::cuda::GpuMat gpuOutput;

	gpuInput.upload(undistoredFront);
	cv::cuda::warpPerspective( gpuInput, gpuOutput, hFront, Size(undistoredFront.size().width,undistoredFront.size().height+0), INTER_CUBIC);//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	//cv::cuda::warpPerspective( gpuInput, gpuOutput, hFront, Size(1280,1280));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  

	gpuOutput.download(perspectiveFront);

	gpuInput.upload(undistoredLeft);
	cv::cuda::warpPerspective( gpuInput, gpuOutput, hLeft, Size(undistoredFront.size().width,undistoredFront.size().height+0), INTER_CUBIC);//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	gpuOutput.download(perspectiveLeft);

	gpuInput.upload(undistoredRight);
	cv::cuda::warpPerspective( gpuInput, gpuOutput, hRight, Size(undistoredFront.size().width,undistoredFront.size().height+0), INTER_CUBIC);//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	gpuOutput.download(perspectiveRight);

	gpuInput.upload(undistoredBack);
	cv::cuda::warpPerspective( gpuInput, gpuOutput, hBack, Size(undistoredFront.size().width,undistoredFront.size().height+0), INTER_CUBIC);//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	gpuOutput.download(perspectiveBack);

	//resize(perspectiveImg,perspectiveImg,Size(1000,1000));

	cv::imwrite("perspectiveFront.png",perspectiveFront);
	cv::imwrite("perspectiveLeft.png",perspectiveLeft);
	cv::imwrite("perspectiveRight.png",perspectiveRight);
	cv::imwrite("perspectiveBack.png",perspectiveBack);

	cv::Mat convertedLeft;
	perspectiveLeft.convertTo(convertedLeft, perspectiveLeft.type(), 0.78, 55);

	cv::imwrite("perspectiveLeftconverted.png",convertedLeft);


	const int SURROUND_VIEW_IMG_WIDTH = 1280;
	const int SURROUND_VIEW_IMG_HEIGHT = 1925;

	const int PERSPECTIVE_IMT_WIDTH = 1280;
	const int PERSPECTIVE_IMT_HEIGHT = 720;

	const int CAR_IMG_WIDTH = 640;
	const int CAR_IMG_HEIGHT = 800;
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

	cv::Mat frontCroped, leftCroped, rightCroped, backCroped, ret;
	ret = cv::Mat(Size(SURROUND_VIEW_IMG_WIDTH, SURROUND_VIEW_IMG_HEIGHT),CV_8UC3,Scalar(255, 255, 255));

	cv::Mat carImg = cv::imread("car.png");
	cv::resize(carImg, carImg, cv::Size(CAR_IMG_WIDTH,CAR_IMG_HEIGHT+80));
	
	frontCroped = perspectiveFront(Rect(FRONT_IMG_CROP_START_X, FRONT_IMG_CROP_START_Y, FRONT_IMG_CROP_WIDTH, FRONT_IMG_CROP_HEIGHT)).clone();
	backCroped = perspectiveBack(Rect(BACK_IMG_CROP_START_X, BACK_IMG_CROP_START_Y, BACK_IMG_CROP_WIDTH, BACK_IMG_CROP_HEIGHT)).clone();
	
	flip(backCroped,backCroped,-1);

	transpose(perspectiveRight, perspectiveRight);
	flip(perspectiveRight, perspectiveRight, 1);
	
	transpose(convertedLeft, convertedLeft);
	flip(convertedLeft, convertedLeft, 0);

	rightCroped = perspectiveRight(Rect(RIGHT_IMG_CROP_START_X, RIGHT_IMG_CROP_START_Y, RIGHT_IMG_CROP_WIDTH, RIGHT_IMG_CROP_HEIGHT)).clone();
	leftCroped = convertedLeft(Rect(LEFT_IMG_CROP_START_X, LEFT_IMG_CROP_START_Y, LEFT_IMG_CROP_WIDTH, LEFT_IMG_CROP_HEIGHT)).clone();

	cv::imwrite("cropFront.png",frontCroped);
	cv::imwrite("cropback.png",backCroped);
	cv::imwrite("rotatedright.png",perspectiveRight);
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
			ret.at<Vec3b>(i,mergeColStart)[0] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[0];
			ret.at<Vec3b>(i,mergeColStart)[1] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[1];
			ret.at<Vec3b>(i,mergeColStart)[2] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[2];

			// cout<<"	front right merge area:"<<mergeColStart<<",,";
		}
	}

	// front left merge
	x1 = LEFT_CROPED_START_X;
	y1 = TOP_MERGE_START_Y-50;
	x2 = LEFT_IMG_CROP_WIDTH-65;
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
	x1 = LEFT_CROPED_START_X+208;
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
			ret.at<Vec3b>(i,mergeColStart)[0] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[0];
			ret.at<Vec3b>(i,mergeColStart)[1] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[1];
			ret.at<Vec3b>(i,mergeColStart)[2] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[2];

			// cout<<"	back right merge area:"<<mergeColStart<<",,";
		}
	}

	// resize(ret,ret,Size(1000,1000));
	cv::imwrite("ret.png",ret);

	return 0;

}