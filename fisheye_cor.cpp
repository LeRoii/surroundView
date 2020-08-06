#include<opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
// #include <opencv2/cudawarping.hpp>

#include <vector>

using std::cout;
using std::endl;
using std::vector;
using namespace cv;
#define PI 3.1415926536
 Point2f getInputPoint(int x, int y,int srcwidth, int srcheight)
 {
    Point2f pfish;
    float theta,phi,r, r2;
    Point3f psph;
    float FOV =(float)PI/180 * 130;
    float FOV2 = (float)PI/180 * 130;
    float width = srcwidth;
    float height = srcheight;

    // Polar angles
    theta = PI * (x / width - 0.5); // -pi/2 to pi/2
    phi = PI * (y / height - 0.5);  // -pi/2 to pi/2

    // Vector in 3D space
    psph.x = cos(phi) * sin(theta);
    psph.y = cos(phi) * cos(theta);
    psph.z = sin(phi) * cos(theta);

    // Calculate fisheye angle and radius
    theta = atan2(psph.z,psph.x);
    phi = atan2(sqrt(psph.x*psph.x+psph.z*psph.z),psph.y);

    r = width * phi / FOV;
    r2 = height * phi / FOV2;

    // Pixel in fisheye space
    pfish.x = 0.5 * width + r * cos(theta);
    pfish.y = 0.5 * height + r2 * sin(theta);
    return pfish;
}

int main()
{
    cv::Matx33d intrinsic_matrix;
	cv::Vec4d distortion_coeffs;

	//right
    intrinsic_matrix << 499.9046978644982, 0, 612.955400120308,
                        0, 500.02613225669, 357.855947068545,
                        0, 0, 1;

    distortion_coeffs << -0.0248636, 0.0124981, -0.0126063, 0.00352282;

    //front
    // intrinsic_matrix << 499.2256225154061, 0, 685.0325527895111,
    //                     0, 499.6344093018186, 288.8632118906361,
    //                     0, 0, 1;

    // distortion_coeffs << -0.0230412, 0.00631978, -0.00455568, 0.000311248;
    cout<<intrinsic_matrix<<endl;   
    cout<<distortion_coeffs<<endl;

    int X_EXPAND = 1200;
    int Y_EXPAND = 350;
    float mmPerPiexl = 2.05f;

	auto newintrinsic_matrix = intrinsic_matrix;
	newintrinsic_matrix(0,0) = intrinsic_matrix(0,0)/2;
	newintrinsic_matrix(1,1) = intrinsic_matrix(1,1)/2;

	cout<<newintrinsic_matrix<<endl;   

    

    cv::Mat distorImg = cv::imread("right.png");
	cv::Size imgSize = distorImg.size();
	float scale = 1.0;
	Size undistorSize = Size(imgSize.width*scale,imgSize.height*scale);
    cv::Mat borderDistoredImage, undistorImg;
    cv::Mat mapx;// = cv::Mat(undistorSize,CV_32FC1);
    cv::Mat mapy;// = cv::Mat(undistorSize,CV_32FC1);
    cv::Mat R = cv::Mat::eye(3,3,CV_32FC1);

	Mat newMatrix = Mat::eye(3,3,CV_32F);
	Mat optMatrix = Mat::eye(3,3,CV_32F);

	auto newmatrix = intrinsic_matrix;

	
	optMatrix = getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, imgSize, 1, undistorSize, 0);
	cout<<"getOptimalNewCameraMatrix"<<endl<<optMatrix<<endl;

	optMatrix.at<double>(0,0) = optMatrix.at<double>(0,0)*0.5f;
    optMatrix.at<double>(1,1) = optMatrix.at<double>(1,1)*0.5f;
	cout<<"getOptimalNewCameraMatrix"<<endl<<optMatrix<<endl;

	fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsic_matrix, distortion_coeffs, imgSize, R, newMatrix, 0.75f, undistorSize, 1.0);
	cout<<"estimateNewCameraMatrixForUndistortRectify"<<endl<<newMatrix<<endl;

	newMatrix.at<float>(0,2) = 1280/2;
    newMatrix.at<float>(1,2) = 720/2;
	// newMatrix.at<float>(0,0) = newMatrix.at<float>(0,0) *0.3f;
    // newMatrix.at<float>(1,1) = newMatrix.at<float>(1,1) *0.3f;

	cout<<"estimateNewCameraMatrixForUndistortRectify"<<endl<<newMatrix<<endl;

    // cv::fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,intrinsic_matrix,imgSize,CV_32FC1,mapx,mapy);
	fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, newMatrix, undistorSize, CV_32FC1, mapx, mapy);

	// fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R,
    //          newintrinsic_matrix, undistorSize, CV_32FC1, mapx, mapy);

	cout<<"initUndistortRectifyMap end"<<endl;


// Mat outImage(distorImg.rows,distorImg.cols,CV_8UC3);
// for(int i=0; i<outImage.cols; i++)
// {
// 	for(int j=0; j<outImage.rows; j++)
// 	{

// 		Point2f inP =  getInputPoint(i,j,distorImg.cols,distorImg.rows);
// 		Point inP2((int)inP.x,(int)inP.y);

// 		if(inP2.x >= distorImg.cols || inP2.y >= distorImg.rows)
// 			continue;

// 		if(inP2.x < 0 || inP2.y < 0)
// 			continue;
// 		Vec3b color = distorImg.at<Vec3b>(inP2);
// 		outImage.at<Vec3b>(Point(i,j)) = color;

// 	}
// }
	// cv::Mat new_intrinsic_mat(intrinsic_matrix);

	// //调节视场大小,乘的系数越小视场越大
	// new_intrinsic_mat.at<double>(0, 0) *= 0.7;
	// new_intrinsic_mat.at<double>(1, 1) *= 0.7;
	// //调节校正图中心，建议置于校正图中心
	// new_intrinsic_mat.at<double>(0, 2) = 0.5 * distorImg.cols;
	// new_intrinsic_mat.at<double>(1, 2) = 0.5 * distorImg.rows;

	// fisheye::undistortImage(distorImg, undistorImg, intrinsic_matrix, distortion_coeffs, new_intrinsic_mat);


    //cv::copyMakeBorder(distorImg,borderDistoredImage,(int)(Y_EXPAND/2),(int)(Y_EXPAND/2),(int)(X_EXPAND/2),(int)(X_EXPAND/2),cv::BORDER_CONSTANT);
    cv::remap(distorImg, undistorImg, mapx, mapy, cv::INTER_CUBIC);

    cv::imwrite("undist.png",undistorImg);

    // //find H

	// //right
	// vector<Point2f> srcPts = {Point2f(313,751), Point2f(533,527), Point2f(1921,722), Point2f(1750,550)};
	// vector<Point2f> dstPts = {Point2f(315,804), Point2f(448,530), Point2f(1918,764), Point2f(1819,555)};
	// Mat hRight = findHomography(srcPts, dstPts);

	// //front
	// srcPts = {Point2f(291,771), Point2f(464, 845), Point2f(620, 468), Point2f(1362,384), Point2f(1652,703), Point2f(1919,568)};
	// dstPts = {Point2f(1015-1560/mmPerPiexl, 771), Point2f(1015-870/mmPerPiexl, 771), Point2f(1015-870/mmPerPiexl, 771-850/mmPerPiexl),
	// 			Point2f(1015+870/mmPerPiexl, 771-860/mmPerPiexl), Point2f(1015+870/mmPerPiexl, 771), Point2f(1015+1560/mmPerPiexl,771)};
	// Mat hFront = findHomography(srcPts, dstPts);

	// //left
	// // srcPts = {Point2f(230,868), Point2f(390,975), Point2f(655,639), Point2f(1376,617), Point2f(1727,967), Point2f(1984,830)};
	// // dstPts = {Point2f(227,984+0), Point2f(390,975), Point2f(389+100,605), Point2f(1727-100,590), Point2f(1727,967), Point2f(2008,962+0)};
	// srcPts = {Point2f(282,785), Point2f(359,831), Point2f(474,515), Point2f(1644,583), Point2f(1768,817), Point2f(1955,741)};
	// dstPts = {Point2f(270,828), Point2f(340,828), Point2f(340,522), Point2f(1780,550), Point2f(1780,828), Point2f(1960,828)};
	
	// Mat hLeft = findHomography(srcPts, dstPts);

	// //back
	// srcPts = {Point2f(357, 805), Point2f(575, 892), Point2f(724, 447), Point2f(1297, 382), Point2f(1494, 894), Point2f(1926, 780)};
	// dstPts = {Point2f(300, 878), Point2f(520,878+300), Point2f(520,450), Point2f(1442, 361), Point2f(1442,878), Point2f(1939, 878)};
	// dstPts = {Point2f(1015-1680/mmPerPiexl, 878), Point2f(1015-800/mmPerPiexl,878), Point2f(1015-800/mmPerPiexl,878-1160/mmPerPiexl), 
	// 		Point2f(1015+800/mmPerPiexl, 878-1380/mmPerPiexl), Point2f(1015+800/mmPerPiexl,878), Point2f(1015+2410/mmPerPiexl, 878)};
	// Mat hBack = findHomography(srcPts, dstPts);


	// srcPts = {Point2f(357, 805), Point2f(724, 447), Point2f(1297, 382), Point2f(1926, 780)};
	// dstPts = {Point2f(260, 898), Point2f(520,450), Point2f(1442, 361), Point2f(1939, 878)};
	// hBack = getPerspectiveTransform(srcPts, dstPts);

	// cv::Mat distoredFront = cv::imread("front.png");
	// cv::Mat distoredLeft = cv::imread("left.png");
	// cv::Mat distoredRight = cv::imread("right.png");
	// cv::Mat distoredBack = cv::imread("back.png");
	// //imshow("111",distoredImg);
	
	// cv::cuda::GpuMat gpuInput;
	// cv::cuda::GpuMat gpuOutput;

	// //cv::Size undistoredImgSize(CAMERA_FRAME_WIDTH*1,CAMERA_FRAME_HEIGHT*1);
	// //cv::Size imgSize((CAMERA_FRAME_WIDTH+X_EXPAND)*1.0,(CAMERA_FRAME_HEIGHT+Y_EXPAND)*1.0);

	// cv::Mat borderDistoredFront, borderDistoredLeft, borderDistoredRight, borderDistoredBack;
	// copyMakeBorder(distoredFront, borderDistoredFront,(int)(Y_EXPAND/2), (int)(Y_EXPAND/2), (int)(X_EXPAND/2), (int)(X_EXPAND/2), BORDER_CONSTANT);
	// copyMakeBorder(distoredLeft, borderDistoredLeft, (int)(Y_EXPAND/2), (int)(Y_EXPAND/2), (int)(X_EXPAND/2), (int)(X_EXPAND/2), BORDER_CONSTANT);
	// copyMakeBorder(distoredRight, borderDistoredRight, (int)(Y_EXPAND/2), (int)(Y_EXPAND/2), (int)(X_EXPAND/2), (int)(X_EXPAND/2), BORDER_CONSTANT);
	// copyMakeBorder(distoredBack, borderDistoredBack, (int)(Y_EXPAND/2), (int)(Y_EXPAND/2), (int)(X_EXPAND/2), (int)(X_EXPAND/2), BORDER_CONSTANT);

	
	// //double alpha = 1.0; 
	// //cv::Mat newCamMatrix = getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, Size(CAMERA_FRAME_WIDTH,CAMERA_FRAME_HEIGHT), alpha, undistoredImgSize);
	// //cv::fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, imgSize, CV_32FC1,mapx,mapy);
	// //cv::imwrite("borderDistoredImage.png",borderDistoredImage);

	// cv::Mat undistoredFront, undistoredLeft, undistoredRight, undistoredBack;
	// cv::remap(borderDistoredFront, undistoredFront, mapx, mapy, cv::INTER_CUBIC);
	// cv::remap(borderDistoredLeft, undistoredLeft, mapx, mapy, cv::INTER_CUBIC);
	// cv::remap(borderDistoredRight, undistoredRight, mapx, mapy, cv::INTER_CUBIC);
	// cv::remap(borderDistoredBack, undistoredBack, mapx, mapy, cv::INTER_CUBIC);

	// cv::imwrite("undistoredFront.png",undistoredFront);
	// cv::imwrite("undistoredLeft.png",undistoredLeft);
	// cv::imwrite("undistoredRight.png",undistoredRight);
	// cv::imwrite("undistoredBack.png",undistoredBack);


	// cv::Mat perspectiveFront, perspectiveLeft, perspectiveRight, perspectiveBack;
	// gpuInput.upload(undistoredFront);
	// cv::cuda::warpPerspective( gpuInput, gpuOutput, hFront, Size(undistoredFront.size().width,undistoredFront.size().height+0));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	// //cv::cuda::warpPerspective( gpuInput, gpuOutput, hFront, Size(1280,1280));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  

	// gpuOutput.download(perspectiveFront);

	// gpuInput.upload(undistoredLeft);
	// cv::cuda::warpPerspective( gpuInput, gpuOutput, hLeft, Size(undistoredFront.size().width,undistoredFront.size().height+0));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	// gpuOutput.download(perspectiveLeft);

	// gpuInput.upload(undistoredRight);
	// cv::cuda::warpPerspective( gpuInput, gpuOutput, hRight, Size(undistoredFront.size().width,undistoredFront.size().height+0));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	// gpuOutput.download(perspectiveRight);

	// gpuInput.upload(undistoredBack);
	// cv::cuda::warpPerspective( gpuInput, gpuOutput, hBack, Size(undistoredFront.size().width,undistoredFront.size().height+0));//, INTER_LINEAR , BORDER_CONSTANT, 0, Stream::Null() );  
	// gpuOutput.download(perspectiveBack);

	// //resize(perspectiveImg,perspectiveImg,Size(1000,1000));

	// cv::imwrite("perspectiveFront.png",perspectiveFront);
	// cv::imwrite("perspectiveLeft.png",perspectiveLeft);
	// cv::imwrite("perspectiveRight.png",perspectiveRight);
	// cv::imwrite("perspectiveBack.png",perspectiveBack);

    // const int CAMERA_FRAME_WIDTH = 1280;
    // const int CAMERA_FRAME_HEIGHT = 720;

	// const int PERSPECTIVE_IMT_WIDTH = 2030;
	// const int PERSPECTIVE_IMT_HEIGHT = 1200;

	// const int SURROUND_VIEW_IMG_WIDTH = 2030;
	// const int SURROUND_VIEW_IMG_HEIGHT = 2400;

	// //pixel on surround view img
	// const int FRONT_VIEW_DIST = 500;//in pixel
	// const int LEFT_VIEW_DIST = 480;
	// const int BACK_VIEW_DIST = 500;
	// const int RIGHT_VIEW_DIST = 480;

	// const int FRONT_CROPED_START_X = 0;
	// const int FRONT_CROPED_START_Y = 0;
	// const int RIGHT_CROPED_START_X =1450;
	// const int RIGHT_CROPED_START_Y = FRONT_VIEW_DIST;
	// const int LEFT_CROPED_START_X = 0;
	// const int LEFT_CROPED_START_Y = FRONT_VIEW_DIST;
	// const int BACK_CROPED_START_X = 0;
	// const int BACK_CROPED_START_Y = 1900;

	// const int TOP_MERGE_START_Y = 140;
	// const int BOT_MERGE_END_Y = 2100;


	// //pixel on perspective img
	// const int FRONT_IMG_CROP_START_X = 0;
	// const int FRONT_IMG_CROP_START_Y = 290;
	// const int FRONT_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH;
	// const int FRONT_IMG_CROP_HEIGHT = FRONT_VIEW_DIST;

	// const int BACK_IMG_CROP_START_X = 0;
	// const int BACK_IMG_CROP_START_Y = 470;
	// const int BACK_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH;
	// const int BACK_IMG_CROP_HEIGHT = BACK_VIEW_DIST;

	// const int RIGHT_IMG_CROP_START_X = 278;
	// const int RIGHT_IMG_CROP_START_Y = 440;
	// const int RIGHT_IMG_CROP_WIDTH = PERSPECTIVE_IMT_WIDTH - RIGHT_CROPED_START_X;
	// const int RIGHT_IMG_CROP_HEIGHT = PERSPECTIVE_IMT_HEIGHT * 2 - FRONT_VIEW_DIST - BACK_VIEW_DIST;

	// const int LEFT_IMG_CROP_START_X = 300;
	// const int LEFT_IMG_CROP_START_Y = 270;
	// const int LEFT_IMG_CROP_WIDTH = 500;
	// const int LEFT_IMG_CROP_HEIGHT = PERSPECTIVE_IMT_HEIGHT * 2 - FRONT_VIEW_DIST - BACK_VIEW_DIST;

	// const int FRONT_RIGHT_MERGE_ROW_DIFF = RIGHT_CROPED_START_Y - RIGHT_IMG_CROP_START_Y;
	// const int FRONT_RIGHT_MERGE_COL_DIFF = RIGHT_CROPED_START_X - RIGHT_IMG_CROP_START_X;
	// const int FRONT_LEFT_MERGE_ROW_DIFF = LEFT_CROPED_START_Y - LEFT_IMG_CROP_START_Y;
	// const int FRONT_LEFT_MERGE_COL_DIFF = LEFT_CROPED_START_X - LEFT_IMG_CROP_START_X;

	// cv::Mat frontCroped, leftCroped, rightCroped, backCroped, ret;
	// //right = cv::imread("rightperspectiveImg.png");
	// ret = cv::Mat(Size(perspectiveFront.size().width, perspectiveFront.size().height*2),CV_8UC3,Scalar(255, 0, 0));
	// //ret = cv::imread("rett.png");
	
	// frontCroped = perspectiveFront(Rect(FRONT_IMG_CROP_START_X, FRONT_IMG_CROP_START_Y, FRONT_IMG_CROP_WIDTH, FRONT_IMG_CROP_HEIGHT)).clone();
	// backCroped = perspectiveBack(Rect(BACK_IMG_CROP_START_X, BACK_IMG_CROP_START_Y, BACK_IMG_CROP_WIDTH, BACK_IMG_CROP_HEIGHT)).clone();

	// flip(backCroped,backCroped,-1);

	// transpose(perspectiveRight, perspectiveRight);
	// flip(perspectiveRight, perspectiveRight, 1);
	// cv::imwrite("rotatedright.png",perspectiveRight);

	// transpose(perspectiveLeft, perspectiveLeft);
	// flip(perspectiveLeft, perspectiveLeft, 0);
	// cv::imwrite("rotatedleft.png",perspectiveLeft);

	// // cout<<"llllllllllllll:"<<right.size()<<endl;
	// // cout<<ret.size()<<endl;
	// rightCroped = perspectiveRight(Rect(RIGHT_IMG_CROP_START_X,RIGHT_IMG_CROP_START_Y,RIGHT_IMG_CROP_WIDTH,RIGHT_IMG_CROP_HEIGHT)).clone();
	// leftCroped = perspectiveLeft(Rect(LEFT_IMG_CROP_START_X, LEFT_IMG_CROP_START_Y, LEFT_IMG_CROP_WIDTH, LEFT_IMG_CROP_HEIGHT)).clone();

	// cv::imwrite("rightCroped.png", rightCroped);
	// cv::imwrite("leftCroped.png", leftCroped);
	// cv::imwrite("frontCroped.png", frontCroped);
	// cv::imwrite("backCroped.png", backCroped);

	// rightCroped.copyTo(ret(Rect(RIGHT_CROPED_START_X, RIGHT_CROPED_START_Y, RIGHT_IMG_CROP_WIDTH,RIGHT_IMG_CROP_HEIGHT)));
	// frontCroped.copyTo(ret(Rect(FRONT_CROPED_START_X, FRONT_CROPED_START_Y, FRONT_IMG_CROP_WIDTH,FRONT_IMG_CROP_HEIGHT)));
	// backCroped.copyTo(ret(Rect(BACK_CROPED_START_X, BACK_CROPED_START_Y, BACK_IMG_CROP_WIDTH, BACK_IMG_CROP_HEIGHT)));
	// leftCroped.copyTo(ret(Rect(LEFT_CROPED_START_X, LEFT_CROPED_START_Y, LEFT_IMG_CROP_WIDTH, LEFT_IMG_CROP_HEIGHT)));

	// printf("251:::::%d,,%d\n",(251-TOP_MERGE_START_Y)/(FRONT_VIEW_DIST-TOP_MERGE_START_Y)*(RIGHT_CROPED_START_X-PERSPECTIVE_IMT_WIDTH)+PERSPECTIVE_IMT_WIDTH,
	// (251-250)*(1650-2130)/(500-250)+2130);

	// int mergeColStart;// = PERSPECTIVE_IMT_WIDTH;
	// for(int i = TOP_MERGE_START_Y;i<=FRONT_VIEW_DIST;i++)
	// //for(int i = TOP_MERGE_START_Y;i<=TOP_MERGE_START_Y+3;i++)
	// {
	// 	//mergeColStart = 2130 - (i-250)*48/25;//	front right merge area
	// 	mergeColStart = (i-TOP_MERGE_START_Y)*(RIGHT_CROPED_START_X-PERSPECTIVE_IMT_WIDTH)/(FRONT_VIEW_DIST-TOP_MERGE_START_Y)+PERSPECTIVE_IMT_WIDTH;
	// 	for(;mergeColStart<=SURROUND_VIEW_IMG_WIDTH;mergeColStart++)
	// 	{
	// 		ret.at<Vec3b>(i,mergeColStart)[0] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[0];
	// 		ret.at<Vec3b>(i,mergeColStart)[1] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[1];
	// 		ret.at<Vec3b>(i,mergeColStart)[2] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[2];

	// 		//cout<<"	front right merge area:"<<mergeColStart<<",,";
	// 	}
	// 	//cout<<"i=="<<i<<endl;

	// 	//front left merge area
	// 	int j_limt = (i-TOP_MERGE_START_Y)*(LEFT_IMG_CROP_WIDTH-LEFT_CROPED_START_X)/(FRONT_VIEW_DIST-TOP_MERGE_START_Y)+LEFT_CROPED_START_X;
	// 	for(int j=0;j<=j_limt;j++)
	// 	{
	// 		ret.at<Vec3b>(i,j)[0] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[0];
	// 		ret.at<Vec3b>(i,j)[1] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[1];
	// 		ret.at<Vec3b>(i,j)[2] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[2];
	// 	}
	// }

	// for(int i=FRONT_VIEW_DIST+LEFT_IMG_CROP_HEIGHT;i<BOT_MERGE_END_Y;i++)
	// {
	// 	//back left merge
	// 	int j_limt = (i-BOT_MERGE_END_Y)*(LEFT_IMG_CROP_WIDTH-LEFT_CROPED_START_X)/(FRONT_VIEW_DIST+LEFT_IMG_CROP_HEIGHT-BOT_MERGE_END_Y)+LEFT_CROPED_START_X;
	// 	for(int j=0;j<=j_limt;j++)
	// 	{
	// 		ret.at<Vec3b>(i,j)[0] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[0];
	// 		ret.at<Vec3b>(i,j)[1] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[1];
	// 		ret.at<Vec3b>(i,j)[2] = perspectiveLeft.at<Vec3b>(i-FRONT_LEFT_MERGE_ROW_DIFF,j-FRONT_LEFT_MERGE_COL_DIFF)[2];
	// 	}

	// 	//back right merge
	// 	mergeColStart = (i-BOT_MERGE_END_Y)*(RIGHT_CROPED_START_X-SURROUND_VIEW_IMG_HEIGHT)/(FRONT_VIEW_DIST+LEFT_IMG_CROP_HEIGHT-BOT_MERGE_END_Y)+SURROUND_VIEW_IMG_HEIGHT;
	// 	for(;mergeColStart<=SURROUND_VIEW_IMG_WIDTH;mergeColStart++)
	// 	{
	// 		ret.at<Vec3b>(i,mergeColStart)[0] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[0];
	// 		ret.at<Vec3b>(i,mergeColStart)[1] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[1];
	// 		ret.at<Vec3b>(i,mergeColStart)[2] = perspectiveRight.at<Vec3b>(i-FRONT_RIGHT_MERGE_ROW_DIFF,mergeColStart-FRONT_RIGHT_MERGE_COL_DIFF)[2];

	// 		//cout<<"	back right merge area:"<<mergeColStart<<",,";
	// 	}
	// }

	// cv::imwrite("ret2.png",ret);


    return 0;


}