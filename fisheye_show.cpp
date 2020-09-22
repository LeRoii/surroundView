#include<opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using namespace cv;

const int CAMERA_FRAME_WIDTH = 1280;
const int CAMERA_FRAME_HEIGHT = 720;

int main()
{
    cv::Matx33d intrinsic_matrix;
	cv::Vec4d distortion_coeffs;


    
    //front
    intrinsic_matrix << 499.2256225154061, 0, 685.0325527895111,
                        0, 499.6344093018186, 288.8632118906361,
                        0, 0, 1;

    distortion_coeffs << -0.0230412, 0.00631978, -0.00455568, 0.000311248;

    //left
    // intrinsic_matrix << 512.0799991633208, 0, 681.3682183385124,
    //                     0, 511.931977341321, 348.725565495493,
    //                     0, 0, 1;

    // distortion_coeffs << -0.0309463, 0.00392602, -0.00515291, 0.00102781;

    //back
    // intrinsic_matrix << 500.8548704340391, 0, 644.1812130625166,
    //                     0, 499.9234264350891, 391.6005802176933,
    //                     0, 0, 1;

    // distortion_coeffs << -0.0136425, -0.0220779, 0.0208222, -0.00740363;

    //right
    // intrinsic_matrix << 499.9046978644982, 0, 612.955400120308,
    //                     0, 500.02613225669, 357.855947068545,
    //                     0, 0, 1;

    // distortion_coeffs << -0.0248636, 0.0124981, -0.0126063, 0.00352282;



    cout<<intrinsic_matrix<<endl;   
    cout<<distortion_coeffs<<endl;

    cv::Size imgSize = Size(CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT);
	float scale = 1.0;
	Size undistorSize = Size(imgSize.width*scale,imgSize.height*scale);
    
    cv::Mat mapx = cv::Mat(undistorSize,CV_32FC1);
    cv::Mat mapy = cv::Mat(undistorSize,CV_32FC1);
    cv::Mat R = cv::Mat::eye(3,3,CV_32FC1);

	Mat newMatrix = Mat::eye(3,3,CV_32F);
	Mat optMatrix = Mat::eye(3,3,CV_32F);

    auto scaled_matrix = intrinsic_matrix;
    // scaled_matrix(0,0) = scaled_matrix(0,0)*


	fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsic_matrix, distortion_coeffs, imgSize, R, newMatrix, 0.75f, undistorSize, 1.0);
	cout<<"estimateNewCameraMatrixForUndistortRectify"<<endl<<newMatrix<<endl;

    newMatrix.at<float>(0,2) = 1280/2;
    newMatrix.at<float>(1,2) = 720/2;

	// newMatrix.at<float>(0,0) = newMatrix.at<float>(0,0) *0.3f;
    // newMatrix.at<float>(1,1) = newMatrix.at<float>(1,1) *0.3f;

	cout<<"estimateNewCameraMatrixForUndistortRectify"<<endl<<newMatrix<<endl;

	fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, newMatrix, undistorSize, CV_32FC1, mapx, mapy);


	cout<<"initUndistortRectifyMap end"<<endl;

    cv::VideoCapture camera;

    camera.open(0);
    if(!camera.isOpened())
    {
        printf("cameraopen failed\n");
        return 0;
    }
    camera.set(cv::CAP_PROP_FRAME_WIDTH,CAMERA_FRAME_WIDTH);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT,CAMERA_FRAME_HEIGHT);

    cv::Mat frame;
    cv::Mat undistorImg;

    while(1)
    {
        camera >> frame;
        cv::remap(frame, undistorImg, mapx, mapy, cv::INTER_CUBIC);

        cv::imshow("distored", frame);
        cv::imshow("undistorImg", undistorImg);
        cv::waitKey(1);
    }
    return 0 ;
}

