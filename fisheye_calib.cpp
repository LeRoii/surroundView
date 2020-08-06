#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include <fstream>
using namespace std;
using namespace cv;

int main()
{
    ofstream fout("caliberation_result.txt");  /**    保存定标结果的文件     **/

    /************************************************************************  
           读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化  
    *************************************************************************/   
    cout<<"start extract corners"<<endl; 
	//以下三行为需要手动修改的参数
    int image_count=  57;                   //图像数量
    Size board_size = Size(6,4);            //定标板上每行、列的角点数
	int x_expand = 0,y_expand = 0;		//x,y方向的扩展(x横向，y纵向)，适当增大可以不损失原图像信息

    vector<Point2f> corners;                //缓存每幅图像上检测到的角点
    vector<vector<Point2f>>  corners_Seq;   //保存检测到的所有角点/   
    vector<Mat>  image_Seq;
	int successImageNum = 0;				//成功提取角点的棋盘图数量	
	bool conner_flag = true;				//所有图像角点提取成功为true，其余为false
    for( int i = 0;  i != image_count ; i++)
    {
        cout<<"img"<<i+1<<"..."<<endl;
        string imageFileName;
        std::stringstream StrStm;
        StrStm<<i+1;
        StrStm>>imageFileName;
        imageFileName += ".png";
        cv::Mat image = imread("../../calib1280_left/img"+imageFileName); 
		//Mat image;//边界扩展后的图片
		//copyMakeBorder(imageSrc,image,(int)(y_expand/2),(int)(y_expand/2),(int)(x_expand/2),(int)(x_expand/2),BORDER_CONSTANT);
        /* 提取角点 */   
        Mat imageGray;
        cvtColor(image, imageGray , CV_RGB2GRAY);
        bool patternfound = findChessboardCorners(image, board_size, corners,CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+ 
            CALIB_CB_FAST_CHECK );
        if (!patternfound)   
        {   
			cout<<"img"<<i+1<<"corner extract failed"<<endl;  
            conner_flag = false;
			break;
        } 
        else
        {   
            /* 亚像素精确化 */
            cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
            /* 绘制检测到的角点并保存 */
            Mat imageTemp = image.clone();
            for (int j = 0; j < corners.size(); j++)
            {
                circle( imageTemp, corners[j], 10, Scalar(0,0,255), 2, 8, 0);
            }
            string imageFileName;
            std::stringstream StrStm;
            StrStm<<i+1;
            StrStm>>imageFileName;
            imageFileName += "_corner.jpg";
            imwrite(imageFileName,imageTemp);
            cout<<"img"<<i+1<<"corner extract completed"<<endl;

			successImageNum = successImageNum + 1;
            corners_Seq.push_back(corners);
        }   
        image_Seq.push_back(image);
    }   
	if (!conner_flag)//如果有提取失败的标定图，退出程序
	{
		cout<<"corner extract failed"<<endl;
		return 0;
	}
    cout<<"corner extract completed"<<endl; 
    /************************************************************************  
           摄像机定标  
    *************************************************************************/   
    cout<<"start calibration"<<endl;  
	Size square_size = Size(55,55);     
	vector<vector<Point3f>>  object_Points;        /****  保存定标板上角点的三维坐标   ****/

    vector<int>  point_counts;                                                         
    /* 初始化定标板上角点的三维坐标 */
	for (int t = 0; t<successImageNum; t++)
    {
        vector<Point3f> tempPointSet;
        for (int i = 0; i<board_size.height; i++)
        {
            for (int j = 0; j<board_size.width; j++)
            {
                /* 假设定标板放在世界坐标系中z=0的平面上 */
                Point3f tempPoint;
                tempPoint.x = i*square_size.width;
                tempPoint.y = j*square_size.height;
                tempPoint.z = 0;
                tempPointSet.push_back(tempPoint);
            }
        }
        object_Points.push_back(tempPointSet);
    }
	for (int i = 0; i< successImageNum; i++)
    {
        point_counts.push_back(board_size.width*board_size.height);
    }
    /* 开始定标 */
    Size image_size = image_Seq[0].size();
    //image_size = Size(image_size.width*1.5,image_size.height*1.2);
    cv::Matx33d intrinsic_matrix;    /*****    摄像机内参数矩阵    ****/
    cv::Vec4d distortion_coeffs;     /* 摄像机的4个畸变系数：k1,k2,k3,k4*/
    std::vector<cv::Vec3d> rotation_vectors;                           /* 每幅图像的旋转向量 */
    std::vector<cv::Vec3d> translation_vectors;                        /* 每幅图像的平移向量 */
    int flags = 0;
    flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flags |= cv::fisheye::CALIB_CHECK_COND;
    flags |= cv::fisheye::CALIB_FIX_SKEW;
    fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
    cout<<"calibration completed\n";   

    /************************************************************************  
           对定标结果进行评价  
    *************************************************************************/   
    cout<<"start evaluation"<<endl;   
    double total_err = 0.0;                   /* 所有图像的平均误差的总和 */   
    double err = 0.0;                        /* 每幅图像的平均误差 */   
    vector<Point2f>  image_points2;             /****   保存重新计算得到的投影点    ****/   

    cout<<"calibration error for each img"<<endl<<endl;   
    for (int i=0;  i<image_count;  i++) 
    {
        vector<Point3f> tempPointSet = object_Points[i];
        /****    通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点     ****/
		fisheye::projectPoints(tempPointSet, image_points2, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs);
        /* 计算新的投影点和旧的投影点之间的误差*/  
        vector<Point2f> tempImagePoint = corners_Seq[i];
        Mat tempImagePointMat = Mat(1,tempImagePoint.size(),CV_32FC2);
        Mat image_points2Mat = Mat(1,image_points2.size(), CV_32FC2);
        for (size_t i = 0 ; i != tempImagePoint.size(); i++)
        {
            image_points2Mat.at<Vec2f>(0,i) = Vec2f(image_points2[i].x, image_points2[i].y);
            tempImagePointMat.at<Vec2f>(0,i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
        }
        err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
        total_err += err/=  point_counts[i];   
        cout<<"No:"<<i+1<<"img mean error"<<err<<"pixel"<<endl;
        fout<<"No:"<<i+1<<"img mean error"<<err<<"pixel"<<endl;
    }   
    cout<<"total mean error:"<<total_err/image_count<<"pixel"<<endl;
    fout<<"total mean error:"<<total_err/image_count<<"pixel"<<endl<<endl;
    cout<<"evaluation completed"<<endl;

    /************************************************************************  
           保存定标结果  
    *************************************************************************/   
    cout<<"save calibration result"<<endl;       
    Mat rotation_matrix = Mat(3,3,CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */   

    fout<<"camera intrinsic parameters"<<endl;   
    fout<<intrinsic_matrix<<endl;   
    fout<<"distortion coefficent\n";  
    fout<<distortion_coeffs<<endl;    

    cout<<intrinsic_matrix<<endl;   
    cout<<distortion_coeffs<<endl;     
    
    for (int i=0; i<image_count; i++) 
    { 
        fout<<"No:"<<i+1<<"img rotation vector"<<endl;   
        fout<<rotation_vectors[i]<<endl;   

        /* 将旋转向量转换为相对应的旋转矩阵 */   
        Rodrigues(rotation_vectors[i],rotation_matrix);   
        fout<<"No:"<<i+1<<"img rotation matrice"<<endl;   
        fout<<rotation_matrix<<endl;   
        fout<<"No:"<<i+1<<"img traslation vector"<<endl;   
        fout<<translation_vectors[i]<<endl;   
    }   
    cout<<"saving completed"<<endl; 
    fout<<endl;


    /************************************************************************  
           显示定标结果  
    *************************************************************************/
    Size undistorSize = Size(image_size.width*1.5,image_size.height*1.5);
    undistorSize = image_size;
    Mat mapx = Mat(undistorSize,CV_32FC1);
    Mat mapy = Mat(undistorSize,CV_32FC1);
    Mat R = Mat::eye(3,3,CV_32F);

    Mat newMatrix = Mat::eye(3,3,CV_32F);
    cout<<"save undistored img"<<endl;

    auto optMatrix = getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, image_size, 1, undistorSize, 0);
    cout<<"getOptimalNewCameraMatrix"<<endl<<optMatrix<<endl; 
    // optMatrix.at<double>(0,0) = optMatrix.at<double>(0,0)/2;
    // optMatrix.at<double>(1,1) = optMatrix.at<double>(1,1)/2;

    fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsic_matrix, distortion_coeffs, image_size, R, newMatrix, 0.5f, undistorSize);
    //balance=1 reserve all pixel
	cout<<"estimateNewCameraMatrixForUndistortRectify"<<endl<<newMatrix<<endl;   

    for (int i = 0 ; i != image_count ; i++)
    {
		fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs, R, optMatrix, undistorSize,CV_32FC1,mapx,mapy);
        // fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R,
        //     getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, image_size, 1, image_size, 0), image_size, CV_32FC1, mapx, mapy);

        Mat t = Mat(undistorSize,CV_8UC3);
        cv::remap(image_Seq[i],t,mapx, mapy, INTER_CUBIC);
        string imageFileName;
        std::stringstream StrStm;
        StrStm<<i+1;
        StrStm>>imageFileName;
        imageFileName += "_d.jpg";
        imwrite(imageFileName,t);
		//imshow(imageFileName, t);
		cout<<"img"<<i+1<<"save"<<endl;
    }
    cout<<"saving completed"<<endl;


    /************************************************************************  
           测试一张图片  
    *************************************************************************/
    // if (1)
    // {
    //     cout<<"TestImage ..."<<endl;
    //     Mat image = imread("a.jpg",1);
	// 	Mat testImage;
	// 	copyMakeBorder(image,testImage,(int)(y_expand/2),(int)(y_expand/2),(int)(x_expand/2),(int)(x_expand/2),BORDER_CONSTANT);
    //     fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,intrinsic_matrix,image_size,CV_32FC1,mapx,mapy);
    //     Mat t = testImage.clone();
    //     cv::remap(testImage,t,mapx, mapy, INTER_LINEAR);

    //     imwrite("TestOutput.jpg",t);
	// 	imshow("TestOutput.jpg", t);
    //     cout<<"保存结束"<<endl;
    // }
	// while (1)
	// {
	// 	if (cvWaitKey(15)==27)
	// 	{
	// 		break;
	// 	}
	// }

    return 0;
}