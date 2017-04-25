//-----------------------------------【头文件包含部分】------------------------------------------------------
//		描述：本代码是对于绝缘体的检测部分
//      name:wuxiangrong,begin 2016/10/25
//      VS2015 opencv2.4.9 win10
//----------------------------------------------------------------------------------------------------------- 
#include "stdafx.h"
#include <iostream>  
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <fstream>
#include <opencv2/contrib/contrib.hpp>

//-----------------------------------【命名空间声明部分】----------------------------------------------------
//		描述：包含程序所使用的命名空间
//----------------------------------------------------------------------------------------------------------- 
using namespace cv;
using namespace std;

//-----------------------------------【调用函数声明部分】-----------------------------------------------------
//		描述：包含程序所使用的调用函数
//------------------------------------------------------------------------------------------------------------ 
float SortGroup(vector<float> VECtor, float num, float part[10][10]);
IplImage* rotateImage1(IplImage* img, int degree);
Mat LinearAdjust(Mat matrix, float fa, float fb);
Point OppositeMap(Point point_in, float cosa, float sina, Mat image_in, Mat image_rota);

//-----------------------------------【main( )函数】----------------------------------------------------------
//		描述：控制台应用程序的入口函数，程序从这里开始
//------------------------------------------------------------------------------------------------------------
int main() {
//---------------【1】载入原始图和Mat变量定义-----------------------------------------------------------------
	
	/*string filepath = "E:\\projects\\Visual studio 15\\Microsoft\\DefectInspection\\DefectInspection\\12\\"; 
	Directory DIR;
	vector<string> filenames = DIR.GetListFiles(filepath, "*.jpg", false);*/

	Mat srcImage = imread("./12\\21.jpg");
	Mat midImage, dstImage,orignal;       //临时变量和目标图的定义
	srcImage.copyTo(orignal);

	system("cd ./testPro&&del/s *.jpg");  //清空文件夹中存储的图像 
	char file_dst[100];                   //定义将要保存文件名空间
	ofstream out("data.txt", ios::trunc);  //清空
	ofstream myfile("data.txt",ios::out);
	if(!myfile) cout<<"error !";
	int window_num = 1;  //框的个数
	
	//参数
	int Pole_Width = 44;  //杆两侧先验宽
	int window_wide = 6;  //滑动窗宽度
	//for (unsigned int m = 0; m < filenames.size(); m++) {

//---------------【2】进行霍夫线变换检测线段------------------------------------------------------------------
		Canny(srcImage, midImage, 50, 200, 3);      //进行一次canny边缘检测
		cvtColor(midImage, dstImage, CV_GRAY2BGR);  //转化边缘检测后的图为灰度图
		//进行霍夫线变换
		vector<Vec4i> lines;       //定义一个矢量结构lines用于存放得到的线段矢量集合
		vector<float> line_slope;  //定义每条线段的斜率
		vector<float> distance;    //线段到（0，0）点的距离
		HoughLinesP(midImage, lines, 1, CV_PI / 180, 50, 100, 4);

//---------------【3】计算每条线段的斜率和到原点的距离--------------------------------------------------------
			//依次在图中绘制出每条线段并计算信息
		for (size_t i = 0; i < lines.size(); i++) {
			Vec4i l = lines[i];
			float slope_num = float(l[3] - l[1]) / (l[2] - l[0]);  //线的斜率
			float point_distance = abs(slope_num * l[0] - l[1]) / (1 + slope_num*slope_num);  //线到(0，0)点的距离
			line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186, 88, 255), 1, CV_AA);
			line_slope.push_back(slope_num);
			distance.push_back(point_distance);
		}

//---------------【4】以所有的斜率信息排序、分组--------------------------------------------------------------
		float part1[10][10];  //提前定义分10组之内
		memset(part1, 0, sizeof(part1));  //数组中内容置0
		vector<float> line_slope_crop(line_slope);  //定义容器的副集
		part1[10][10] = SortGroup(line_slope_crop, 0.006, part1);  //排序分组存储在数组[10][10]中
		int name = 0;  //图片命名
		//在分完组的Part1数组中，分析每组（行）中数据
		for (size_t part1_i = 0; part1_i < 10; part1_i++) {
			vector<Vec4i> linepart;  //在斜率相同的一组中
			vector<float> distancepart;
			vector<float> slopepart;
			int EachGroup_num = 0;  //每组的线段个数
			for (size_t part1_j = 0; part1_j < 10; part1_j++) {
				//将分组后，每组的线段点、线率、到（0，0）点的距离信息重新存储入新的容器
				for (size_t dis_i = 0; dis_i < distance.size(); dis_i++) {
					if (part1[part1_i][part1_j] == line_slope[dis_i]) {
						linepart.push_back(lines[dis_i]);
						distancepart.push_back(distance[dis_i]);
						slopepart.push_back(line_slope[dis_i]);
						EachGroup_num = EachGroup_num + 1;
					}
				}
			}
			//对没有信息的数组信息省去处理
			if (EachGroup_num < 1) {
				break;
			}

//---------------【5】以每组中的线段到（0，0）点距离的信息排序、分组-----------------------------------------
			float part2[10][10];
			memset(part2, 0, sizeof(part2)); //数组中内容置0
			vector<float> distancepart_crop(distancepart);
			part2[10][10] = SortGroup(distancepart_crop, 10, part2);
			//在分完组的Part2数组中，分析每组（行）中数据
			for (size_t part2_i = 0; part2_i < 10; part2_i++) {
				int part_num = 0;  //每条杆上的线段个数
				int sum_x1 = 0, sum_y1 = 0, sum_x2 = 0, sum_y2 = 0;  //点信息
				for (size_t part2_j = 0; part2_j < 10; part2_j++) {
					//将条杆上的线段电信息统计
					for (size_t disPart_i = 0; disPart_i < distancepart.size(); disPart_i++) {
						if (part2[part2_i][part2_j] == distancepart[disPart_i]) {
							part_num = part_num + 1;
							sum_x1 = sum_x1 + linepart[disPart_i][0];
							sum_y1 = sum_y1 + linepart[disPart_i][1];
							sum_x2 = sum_x2 + linepart[disPart_i][2];
							sum_y2 = sum_y2 + linepart[disPart_i][3];
						}
					}
				}
				//对没有信息的数组信息省去处理
				if (part_num < 1) {
					break;
				}
				//刷新杆的点信息及斜率
				sum_x1 = sum_x1 / part_num;
				sum_y1 = sum_y1 / part_num;
				sum_x2 = sum_x2 / part_num;
				sum_y2 = sum_y2 / part_num;
				float part_slops = 0;
				part_slops = float(sum_y2 - sum_y1) / (sum_x2 - sum_x1);  //重新算得斜率

//---------------【6】以每条杆的斜率信息旋转原图（使杆水平）----------------------------------------------------------------
				double theta = 0;  //定义旋转角度
				theta = atan(part_slops)*(180 / CV_PI);
				float cosa = float(cos(double(theta * CV_PI / 180)));
				float sina = float(sin(double(theta * CV_PI / 180)));
				//图像从Mat转为IplImage*旋转
				IplImage * srcImage_pro;
				srcImage_pro = &IplImage(srcImage);
				Mat src = rotateImage1(srcImage_pro, theta);  //旋转函数
				imshow("旋转后图", src);
				sprintf_s(file_dst, "./testPro\\cool_rotate\\%d.jpg", name);  //旋转图像保存
				imwrite(file_dst, src);

				//图像旋转后，原图像到旋转后图像点的映射
				int point1_x = sum_x1 * cosa + sum_y1 * sina - cosa*srcImage.cols / 2 - sina*srcImage.rows / 2 + src.cols / 2;
				int point1_y = -sum_x1 * sina + sum_y1 * cosa + sina*srcImage.cols / 2 - cosa*srcImage.rows / 2 + src.rows / 2;
				int point2_x = sum_x2 * cosa + sum_y2 * sina - cosa*srcImage.cols / 2 - sina*srcImage.rows / 2 + src.cols / 2;
				int point2_y = -sum_x2 * sina + sum_y2 * cosa + sina*srcImage.cols / 2 - cosa*srcImage.rows / 2 + src.rows / 2;

				//映射公式的检测
				Point one(srcImage.cols / 2 + 2, srcImage.rows / 2 + 2), two(src.cols / 2 + 2, src.rows / 2 + 2);
				line(srcImage, Point(sum_x1, sum_y1), Point(sum_x2, sum_y2), Scalar(186, 88, 255), 1, CV_AA);
				line(srcImage, Point(srcImage.cols / 2, srcImage.rows / 2), one, Scalar(186, 88, 255), 1, CV_AA);  //中点
				line(src, Point(point1_x, point1_y), Point(point2_x, point2_y), Scalar(186, 88, 255), 1, CV_AA);
				line(src, Point(src.cols / 2, src.rows / 2), two, Scalar(186, 88, 255), 1, CV_AA);  //中点
				imshow("映射原图", srcImage);
				imshow("映射旋转图", src);

//---------------【6】截取整条杆的一定宽度的信息----------------------------------------------------------------------------
				Rect rect1(0, point1_y - Pole_Width / 2, src.cols, Pole_Width);  //先验的给出绝缘体的观测宽度
				Mat image_roi1 = src(rect1);  //截取矩形
				imshow("截取整条杆", image_roi1);
				sprintf_s(file_dst, "./testPro\\coolblock\\%d.jpg", name);  //截取含杆信息图片
				imwrite(file_dst, image_roi1);
				//对截取图像简单预处理
				Mat image_roi, image_roi1_gray, image_equa;
				cvtColor(image_roi1, image_roi1_gray, CV_BGR2GRAY);
				image_roi = LinearAdjust(image_roi1_gray, 35, 200);  //线性动态调整函数 35, 200
				sprintf_s(file_dst, "./testPro\\cool_blockAdj\\%d.jpg", name);  //保存线性调整后的图像
				imwrite(file_dst, image_roi);
				equalizeHist(image_roi, image_equa);  //直方图均衡化
				imshow("1", image_roi);
				imshow("2", image_equa);
				
				
				waitKey(20);
//---------------【7】对截取的图片的二值图像素做垂直投影--------------------------------------------------------------------
				Mat Binary;
				//自适应阈值二值化
				//adaptiveThreshold(image_roi,Binary,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY_INV,25,10);
				threshold(image_equa, Binary, 20, 255, CV_THRESH_BINARY);  //20
				imshow("二值化", Binary);
				sprintf_s(file_dst, "./testPro\\cool_blockBIN\\%d.jpg", name);  //保存二值化后的图像
				imwrite(file_dst, Binary);
				//垂直投影
				IplImage * bin;
				bin = &IplImage(Binary);
				IplImage* paintx = cvCreateImage(cvGetSize(bin), IPL_DEPTH_8U, 1);
				cvZero(paintx);  //初始化定义的paintx
				int* v = new int[bin->width];  //开辟指针v空间并初始化
				memset(v, 0, bin->width * 4);
				CvScalar temp;
				for (size_t xx = 0; xx < bin->width; xx++) {
					for (size_t yy = 0; yy < bin->height; yy++) {
						temp = cvGet2D(bin, yy, xx);  //获取该点像素值
						if (temp.val[0] == 0)
							v[xx]++;
					}
				}

				for (size_t xx = 0; xx < bin->width; xx++) {
					for (size_t yy = 0; yy < v[xx]; yy++) {
						temp.val[0] = 255;
						cvSet2D(paintx, yy, xx, temp);  //像素点填充
					}
				}
				//cvNamedWindow("垂直积分投影", 1);
				//cvShowImage("垂直积分投影", paintx);
				Mat paintxx = paintx;
				sprintf_s(file_dst, "./testPro\\cool_shadow\\%d.jpg", name);  //保存垂直投影后图像
				imwrite(file_dst, paintxx);


//---------------【8】用构造滑动窗判断的方法再次截取图像得到绝缘体----------------------------------------------------------
				int OneZero[1000];  //定义1000长的数组并初始化0
				memset(OneZero, 0, sizeof(OneZero));
				for (size_t uu = 2; uu < bin->width; uu++) {
					int Max = Binary.rows - v[uu], Min = Binary.rows - v[uu];
					for (size_t vv = 1; vv < window_wide; vv++) {  //用6个像素宽的滑窗 搜索临近间的像素差
						if (Binary.rows - v[uu + vv] > Max) {  //寻找最大值
							Max = Binary.rows - v[uu + vv];
						}
						if (Binary.rows - v[uu + vv] < Min) {  //寻找最小值
							Min = Binary.rows - v[uu + vv];
						}
					}
					if (Max - Min > 4) {  //如果最大值与最小值差6以上，则标记为1，否则为0
						OneZero[uu] = 1;
					}
				}
				int Long = 0, Max_Long = 0, orig_point = 0;
				for (size_t array_i = 1; array_i < 1000; array_i++) {
					//找出数组中连续标记为1最长序列的起点orig_point和长度Max_Long
					if (OneZero[array_i] == 1) {
						if (OneZero[array_i - 1] == 1) {
							Long = Long + 1;
						}
						else {
							if (Long > Max_Long) {
								Max_Long = Long;
								orig_point = array_i - Max_Long;
							}
							Long = 0;
						}
					}
				}

				//当标记为1序列长大于30起确定其为绝缘体
				if (Max_Long > window_wide * 7)
				{
					Rect rec(orig_point, 0, Max_Long, paintxx.rows);  //在截取图像中框出绝缘体
					rectangle(image_roi1, rec, Scalar(0, 0, 255), 1, 8, 0);
					sprintf_s(file_dst, "./testPro\\cool_slice\\%d.jpg", name);
					imwrite(file_dst, image_roi1);

					Rect rec_rota(orig_point, point1_y - Pole_Width / 2, Max_Long, paintxx.rows);  //在旋转图像中框出绝缘体
					rectangle(src, rec_rota, Scalar(0, 0, 255), 1, 8, 0);
					sprintf_s(file_dst, "./testPro\\cool_slice_rota\\%d.jpg", name);
					imwrite(file_dst, src);

//---------------【9】检测框进行简单调整------------------------------------------------------------------------------------
									//对检测框进行简单调整
					Mat image_roi2 = src(rec_rota);  //截取矩形
					Mat image_ROI, image_roi2_gray, Binary_roi;
					IplImage * Bin_roi;
					cvtColor(image_roi2, image_roi2_gray, CV_BGR2GRAY);
					image_ROI = LinearAdjust(image_roi2_gray, 35, 200);  //线性动态调整函数
					threshold(image_ROI, Binary_roi, 20, 255, CV_THRESH_BINARY);
					Bin_roi = &IplImage(Binary_roi);
					imshow("dfsdf", Binary_roi);
					//水平投影
					int* H = new int[Bin_roi->height];  //开辟指针H空间并初始化
					memset(H, 0, Bin_roi->height * 4);
					CvScalar temp;
					for (size_t xx = 0; xx < Bin_roi->height; xx++) {
						for (size_t yy = 0; yy < Bin_roi->width; yy++) {
							temp = cvGet2D(Bin_roi, xx, yy);  //获取该点像素值
							if (temp.val[0] == 0)
								H[xx]++;
						}
					}

					IplImage* paintxy = cvCreateImage(cvGetSize(bin), IPL_DEPTH_8U, 1);
					cvZero(paintxy);  //初始化定义的paintx
					for (size_t xx = 0; xx < Bin_roi->height; xx++) {
						for (size_t yy = 0; yy < H[xx]; yy++) {
							temp.val[0] = 255;
							cvSet2D(paintxy, xx, yy, temp);  //像素点填充
						}
					}
					cvNamedWindow("水平积分投影", 1);
					cvShowImage("水平积分投影", paintxy);
					//waitKey(0);
					float top = 0, bottom = 0;  //对检测框上下边调整
					for (size_t xx = 1; xx <= Bin_roi->height - 1; xx++) {
						if (H[xx] < Bin_roi->width - 4) {
							top = xx;
							break;
						}
					}
					for (size_t yy = Bin_roi->height - 1; yy >= 0; yy--) {
						if (H[yy] < Bin_roi->width - 4) {
							bottom = Bin_roi->height - 1 - yy;
							break;
						}
					}

					orig_point = orig_point - 4;  //检测到的框左右个扩展4个像素
					Max_Long = Max_Long + 8;
					top = top - 4;                //检测到的框上下个扩展4个像素
					bottom = bottom - 4;

					//从旋转图像到原图像的点逆变换
					Point Rota_One(orig_point, point1_y - Pole_Width / 2 + top);  //四个点,
					Point Rota_Two(orig_point, point1_y + Pole_Width / 2 - bottom);
					Point Rota_Three(orig_point + Max_Long, point1_y - Pole_Width / 2 + top);
					Point Rota_Four(orig_point + Max_Long, point1_y + Pole_Width / 2 - bottom);
					Point pointRO1 = OppositeMap(Rota_One, cosa, sina, srcImage, src);  //逆变换后的四个对应点
					Point pointRO2 = OppositeMap(Rota_Two, cosa, sina, srcImage, src);
					Point pointRO3 = OppositeMap(Rota_Three, cosa, sina, srcImage, src);
					Point pointRO4 = OppositeMap(Rota_Four, cosa, sina, srcImage, src);

					myfile << "第"<< window_num <<"个框的信息：" << endl
						<< Rota_One << "、" << Rota_Two << "、" << Rota_Three << "、" << Rota_Four << endl;
					//在原图中画出对应的框
					line(orignal, pointRO1, pointRO2, Scalar(0, 0, 255), 1, CV_AA);
					line(orignal, pointRO1, pointRO3, Scalar(0, 0, 255), 1, CV_AA);
					line(orignal, pointRO2, pointRO4, Scalar(0, 0, 255), 1, CV_AA);
					line(orignal, pointRO3, pointRO4, Scalar(0, 0, 255), 1, CV_AA);
					int name_out = 0;
					sprintf_s(file_dst, "./testPro\\out\\%d.jpg", name_out);  //保存绝缘体检测后的图
					imwrite(file_dst, orignal);
					imshow("绝缘体检测后的图像", orignal);
					window_num = window_num + 1;
				}
				name = name + 1;
			}
		}
	//}
	myfile.close();
	waitKey(0);
	return 0;
}

//数字容器的排序分组函数
float SortGroup(vector<float> VECtor, float GroupThreshold,  float part[10][10])
{
	//vector<float> VectorCrop = VECtor;  //定义VECtord的一个副集
	sort(VECtor.begin(), VECtor.end());  //排序
	memset(part, 0, sizeof(part)); //数组中内容置0
	int ss = 0, s = 0;
	part[ss][s] = VECtor[0];
	for (size_t i = 1; i < VECtor.size(); i++) {
		s = s + 1;
		if (abs(VECtor[i - 1] - VECtor[i]) > GroupThreshold) {
			ss = ss + 1;
			s = 0;
		}
		part[ss][s] = VECtor[i];
	}
	return part[10][10];
}

//旋转图像内容不变，尺寸相应变大  
IplImage* rotateImage1(IplImage* img,int degree){  
    double angle = degree  * CV_PI / 180.; // 弧度    
    double a = sin(angle), b = cos(angle);   
    int width = img->width;    
    int height = img->height;    
    int width_rotate= int(height * fabs(a) + width * fabs(b));    
    int height_rotate=int(width * fabs(a) + height * fabs(b));    
    //旋转数组map  
    // [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]  
    // [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]  
    float map[6];  
    CvMat map_matrix = cvMat(2, 3, CV_32F, map);    
    // 旋转中心  
    CvPoint2D32f center = cvPoint2D32f((width+1) / 2,( height + 1) / 2);    
    cv2DRotationMatrix(center, degree, 1.0, &map_matrix);    
    map[2] += (width_rotate - width) / 2;    
    map[5] += (height_rotate - height) / 2;    
    IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate), 8, 3);   
    //对图像做仿射变换  
    //CV_WARP_FILL_OUTLIERS - 填充所有输出图像的象素。  
    //如果部分象素落在输入图像的边界外，那么它们的值设定为 fillval.  
    //CV_WARP_INVERSE_MAP - 指定 map_matrix 是输出图像到输入图像的反变换，  
    cvWarpAffine( img,img_rotate, &map_matrix, CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS, cvScalarAll(0));    
    return img_rotate;  
}  

//线性动态调整函数
Mat LinearAdjust(Mat matrix, float fa, float fb)
{
	float bled = 255 / (fb - fa);
	Mat matrix_out = Mat::zeros(matrix.rows,matrix.cols,CV_8UC1);
	for (int i = 0; i < matrix.rows-1; i++)
	{
		for (int j = 0; j < matrix.cols-1; j++)
		{
			int uu = matrix.at<uchar>(i, j);
			if (matrix.at<uchar>(i, j) >= fa && matrix.at<uchar>(i, j) < fb)
			{
				float rr = bled *(matrix.at<uchar>(i, j) - fa);
				matrix_out.at<uchar>(i, j) = rr;
			}
			if (matrix.at<uchar>(i, j) >= fb)
			{
				matrix_out.at<uchar>(i, j) = 255;
			}
		}
	}
	return matrix_out;
}

//从旋转图像到原图像的点逆变换
Point OppositeMap(Point point_in,float cosa, float sina, Mat image_in, Mat image_rota) {
	Point point;
	point.x = point_in.x * cosa - point_in.y * sina - cosa*image_rota.cols / 2 + sina*image_rota.rows / 2 + image_in.cols / 2;
	point.y = point_in.x * sina + point_in.y * cosa - sina*image_rota.cols / 2 - cosa*image_rota.rows / 2 + image_in.rows / 2;
	return point;
}