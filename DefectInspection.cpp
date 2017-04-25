//-----------------------------------��ͷ�ļ��������֡�------------------------------------------------------
//		�������������Ƕ��ھ�Ե��ļ�ⲿ��
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

//-----------------------------------�������ռ��������֡�----------------------------------------------------
//		����������������ʹ�õ������ռ�
//----------------------------------------------------------------------------------------------------------- 
using namespace cv;
using namespace std;

//-----------------------------------�����ú����������֡�-----------------------------------------------------
//		����������������ʹ�õĵ��ú���
//------------------------------------------------------------------------------------------------------------ 
float SortGroup(vector<float> VECtor, float num, float part[10][10]);
IplImage* rotateImage1(IplImage* img, int degree);
Mat LinearAdjust(Mat matrix, float fa, float fb);
Point OppositeMap(Point point_in, float cosa, float sina, Mat image_in, Mat image_rota);

//-----------------------------------��main( )������----------------------------------------------------------
//		����������̨Ӧ�ó������ں�������������￪ʼ
//------------------------------------------------------------------------------------------------------------
int main() {
//---------------��1������ԭʼͼ��Mat��������-----------------------------------------------------------------
	
	/*string filepath = "E:\\projects\\Visual studio 15\\Microsoft\\DefectInspection\\DefectInspection\\12\\"; 
	Directory DIR;
	vector<string> filenames = DIR.GetListFiles(filepath, "*.jpg", false);*/

	Mat srcImage = imread("./12\\21.jpg");
	Mat midImage, dstImage,orignal;       //��ʱ������Ŀ��ͼ�Ķ���
	srcImage.copyTo(orignal);

	system("cd ./testPro&&del/s *.jpg");  //����ļ����д洢��ͼ�� 
	char file_dst[100];                   //���彫Ҫ�����ļ����ռ�
	ofstream out("data.txt", ios::trunc);  //���
	ofstream myfile("data.txt",ios::out);
	if(!myfile) cout<<"error !";
	int window_num = 1;  //��ĸ���
	
	//����
	int Pole_Width = 44;  //�����������
	int window_wide = 6;  //���������
	//for (unsigned int m = 0; m < filenames.size(); m++) {

//---------------��2�����л����߱任����߶�------------------------------------------------------------------
		Canny(srcImage, midImage, 50, 200, 3);      //����һ��canny��Ե���
		cvtColor(midImage, dstImage, CV_GRAY2BGR);  //ת����Ե�����ͼΪ�Ҷ�ͼ
		//���л����߱任
		vector<Vec4i> lines;       //����һ��ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������
		vector<float> line_slope;  //����ÿ���߶ε�б��
		vector<float> distance;    //�߶ε���0��0����ľ���
		HoughLinesP(midImage, lines, 1, CV_PI / 180, 50, 100, 4);

//---------------��3������ÿ���߶ε�б�ʺ͵�ԭ��ľ���--------------------------------------------------------
			//������ͼ�л��Ƴ�ÿ���߶β�������Ϣ
		for (size_t i = 0; i < lines.size(); i++) {
			Vec4i l = lines[i];
			float slope_num = float(l[3] - l[1]) / (l[2] - l[0]);  //�ߵ�б��
			float point_distance = abs(slope_num * l[0] - l[1]) / (1 + slope_num*slope_num);  //�ߵ�(0��0)��ľ���
			line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186, 88, 255), 1, CV_AA);
			line_slope.push_back(slope_num);
			distance.push_back(point_distance);
		}

//---------------��4�������е�б����Ϣ���򡢷���--------------------------------------------------------------
		float part1[10][10];  //��ǰ�����10��֮��
		memset(part1, 0, sizeof(part1));  //������������0
		vector<float> line_slope_crop(line_slope);  //���������ĸ���
		part1[10][10] = SortGroup(line_slope_crop, 0.006, part1);  //�������洢������[10][10]��
		int name = 0;  //ͼƬ����
		//�ڷ������Part1�����У�����ÿ�飨�У�������
		for (size_t part1_i = 0; part1_i < 10; part1_i++) {
			vector<Vec4i> linepart;  //��б����ͬ��һ����
			vector<float> distancepart;
			vector<float> slopepart;
			int EachGroup_num = 0;  //ÿ����߶θ���
			for (size_t part1_j = 0; part1_j < 10; part1_j++) {
				//�������ÿ����߶ε㡢���ʡ�����0��0����ľ�����Ϣ���´洢���µ�����
				for (size_t dis_i = 0; dis_i < distance.size(); dis_i++) {
					if (part1[part1_i][part1_j] == line_slope[dis_i]) {
						linepart.push_back(lines[dis_i]);
						distancepart.push_back(distance[dis_i]);
						slopepart.push_back(line_slope[dis_i]);
						EachGroup_num = EachGroup_num + 1;
					}
				}
			}
			//��û����Ϣ��������Ϣʡȥ����
			if (EachGroup_num < 1) {
				break;
			}

//---------------��5����ÿ���е��߶ε���0��0����������Ϣ���򡢷���-----------------------------------------
			float part2[10][10];
			memset(part2, 0, sizeof(part2)); //������������0
			vector<float> distancepart_crop(distancepart);
			part2[10][10] = SortGroup(distancepart_crop, 10, part2);
			//�ڷ������Part2�����У�����ÿ�飨�У�������
			for (size_t part2_i = 0; part2_i < 10; part2_i++) {
				int part_num = 0;  //ÿ�����ϵ��߶θ���
				int sum_x1 = 0, sum_y1 = 0, sum_x2 = 0, sum_y2 = 0;  //����Ϣ
				for (size_t part2_j = 0; part2_j < 10; part2_j++) {
					//�������ϵ��߶ε���Ϣͳ��
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
				//��û����Ϣ��������Ϣʡȥ����
				if (part_num < 1) {
					break;
				}
				//ˢ�¸˵ĵ���Ϣ��б��
				sum_x1 = sum_x1 / part_num;
				sum_y1 = sum_y1 / part_num;
				sum_x2 = sum_x2 / part_num;
				sum_y2 = sum_y2 / part_num;
				float part_slops = 0;
				part_slops = float(sum_y2 - sum_y1) / (sum_x2 - sum_x1);  //�������б��

//---------------��6����ÿ���˵�б����Ϣ��תԭͼ��ʹ��ˮƽ��----------------------------------------------------------------
				double theta = 0;  //������ת�Ƕ�
				theta = atan(part_slops)*(180 / CV_PI);
				float cosa = float(cos(double(theta * CV_PI / 180)));
				float sina = float(sin(double(theta * CV_PI / 180)));
				//ͼ���MatתΪIplImage*��ת
				IplImage * srcImage_pro;
				srcImage_pro = &IplImage(srcImage);
				Mat src = rotateImage1(srcImage_pro, theta);  //��ת����
				imshow("��ת��ͼ", src);
				sprintf_s(file_dst, "./testPro\\cool_rotate\\%d.jpg", name);  //��תͼ�񱣴�
				imwrite(file_dst, src);

				//ͼ����ת��ԭͼ����ת��ͼ����ӳ��
				int point1_x = sum_x1 * cosa + sum_y1 * sina - cosa*srcImage.cols / 2 - sina*srcImage.rows / 2 + src.cols / 2;
				int point1_y = -sum_x1 * sina + sum_y1 * cosa + sina*srcImage.cols / 2 - cosa*srcImage.rows / 2 + src.rows / 2;
				int point2_x = sum_x2 * cosa + sum_y2 * sina - cosa*srcImage.cols / 2 - sina*srcImage.rows / 2 + src.cols / 2;
				int point2_y = -sum_x2 * sina + sum_y2 * cosa + sina*srcImage.cols / 2 - cosa*srcImage.rows / 2 + src.rows / 2;

				//ӳ�乫ʽ�ļ��
				Point one(srcImage.cols / 2 + 2, srcImage.rows / 2 + 2), two(src.cols / 2 + 2, src.rows / 2 + 2);
				line(srcImage, Point(sum_x1, sum_y1), Point(sum_x2, sum_y2), Scalar(186, 88, 255), 1, CV_AA);
				line(srcImage, Point(srcImage.cols / 2, srcImage.rows / 2), one, Scalar(186, 88, 255), 1, CV_AA);  //�е�
				line(src, Point(point1_x, point1_y), Point(point2_x, point2_y), Scalar(186, 88, 255), 1, CV_AA);
				line(src, Point(src.cols / 2, src.rows / 2), two, Scalar(186, 88, 255), 1, CV_AA);  //�е�
				imshow("ӳ��ԭͼ", srcImage);
				imshow("ӳ����תͼ", src);

//---------------��6����ȡ�����˵�һ����ȵ���Ϣ----------------------------------------------------------------------------
				Rect rect1(0, point1_y - Pole_Width / 2, src.cols, Pole_Width);  //����ĸ�����Ե��Ĺ۲���
				Mat image_roi1 = src(rect1);  //��ȡ����
				imshow("��ȡ������", image_roi1);
				sprintf_s(file_dst, "./testPro\\coolblock\\%d.jpg", name);  //��ȡ������ϢͼƬ
				imwrite(file_dst, image_roi1);
				//�Խ�ȡͼ���Ԥ����
				Mat image_roi, image_roi1_gray, image_equa;
				cvtColor(image_roi1, image_roi1_gray, CV_BGR2GRAY);
				image_roi = LinearAdjust(image_roi1_gray, 35, 200);  //���Զ�̬�������� 35, 200
				sprintf_s(file_dst, "./testPro\\cool_blockAdj\\%d.jpg", name);  //�������Ե������ͼ��
				imwrite(file_dst, image_roi);
				equalizeHist(image_roi, image_equa);  //ֱ��ͼ���⻯
				imshow("1", image_roi);
				imshow("2", image_equa);
				
				
				waitKey(20);
//---------------��7���Խ�ȡ��ͼƬ�Ķ�ֵͼ��������ֱͶӰ--------------------------------------------------------------------
				Mat Binary;
				//����Ӧ��ֵ��ֵ��
				//adaptiveThreshold(image_roi,Binary,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY_INV,25,10);
				threshold(image_equa, Binary, 20, 255, CV_THRESH_BINARY);  //20
				imshow("��ֵ��", Binary);
				sprintf_s(file_dst, "./testPro\\cool_blockBIN\\%d.jpg", name);  //�����ֵ�����ͼ��
				imwrite(file_dst, Binary);
				//��ֱͶӰ
				IplImage * bin;
				bin = &IplImage(Binary);
				IplImage* paintx = cvCreateImage(cvGetSize(bin), IPL_DEPTH_8U, 1);
				cvZero(paintx);  //��ʼ�������paintx
				int* v = new int[bin->width];  //����ָ��v�ռ䲢��ʼ��
				memset(v, 0, bin->width * 4);
				CvScalar temp;
				for (size_t xx = 0; xx < bin->width; xx++) {
					for (size_t yy = 0; yy < bin->height; yy++) {
						temp = cvGet2D(bin, yy, xx);  //��ȡ�õ�����ֵ
						if (temp.val[0] == 0)
							v[xx]++;
					}
				}

				for (size_t xx = 0; xx < bin->width; xx++) {
					for (size_t yy = 0; yy < v[xx]; yy++) {
						temp.val[0] = 255;
						cvSet2D(paintx, yy, xx, temp);  //���ص����
					}
				}
				//cvNamedWindow("��ֱ����ͶӰ", 1);
				//cvShowImage("��ֱ����ͶӰ", paintx);
				Mat paintxx = paintx;
				sprintf_s(file_dst, "./testPro\\cool_shadow\\%d.jpg", name);  //���洹ֱͶӰ��ͼ��
				imwrite(file_dst, paintxx);


//---------------��8���ù��컬�����жϵķ����ٴν�ȡͼ��õ���Ե��----------------------------------------------------------
				int OneZero[1000];  //����1000�������鲢��ʼ��0
				memset(OneZero, 0, sizeof(OneZero));
				for (size_t uu = 2; uu < bin->width; uu++) {
					int Max = Binary.rows - v[uu], Min = Binary.rows - v[uu];
					for (size_t vv = 1; vv < window_wide; vv++) {  //��6�����ؿ�Ļ��� �����ٽ�������ز�
						if (Binary.rows - v[uu + vv] > Max) {  //Ѱ�����ֵ
							Max = Binary.rows - v[uu + vv];
						}
						if (Binary.rows - v[uu + vv] < Min) {  //Ѱ����Сֵ
							Min = Binary.rows - v[uu + vv];
						}
					}
					if (Max - Min > 4) {  //������ֵ����Сֵ��6���ϣ�����Ϊ1������Ϊ0
						OneZero[uu] = 1;
					}
				}
				int Long = 0, Max_Long = 0, orig_point = 0;
				for (size_t array_i = 1; array_i < 1000; array_i++) {
					//�ҳ��������������Ϊ1����е����orig_point�ͳ���Max_Long
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

				//�����Ϊ1���г�����30��ȷ����Ϊ��Ե��
				if (Max_Long > window_wide * 7)
				{
					Rect rec(orig_point, 0, Max_Long, paintxx.rows);  //�ڽ�ȡͼ���п����Ե��
					rectangle(image_roi1, rec, Scalar(0, 0, 255), 1, 8, 0);
					sprintf_s(file_dst, "./testPro\\cool_slice\\%d.jpg", name);
					imwrite(file_dst, image_roi1);

					Rect rec_rota(orig_point, point1_y - Pole_Width / 2, Max_Long, paintxx.rows);  //����תͼ���п����Ե��
					rectangle(src, rec_rota, Scalar(0, 0, 255), 1, 8, 0);
					sprintf_s(file_dst, "./testPro\\cool_slice_rota\\%d.jpg", name);
					imwrite(file_dst, src);

//---------------��9��������м򵥵���------------------------------------------------------------------------------------
									//�Լ�����м򵥵���
					Mat image_roi2 = src(rec_rota);  //��ȡ����
					Mat image_ROI, image_roi2_gray, Binary_roi;
					IplImage * Bin_roi;
					cvtColor(image_roi2, image_roi2_gray, CV_BGR2GRAY);
					image_ROI = LinearAdjust(image_roi2_gray, 35, 200);  //���Զ�̬��������
					threshold(image_ROI, Binary_roi, 20, 255, CV_THRESH_BINARY);
					Bin_roi = &IplImage(Binary_roi);
					imshow("dfsdf", Binary_roi);
					//ˮƽͶӰ
					int* H = new int[Bin_roi->height];  //����ָ��H�ռ䲢��ʼ��
					memset(H, 0, Bin_roi->height * 4);
					CvScalar temp;
					for (size_t xx = 0; xx < Bin_roi->height; xx++) {
						for (size_t yy = 0; yy < Bin_roi->width; yy++) {
							temp = cvGet2D(Bin_roi, xx, yy);  //��ȡ�õ�����ֵ
							if (temp.val[0] == 0)
								H[xx]++;
						}
					}

					IplImage* paintxy = cvCreateImage(cvGetSize(bin), IPL_DEPTH_8U, 1);
					cvZero(paintxy);  //��ʼ�������paintx
					for (size_t xx = 0; xx < Bin_roi->height; xx++) {
						for (size_t yy = 0; yy < H[xx]; yy++) {
							temp.val[0] = 255;
							cvSet2D(paintxy, xx, yy, temp);  //���ص����
						}
					}
					cvNamedWindow("ˮƽ����ͶӰ", 1);
					cvShowImage("ˮƽ����ͶӰ", paintxy);
					//waitKey(0);
					float top = 0, bottom = 0;  //�Լ������±ߵ���
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

					orig_point = orig_point - 4;  //��⵽�Ŀ����Ҹ���չ4������
					Max_Long = Max_Long + 8;
					top = top - 4;                //��⵽�Ŀ����¸���չ4������
					bottom = bottom - 4;

					//����תͼ��ԭͼ��ĵ���任
					Point Rota_One(orig_point, point1_y - Pole_Width / 2 + top);  //�ĸ���,
					Point Rota_Two(orig_point, point1_y + Pole_Width / 2 - bottom);
					Point Rota_Three(orig_point + Max_Long, point1_y - Pole_Width / 2 + top);
					Point Rota_Four(orig_point + Max_Long, point1_y + Pole_Width / 2 - bottom);
					Point pointRO1 = OppositeMap(Rota_One, cosa, sina, srcImage, src);  //��任����ĸ���Ӧ��
					Point pointRO2 = OppositeMap(Rota_Two, cosa, sina, srcImage, src);
					Point pointRO3 = OppositeMap(Rota_Three, cosa, sina, srcImage, src);
					Point pointRO4 = OppositeMap(Rota_Four, cosa, sina, srcImage, src);

					myfile << "��"<< window_num <<"�������Ϣ��" << endl
						<< Rota_One << "��" << Rota_Two << "��" << Rota_Three << "��" << Rota_Four << endl;
					//��ԭͼ�л�����Ӧ�Ŀ�
					line(orignal, pointRO1, pointRO2, Scalar(0, 0, 255), 1, CV_AA);
					line(orignal, pointRO1, pointRO3, Scalar(0, 0, 255), 1, CV_AA);
					line(orignal, pointRO2, pointRO4, Scalar(0, 0, 255), 1, CV_AA);
					line(orignal, pointRO3, pointRO4, Scalar(0, 0, 255), 1, CV_AA);
					int name_out = 0;
					sprintf_s(file_dst, "./testPro\\out\\%d.jpg", name_out);  //�����Ե������ͼ
					imwrite(file_dst, orignal);
					imshow("��Ե������ͼ��", orignal);
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

//����������������麯��
float SortGroup(vector<float> VECtor, float GroupThreshold,  float part[10][10])
{
	//vector<float> VectorCrop = VECtor;  //����VECtord��һ������
	sort(VECtor.begin(), VECtor.end());  //����
	memset(part, 0, sizeof(part)); //������������0
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

//��תͼ�����ݲ��䣬�ߴ���Ӧ���  
IplImage* rotateImage1(IplImage* img,int degree){  
    double angle = degree  * CV_PI / 180.; // ����    
    double a = sin(angle), b = cos(angle);   
    int width = img->width;    
    int height = img->height;    
    int width_rotate= int(height * fabs(a) + width * fabs(b));    
    int height_rotate=int(width * fabs(a) + height * fabs(b));    
    //��ת����map  
    // [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]  
    // [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]  
    float map[6];  
    CvMat map_matrix = cvMat(2, 3, CV_32F, map);    
    // ��ת����  
    CvPoint2D32f center = cvPoint2D32f((width+1) / 2,( height + 1) / 2);    
    cv2DRotationMatrix(center, degree, 1.0, &map_matrix);    
    map[2] += (width_rotate - width) / 2;    
    map[5] += (height_rotate - height) / 2;    
    IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate), 8, 3);   
    //��ͼ��������任  
    //CV_WARP_FILL_OUTLIERS - ����������ͼ������ء�  
    //�������������������ͼ��ı߽��⣬��ô���ǵ�ֵ�趨Ϊ fillval.  
    //CV_WARP_INVERSE_MAP - ָ�� map_matrix �����ͼ������ͼ��ķ��任��  
    cvWarpAffine( img,img_rotate, &map_matrix, CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS, cvScalarAll(0));    
    return img_rotate;  
}  

//���Զ�̬��������
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

//����תͼ��ԭͼ��ĵ���任
Point OppositeMap(Point point_in,float cosa, float sina, Mat image_in, Mat image_rota) {
	Point point;
	point.x = point_in.x * cosa - point_in.y * sina - cosa*image_rota.cols / 2 + sina*image_rota.rows / 2 + image_in.cols / 2;
	point.y = point_in.x * sina + point_in.y * cosa - sina*image_rota.cols / 2 - cosa*image_rota.rows / 2 + image_in.rows / 2;
	return point;
}