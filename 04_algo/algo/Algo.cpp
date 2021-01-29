// Copyright 2019 Jeffrey Hao. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =============================================================================

#include "Algo.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace spek_model;


// 
// Color dimension reduction.
//  
Mat3b Algo::fnProcessImgByKmean(const Mat3b img, const int K)
{
	if (K < 2)
	{
		// Invalid K, do nothing.
		return img;
	}

	Mat img_copy = img.clone();
	Mat rgbChannels[3];
	split(img, rgbChannels);
	double minVal_0 = 0.0, maxVal_0 = 0.0;
	double minVal_1 = 0.0, maxVal_1 = 0.0;
	double minVal_2 = 0.0, maxVal_2 = 0.0;
	minMaxIdx(rgbChannels[0], &minVal_0, &maxVal_0);
	minMaxIdx(rgbChannels[1], &minVal_1, &maxVal_1);
	minMaxIdx(rgbChannels[2], &minVal_2, &maxVal_2);
	// cout << minVal_0 << ", " << maxVal_0 << endl <<  
	// 	minVal_1 << ", " << maxVal_1 << endl <<  
	// 	minVal_2 << ", " << maxVal_2 << endl << endl;
	if (minVal_0 == maxVal_0 && 
			minVal_1 == maxVal_1 && 
			minVal_2 == maxVal_2)
	{
		return img;
	}

	// 0. Prepare arguments for kmeans.
	cv::Mat reshaped_img = img_copy.reshape(1, img.cols * img.rows);

	cv::Mat reshaped_img32f, labels, centers;
	reshaped_img.convertTo(reshaped_img32f, CV_32FC1, 1.0 / 255.0);

	// 1. do kmeans
	cv::kmeans(reshaped_img32f, K, labels,
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
			3, KMEANS_PP_CENTERS, centers);

	// 2. convert to rgb mat
	cv::Mat rgb_img(img.rows, img.cols, CV_8UC3);
	cv::MatIterator_<cv::Vec3b> rgb_first  = rgb_img.begin<cv::Vec3b>();
	cv::MatIterator_<cv::Vec3b> rgb_last   = rgb_img.end<cv::Vec3b>();
	cv::MatConstIterator_<int> label_first = labels.begin<int>();

	cv::Mat centers_u8;
	centers.convertTo(centers_u8, CV_8UC1, 255.0);
	cv::Mat centers_u8c3 = centers_u8.reshape(3);

	while (rgb_first != rgb_last)
	{
		const cv::Vec3b &rgb = centers_u8c3.ptr<cv::Vec3b>(*label_first)[0];
		*rgb_first = rgb;
		++rgb_first;
		++label_first;
	}

	return rgb_img;
}


//
// Calculate hist of gray image.
//
Mat Algo::fnGetHist4GrayImage(const Mat1b gray)
{
	// 1. Set histogram bins count
	const int bins = 256;
	const int histSize[] = {bins};

	// 2. Set ranges for histogram bins
	const float lranges[] = {0, bins};
	const float *ranges[] = {lranges};

	// 3. Create matrix for histogram
	cv::Mat hist;
	int channels[] = {0};

	// 4. Create matrix for histogram visualization
	cv::calcHist(&gray, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

	// 5. Debug: draw for each channel
	for (int i = 0; i < bins; i++)
	{
		if (hist.at<float>(i) >= 1)
		{
			cout << i << " : " << hist.at<float>(i) << endl;
		}
	}

	return hist;
}


// 
// zoom in without blur
// 
bool Algo::fnNaiveResize(const Mat gray, const int scale, Mat& output)
{
	if (gray.cols*scale != output.cols || 
			gray.rows*scale != output.rows)
	{
		return false;
	}

	for (int row = 0; row < output.rows; row++)
	{
		for (int col = 0; col < output.cols; col++)
		{
			output.at<uchar>(row, col) = gray.at<uchar>(row/scale, col/scale);
			// cout << (int)output.at<uchar>(row, col) << ",";
		}
		// cout << endl;
	}
	// cout << endl;

	return true;
}


// 
// Transform image to edges through sobel operator.
//
int Algo::fnSobelDetection(const Mat1b img_gray, Mat1b& output)
{
	// [sobel].
 	int ksize = 1;
 	int scale = 1;
 	int delta = 0;
 	int ddepth = CV_16S;

	Mat3b grad;
   	Mat grad_x, grad_y;
   	Mat abs_grad_x, abs_grad_y;

   	// Gradient X, then converting back to CV_8U
   	Sobel(img_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
   	convertScaleAbs(grad_x, abs_grad_x);
   	// Gradient Y, then converting back to CV_8U
   	Sobel(img_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
   	convertScaleAbs(grad_y, abs_grad_y);

   	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, output);

	return 0;
}


// 
// Transform image to edges through Laplacian operator.
//
int Algo::fnLaplacianDetection(const Mat1b img_gray, Mat1b &output)
{
	Laplacian(img_gray, output, CV_8UC1);

	return 0;
}


// 
// This is for debugging.
//
int Algo::fnGrayHist(const Mat1b gray)
{
	/// Establish the number of bins
	const int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	const float range[] = {0, histSize};
	const float *histRange = {range};

	bool uniform = true;
	bool accumulate = false;

	Mat hist;

	/// Compute the histograms:
	calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	// Draw the histograms for B, G and R
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			 Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
			 Scalar(255, 0, 0), 2, 8, 0);
	}

	/// Display
	// namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);

	return 0;
}


//
// Find the most light color pixel.
//
int Algo::fnFindLightestColor(const Mat1b gray)
{
	int backgroundGrayColor = 255;

	// 1. Set histogram bins count
	int bins = 256;
	int histSize[] = {bins};

	// 2. Set ranges for histogram bins
	float lranges[] = {0, 256};
	const float *ranges[] = {lranges};

	// 3. create matrix for histogram
	cv::Mat hist;
	int channels[] = {0};

	// 4. create matrix for histogram visualization
	cv::calcHist(&gray, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

	// 5. background color should be the most light Gray.
	for (int b = bins - 1; b > 0; b--)
	{
		if (0 != hist.at<float>(b))
		{
			backgroundGrayColor = b;
			break;
		}
	}

	return backgroundGrayColor;
}


//
// Find the most dark color pixel.
//
int Algo::fnFindDarkestColor(const Mat1b gray)
{
	int darkestColor = 0;

	// 1. Set histogram bins count
	int bins = 256;
	int histSize[] = {bins};

	// 2. Set ranges for histogram bins
	float lranges[] = {0, 256};
	const float *ranges[] = {lranges};

	// 3. create matrix for histogram
	cv::Mat hist;
	int channels[] = {0};

	// 4. create matrix for histogram visualization
	cv::calcHist(&gray, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

	// 5. tag color should be the most dark Gray.
	for (int b = 0; b < bins; b++)
	{
		if (0 != hist.at<float>(b))
		{
			darkestColor = b;
			break;
		}
	}

	return darkestColor;
}


//
// Threshold: transform image to black-whtie image.
//
int Algo::fnAutoThreshold(const Mat1b gray, Mat1b &output)
{
	Mat3b img_knn;
	cvtColor(gray, img_knn, COLOR_GRAY2BGR);
	img_knn = fnProcessImgByKmean(img_knn, KNN_BINARY);
	Mat1b gray_knn;
	cvtColor(img_knn, gray_knn, COLOR_BGR2GRAY);

	// fnGrayHist(gray_knn);

	int darkest_color = fnFindDarkestColor(gray_knn);
	threshold(gray_knn, output, darkest_color, 255, THRESH_BINARY);
	// imshow("debug 01", gray_knn);
	// imshow("debug 02", output);
	// waitKey(0);

	return 0;
}

//
// Threshold: transform image to black-whtie image.
//
int Algo::fnKnnCloseOperation(const int blur_core, const Mat1b gray, Mat1b &output)
{
	Mat3b img_knn;
	cvtColor(gray, img_knn, COLOR_GRAY2BGR);
 	GaussianBlur(img_knn, img_knn, cv::Size(blur_core, blur_core), 0, 0, BORDER_DEFAULT);
	img_knn = fnProcessImgByKmean(img_knn, KNN_TRIPLE);
	Mat1b gray_knn;
	cvtColor(img_knn, gray_knn, COLOR_BGR2GRAY);

	// fnGrayHist(gray_knn);

	int darkest_color = fnFindDarkestColor(gray_knn);
	threshold(gray_knn, output, darkest_color, 255, THRESH_BINARY);

	return 0;
}


//
// return the difference value of two color. 
//
int Algo::fnGrayDiffer(Mat1b knnGray)
{
	Mat hist = Algo::fnGetHist4GrayImage(knnGray);
	const int bins = 256;

	vector<int> vecColor;
	for (int b = bins - 1; b >= 0; b--)
	{
		if (0 != hist.at<float>(b))
		{
			// dcout << "[gray] " << b << endl;
			vecColor.push_back(b);
		}
	}

	if (vecColor.size() < 2) return -1;

	// dcout << "[color diff] " << abs(vecColor.at(0)-vecColor.at(1)) << endl;
	return abs(vecColor.at(0) - vecColor.at(1));
}

// @brief: calculate the variance value except mask area.
// @param[in]: gray image.
// @param[in]: mask.
// @ret: variance value
// @birth: n/a
double Algo::fnCalGrayVarianceWithMask(const Mat1b gray, const Mat1b mask)
{
	Mat mat_mean, mat_sd; 
    meanStdDev(gray, mat_mean, mat_sd);  
    // double rst_m  = mat_mean.at<double>(0,0);  
    double rst_sd = mat_sd.at<double>(0,0);  

    // dcout << "\t\t\tMean: " << rst_m << " , StdDev: " << rst_sd << endl;  
	return rst_sd;
}

// @brief: calculate the mean value except mask area.
// @param[in]: gray image.
// @param[in]: mask.
// @ret: mean value
// @birth: n/a
double Algo::fnCalGrayMeanWithMask(const Mat1b gray, const Mat1b mask)
{
	Mat mat_mean, mat_sd; 
    meanStdDev(gray, mat_mean, mat_sd);  
    double rst_m  = mat_mean.at<double>(0,0);  
    // double rst_sd = mat_sd.at<double>(0,0);  

    // dcout << "\t\t\tMean: " << rst_m << " , StdDev: " << rst_sd << endl;  
	return rst_m;
}

// @brief: binary classification
//	Except the black background, is it the same color of both images according to Variance of single channel?
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[in]: threshold
// @param[in]: title for print log.
// @ret: yes/no.
// @birth: n/a
bool Algo::fnisValidSingleChannelVar(const Mat1b in_roiMask, const Mat1b in_roiImg1, const Mat1b in_roiImg2, const float threshold, const string title)
{
	Mat1b single_roiImg1 = in_roiImg1.clone();
	Mat1b single_roiImg2 = in_roiImg2.clone();

	// slightly blur
	const int blur_core = 3;
 	GaussianBlur(single_roiImg1, single_roiImg1, cv::Size(blur_core, blur_core), 0, 0, BORDER_DEFAULT);
 	GaussianBlur(single_roiImg2, single_roiImg2, cv::Size(blur_core, blur_core), 0, 0, BORDER_DEFAULT);

#ifdef DEEP_DEBUG 
	// imshow("roiMask",  roiMask);
	imshow("roiImg1", in_roiImg1);
	imshow("roiImg2", in_roiImg2);
	// moveWindow("roiMask",  gapCol4imshow*0, gapRow4imshow*2);
	moveWindow("roiImg1", gapCol4imshow*2, gapRow4imshow*2);
	moveWindow("roiImg2", gapCol4imshow*4, gapRow4imshow*2);
#endif

	const float roiImg1_sd = (float)fnCalGrayVarianceWithMask(single_roiImg1, in_roiMask);
	const float roiImg2_sd = (float)fnCalGrayVarianceWithMask(single_roiImg2, in_roiMask);

	//
	// if larger than threshold, it's not single color, which should be removed.
	// so, larger threshold --> more 'true' less 'false' --> more strict --> less avaliable areas.
	//
	if (threshold < roiImg1_sd)
	{
		icout << "\t\t\tAlgo --> " << title << ": is single color? [false]: (threshold) " << threshold << " ? (variance1) " << roiImg1_sd << endl;
		return false;
	}

	if (threshold < roiImg2_sd)
	{
		icout << "\t\t\tAlgo --> " << title << ": is single color? [false]: (threshold) " << threshold << " ? (variance2) " << roiImg2_sd << endl;
		return false;
	}

	icout << "\t\t\tAlgo --> " << title << ": is single color? [true] : (threshold) " << threshold << " ? (variance1) " << roiImg1_sd << ", (variance2) " << roiImg2_sd << endl;
	return true;
}


// @brief: binary classification
//	Except the black background, is it the same color of both images according to mean difference on single channel?
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[in]: threshold
// @ret: yes/no.
// @birth: n/a
bool Algo::fnisValidSingleChannelMean(const Mat1b in_roiMask, const Mat1b in_roiImg1, const Mat1b in_roiImg2, const float threshold, const string title)
{
	Mat1b single_roiImg1 = in_roiImg1.clone();
	Mat1b single_roiImg2 = in_roiImg2.clone();

	// slightly blur
	const int blur_core = 3;
 	GaussianBlur(single_roiImg1, single_roiImg1, cv::Size(blur_core, blur_core), 0, 0, BORDER_DEFAULT);
 	GaussianBlur(single_roiImg2, single_roiImg2, cv::Size(blur_core, blur_core), 0, 0, BORDER_DEFAULT);

#ifdef DEEP_DEBUG 
	// imshow("roiMask",  roiMask);
	imshow("roiImg1", in_roiImg1);
	imshow("roiImg2", in_roiImg2);
	// moveWindow("roiMask",  gapCol4imshow*0, gapRow4imshow*2);
	moveWindow("roiImg1", gapCol4imshow*2, gapRow4imshow*2);
	moveWindow("roiImg2", gapCol4imshow*4, gapRow4imshow*2);
#endif

	const float roiImg1_m = (float)fnCalGrayMeanWithMask(single_roiImg1, in_roiMask);
	const float roiImg2_m = (float)fnCalGrayMeanWithMask(single_roiImg2, in_roiMask);
	const float roiImgDiff_m = abs(roiImg1_m - roiImg2_m);

	//
	// if mean difference is too large, it's not single color.
	// so, larger threshold --> more 'true' less 'false' --> more strict --> less avaliable areas.
	if (threshold < roiImgDiff_m)
	{
		icout << "\t\t\tAlgo --> " << title << ": is single color? [false]: (threshold) " << threshold << " ? (mean diff) " << roiImgDiff_m << endl;
		return false;
	}

	icout << "\t\t\tAlgo --> " << title << ": is single color? [true] : (threshold) " << threshold << " ? (mean diff) " << roiImgDiff_m << endl;
	return true;
}


// @brief: binary classification
//	compared with the bndbox, the scare should be tiny.
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[in]: min threshold
// @param[in]: max threshold
// @ret: yes/no.
// @birth: n/a
bool Algo::isValidAreaBndboxRatio(const Mat1b in_mask, const Mat3b in_roiImg1, const Mat3b in_roiImg2, const float minThreshold, const float maxThreshold)
{
	// This is to remove the long edges after edge detection.
	const int bndbox_area = in_roiImg1.rows*in_roiImg1.cols;
	const int area = countNonZero(in_mask);
	float actualAreaBndboxRatio = (1.0*area)/bndbox_area;

	if (actualAreaBndboxRatio > maxThreshold || actualAreaBndboxRatio < minThreshold)
	{
		icout << "\t\t\tAlgo --> is valid areaBndboxRatio? [false]: actualAreaBndboxRatio = " << actualAreaBndboxRatio << ", threshold is [" << minThreshold << ", " << maxThreshold << "]" << endl;
		return false;
	}

	icout << "\t\t\tAlgo --> is valid areaBndboxRatio? [true]: actualAreaBndboxRatio = " << actualAreaBndboxRatio << ", threshold is [" << minThreshold << ", " << maxThreshold << "]" << endl;
	return true;
}

// @brief: binary classification
//	Except the black background, is it the same color of both images according to variance of hue channel?
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[in]: threshold
// @ret: yes/no.
// @birth: n/a
bool Algo::isSingleColorArea_var(const Mat1b in_roiMask, const Mat3b in_roiImg1, const Mat3b in_roiImg2, const float threshold)
{
	Mat roiGray1, roiGray2;
	cvtColor(in_roiImg1, roiGray1, CV_BGR2GRAY);
	cvtColor(in_roiImg2, roiGray2, CV_BGR2GRAY);
	const int blur_core = 3;
 	GaussianBlur(roiGray1, roiGray1, cv::Size(blur_core, blur_core), 0, 0, BORDER_DEFAULT);
 	GaussianBlur(roiGray2, roiGray2, cv::Size(blur_core, blur_core), 0, 0, BORDER_DEFAULT);

	return fnisValidSingleChannelVar(in_roiMask, roiGray1, roiGray2, threshold, "Gray Variance");
}

// @brief: binary classification
//	Except the black background, is it the same color of both images according to difference on hue mean?
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[in]: threshold
// @ret: yes/no.
// @birth: n/a
bool Algo::isSingleColorArea_mean(const Mat1b in_roiMask, const Mat3b in_roiImg1, const Mat3b in_roiImg2, const float threshold)
{
	Mat roiGray1, roiGray2;
	cvtColor(in_roiImg1, roiGray1, CV_BGR2GRAY);
	cvtColor(in_roiImg2, roiGray2, CV_BGR2GRAY);
	const int blur_core = 3;
 	GaussianBlur(roiGray1, roiGray1, cv::Size(blur_core, blur_core), 0, 0, BORDER_DEFAULT);
 	GaussianBlur(roiGray2, roiGray2, cv::Size(blur_core, blur_core), 0, 0, BORDER_DEFAULT);

	return fnisValidSingleChannelMean(in_roiMask, roiGray1, roiGray2, threshold, "Gray Mean   ");
}

// @brief: binary classification
//	Except the black background, is it the same color of both images according to variance of hue channel?
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[in]: threshold
// @ret: yes/no.
// @birth: n/a
bool Algo::isSingleHueArea_var(const Mat1b in_roiMask, const Mat3b in_roiImg1, const Mat3b in_roiImg2, const float threshold)
{
	// channels[0], channels[1], channels[2] will contain your H, S, V respectively.
	Mat3b hsv_roiImg1, hsv_roiImg2;
	cvtColor(in_roiImg1, hsv_roiImg1, COLOR_BGR2HSV);
	cvtColor(in_roiImg2, hsv_roiImg2, COLOR_BGR2HSV);

	vector<Mat> channels;
	split(hsv_roiImg1, channels);
	Mat1b hue_roiImg1 = channels[0].clone();
	split(hsv_roiImg2, channels);
	Mat1b hue_roiImg2 = channels[0].clone();

	return fnisValidSingleChannelVar(in_roiMask, hue_roiImg1, hue_roiImg2, threshold, "Hue Variance");
}

// @brief: binary classification
//	Except the black background, is it the same color of both images according to difference on hue mean?
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[in]: threshold
// @ret: yes/no.
// @birth: n/a
bool Algo::isSingleHueArea_mean(const Mat1b in_roiMask, const Mat3b in_roiImg1, const Mat3b in_roiImg2, const float threshold)
{
	// channels[0], channels[1], channels[2] will contain your H, S, V respectively.
	Mat3b hsv_roiImg1, hsv_roiImg2;
	cvtColor(in_roiImg1, hsv_roiImg1, COLOR_BGR2HSV);
	cvtColor(in_roiImg2, hsv_roiImg2, COLOR_BGR2HSV);

	vector<Mat> channels;
	split(hsv_roiImg1, channels);
	Mat1b hue_roiImg1 = channels[0].clone();
	split(hsv_roiImg2, channels);
	Mat1b hue_roiImg2 = channels[0].clone();

	return fnisValidSingleChannelMean(in_roiMask, hue_roiImg1, hue_roiImg2, threshold, "Gray Mean   ");
}

// @brief: binary classifier only need small interest areas, but not the whole picture.
// 	return the number of contours can be found.
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[out]: roi of image
// @param[out]: roi of its pair
// @param[out]: roi of mask
// @param[out]: Rect of roi
// @param[in]: use mask or not.
// @ret: number of countours found.
// @birth: n/a
int Algo::fnGetRoiPair4Classifier(const Mat1b mask, const Mat3b img1, const Mat3b img2, 	\
	Mat3b& output_roiImg1, Mat3b& output_roiImg2, Mat1b& output_roiMask, Rect& output_rect, const bool use_mask)
{
	// (1) is it a valid mask?
	if (0 == countNonZero(mask)) {
		// it is a pure black mask, there is no interest area.
		return 0;
	}

	// (2) find the valid areas to check later.
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	if (contours.size() < 1) {
		return contours.size();
	}

	// (3) traverse each contour and then calculate a bigger one holding whole of them.
	std::vector<Rect> boundRect(contours.size());
	int rst_min_x = -1;
	int rst_min_y = -1;
	int rst_max_x = 0;
	int rst_max_y = 0;

	for (int i = 0; i < (int)contours.size(); i++)
	{
		boundRect[i] = boundingRect(contours[i]);

		int min_x = boundRect[i].tl().x;
		rst_min_x = (rst_min_x < min_x)&&(rst_min_x>=0) ? rst_min_x : min_x;
		int min_y = boundRect[i].tl().y;
		rst_min_y = (rst_min_y < min_y)&&(rst_min_y>=0) ? rst_min_y : min_y;

		int max_x = boundRect[i].br().x;
		rst_max_x = rst_max_x > max_x ? rst_max_x : max_x;
		int max_y = boundRect[i].br().y;
		rst_max_y = rst_max_y > max_y ? rst_max_y : max_y;

		if (rst_min_x < 0 || rst_min_y < 0 || rst_max_x > mask.cols || rst_max_y > mask.rows)
		{	
			ecout << "Error! Out of bound in " << __FUNCTION__ << endl;
			ecout << rst_min_x << " < 0 || " << rst_min_y << " < 0 || " << rst_max_x << " >= " << mask.cols << " || " << rst_max_y << " >= " << mask.rows << endl;

			// debug, green bounding box.
			// rectangle(img1, boundRect[i], Scalar(0, 255, 0));
			// rectangle(img2, boundRect[i], Scalar(0, 255, 0));
			// imshow("img1 box", img1);
			// imshow("img2 box", img2);
			// imshow("mask", mask);
			// waitKey(0);

			return -1;
		}
	}

	// (4) get a new whole boudning box covering all areas: Rect var(x, y, width, height);
	Rect rst_boundRect = Rect(rst_min_x, rst_min_y, abs(rst_max_x-rst_min_x), abs(rst_max_y-rst_min_y));
	output_rect = rst_boundRect;

	// debug.
	// cout << "debug --> " << rst_boundRect << endl;
	// rectangle(img1, rst_boundRect, Scalar(0, 255, 0));
	// rectangle(img2, rst_boundRect, Scalar(0, 255, 0));
	// imshow("img1 box", img1);
	// imshow("img2 box", img2);
	// imshow("mask", mask);

	Mat3b roiImg1, roiImg2;
	mask(rst_boundRect).copyTo(output_roiMask);
	img1(rst_boundRect).copyTo(roiImg1);
	img2(rst_boundRect).copyTo(roiImg2);

	if (use_mask == true)
	{
		roiImg1.copyTo(output_roiImg1, output_roiMask);
		roiImg2.copyTo(output_roiImg2, output_roiMask);
	}
	else
	{
		roiImg1.copyTo(output_roiImg1);
		roiImg2.copyTo(output_roiImg2);
	}

	// This should be only 1 contour.
	return 1;
}

// @brief: separate the large one into small pieces
// @param[in]: mask
// @param[in]: image.
// @param[in]: its pair
// @param[in]: number of rows
// @param[in]: number of cols
// @param[in]: class names
// @param[out]: small mask list after separation.
// @ret: 0 (success)
// @birth: n/a
int Algo::fnSeparateLargeRoiMask(const Mat1b roi_mask, const Mat3b pristine, const Mat3b pristine_partner, \
	const int c_iDiv_row, const int c_iDiv_col, vector<Mat1b>& output)
{
	dcout << __FUNCTION__ << ", " << __LINE__ << endl;

	Mat1b c_img = roi_mask.clone();

	int iGap_row = c_img.rows / c_iDiv_row;
	int iGap_col = c_img.cols / c_iDiv_col;

	for (int idxRow = 0; idxRow < c_iDiv_row; idxRow++)
	{
		for (int idxCol = 0; idxCol < c_iDiv_col; idxCol++)
		{
			int segmentWidth  = iGap_col;
			int segmentHeight = iGap_row;

			if (idxRow == c_iDiv_row - 1)
			{
				segmentHeight = c_img.rows - idxRow * iGap_row;
			}
			if (idxCol == c_iDiv_col - 1)
			{
				segmentWidth = c_img.cols - idxCol * iGap_col;
			}
			Rect RectSegment(idxCol * iGap_col, idxRow * iGap_row, segmentWidth, segmentHeight);

			Mat1b output_mask = Mat(roi_mask.rows, roi_mask.cols, CV_8UC1, Scalar(0));
			roi_mask(RectSegment).copyTo(output_mask(RectSegment));
			output.push_back(output_mask);
		}
	}

	return 0;
}

// @brief: overlay partial mask into one.
// @param[in]: partial mask list.
// @param[out]: overlay result.
// @ret: void
// @birth: v0.98
void fnOverlayMasks(const vector<Mat1b> vecMask, const Mat1b &mask_out)
{
	for(auto mask: vecMask)
	{
		bitwise_or(mask, mask_out, mask_out);
	}
}

// @brief: [debug] show the debugging charts.
// @param[in]: image 1
// @param[in]: image 2
// @param[in]: contourLine
// @param[in]: save folder path.
// @param[in/out]: iLabel in option
// @ret: void
// @birth: v0.86
void Algo::fnThermodynamicChart(const Mat3b image1, const Mat3b image2, const Mat1b prob_mask, string const sSaveFolderPath, int *p_index)
{
	Mat3b pristine = image1.clone();
	Mat3b pristine_partner = image2.clone();
	Mat1b contourLines = prob_mask.clone();

	// Thermodynamic Chart
	Mat3b resultRGB = Mat(contourLines.rows, contourLines.cols, CV_8UC3, Scalar(0,0,0));

	// scan every image pixel
	for (int i = 0; i < contourLines.rows; i++)
	{
		for (int j = 0; j < contourLines.cols; j++)
		{
			int grayPixel = contourLines.at<uchar>(i, j);
			if (grayPixel == 0) 
				continue;

			int r = grayPixel;
			int g = 0;
			int b = 0;
			// int b = 255 - grayPixel;

			resultRGB.at<Vec3b>(i, j)[0] = b;
			resultRGB.at<Vec3b>(i, j)[1] = g;
			resultRGB.at<Vec3b>(i, j)[2] = r;
		}
	}

	const float photo_alpha = 0.5;
	resize(pristine, pristine, cv::Size(resultRGB.cols, resultRGB.rows));
	addWeighted(resultRGB, 1-photo_alpha, pristine, photo_alpha, 0, pristine);
	// imshow("pristine", pristine);

	resize(pristine_partner, pristine_partner, cv::Size(resultRGB.cols, resultRGB.rows));
	addWeighted(resultRGB, 1-photo_alpha, pristine_partner, photo_alpha, 0, pristine_partner);
	// imshow("pristine partner", pristine_partner);
	// waitKey(0);
 
	Mat img_merge;
	Mat outImg_left, outImg_right;
	cv::Size size(pristine.cols + pristine_partner.cols, MAX(pristine.rows, pristine_partner.rows));
 
	img_merge = Mat(size, CV_8UC3);
	outImg_left  = img_merge(Rect(0, 0, pristine.cols, pristine.rows));
	outImg_right = img_merge(Rect(pristine.cols, 0, pristine_partner.cols, pristine_partner.rows));
 
	pristine.copyTo(outImg_left);
	pristine_partner.copyTo(outImg_right);

	if (sSaveFolderPath != "" && p_index != nullptr)
	{
		*p_index += 1;

		string savePath = spek_fs::fnPathJoin(sSaveFolderPath, to_string(*p_index) + "_ThermodynamicChart.jpg");
		imwrite(savePath, img_merge);
	}

	int clk = clock();
	string title = "debug: img_merge interest areas." + to_string(clk);
	// imshow(title, img_merge);

	return;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// CNN api.
///////////////////////////////////////////////////////////////////////////////////////////////////

// cnn tool api for bndbox.
bool CNNBndboxFilter_isMoreSquare(const Bndbox &aBndbox, const Bndbox &bBndbox)
{
	int aLeft = aBndbox.left;
	int aRight = aBndbox.right;
	int aTop = aBndbox.top;
	int aBottom = aBndbox.bottom;

	int bLeft = bBndbox.left;
	int bRight = bBndbox.right;
	int bTop = bBndbox.top;
	int bBottom = bBndbox.bottom;

	float aRate, bRate;
	aRate = ((aRight - aLeft) > (aBottom - aTop)) ? ((float)(aRight - aLeft) / (aBottom - aTop)) : ((float)(aBottom - aTop) / (aRight - aLeft));
	bRate = ((bRight - bLeft) > (bBottom - bTop)) ? ((float)(bRight - bLeft) / (bBottom - bTop)) : ((float)(bBottom - bTop) / (bRight - bLeft));

	return aRate < bRate;
}

// cnn tool api for bndbox.
bool CNNBndboxFilter_hasOverlap(const Bndbox &aBndbox, const Bndbox &bBndbox)
{
	int aLeft = aBndbox.left;
	int aRight = aBndbox.right;
	int aTop = aBndbox.top;
	int aBottom = aBndbox.bottom;

	int bLeft = bBndbox.left;
	int bRight = bBndbox.right;
	int bTop = bBndbox.top;
	int bBottom = bBndbox.bottom;

	return !((aRight <= bLeft) || (bRight <= aLeft) || (aBottom <= bTop) || (bBottom <= aTop));
}

int CNNBndboxFilter_rmOverlap(std::vector<Bndbox> &vecBndbox)
{
	if (0 == vecBndbox.size())
	{
		return 0;
	}

	for (int i = vecBndbox.size() - 1; i >= 0; i--)
	{
		// Bndbox bndbox = vecBndbox[i];
		// int left = bndbox.left;
		// int right = bndbox.right;
		// int top = bndbox.top;
		// int bottom = bndbox.bottom;

		for (int j = 0; j < i; j++)
		{
			// Bndbox eachBndbox = vecBndbox[j];
			// int eachLeft = eachBndbox.left;
			// int eachRight = eachBndbox.right;
			// int eachTop = eachBndbox.top;
			// int eachBottom = eachBndbox.bottom;

			//If it overlaps any othera, eliminate less square one.
			if (CNNBndboxFilter_hasOverlap(vecBndbox[i], vecBndbox[j]) && (!CNNBndboxFilter_isMoreSquare(vecBndbox[i], vecBndbox[j])))
			{
				vecBndbox.erase(vecBndbox.begin() + i);
				break;
			}
		}
	}

	return 0;
}

int CNNBndboxFilter_rmRedundant(std::vector<Bndbox> &vecBndbox)
{
	if (0 == vecBndbox.size())
	{
		return 0;
	}

	sort(vecBndbox.begin(), vecBndbox.end(), CNNBndboxFilter_isMoreSquare);

	for (int i = vecBndbox.size() - 1; i > 0; i--)
	{
		Bndbox bndbox = vecBndbox[i];

		int left = bndbox.left;
		int right = bndbox.right;
		int top = bndbox.top;
		int bottom = bndbox.bottom;
		int pos_centroidX = left + (right - left) / 2;
		int pos_centroidY = top + (bottom - top) / 2;

		for (int j = 0; j < i; j++)
		{
			Bndbox eachBndbox = vecBndbox[j];

			int eachLeft = eachBndbox.left;
			int eachRight = eachBndbox.right;
			int eachTop = eachBndbox.top;
			int eachBottom = eachBndbox.bottom;

			// If its centroid is in any 'more square' bnd box, eliminate it.
			if (pos_centroidX < eachRight && pos_centroidX > eachLeft &&
				pos_centroidY < eachBottom && pos_centroidY > eachTop)
			{
				vecBndbox.erase(vecBndbox.begin() + i);
				break;
			}
		}
	}

	return 0;
}

int CNNBndboxFilter(std::vector<Bndbox> &vecBndbox)
{
	/**
	 * Add new filter here.
	 * Each filter should be independent. 
	 */
	CNNBndboxFilter_rmRedundant(vecBndbox);
	CNNBndboxFilter_rmOverlap(vecBndbox);

	return 0;
}


bool CNNLoadModelContent(string model_path)
{
	ifstream file_handle(model_path, ios::in | ios::binary);
	if (!file_handle.is_open())
	{
		ecout << "failed to open " << model_path << endl;
		return false;
	}

	return true;
}

bool CNNLoadConfigContent(string config_path)
{
	ifstream file_handle(config_path, ios::in | ios::binary);
	if (!file_handle.is_open())
	{
		ecout << "failed to open " << config_path << endl;
		return false;
	}

	return true;
}

// @brief: image segmentation by unet of tf c++
// @param[in]: image
// @param[in]: bounding box confidence threshold
// @param[in]: mask confidence threshold
// @param[in]: cnn model path.
// @param[in]: cnn config path.
// @param[out]: mask output.
// @ret: number of mask
// @birth: n/a
int Algo::fnUNetScan4Segmentation(const Mat3b img, const float confBndbox_threshold, const float confMask_threshold, \
						const string model_path, const string config_path, vector<Mat1b> &out_vecMask)
{
	out_vecMask.clear();

	// (1) load tflite mobilenet
	static UnetInterface *p_unet = new UnetInterface(model_path);
	if (false == p_unet->init())
	{
		ecout << "error: cannot load unet from " << model_path << endl;
		return 0;
	}

	// (2) inference.
	int numObj = p_unet->predict(img, out_vecMask);

	// (3) post process. [invalid]
	// for(auto& mask: out_vecMask)
	// {
	// 	threshold(mask, mask, 1, 255, THRESH_BINARY);
	// }

	return numObj;
}

// @brief: image segmentation by unet of tf c++, enhanced.
// 	it will detect the whole picture, then
//	detect smaller part on left top, right top, left bottom, right bottom.
// @param[in]: image
// @param[in]: bounding box confidence threshold
// @param[in]: mask confidence threshold
// @param[in]: cnn model path.
// @param[in]: cnn config path.
// @param[out]: mask output.
// @ret: number of mask
// @birth: n/a
int Algo::fnUNetScan4Segmentation_advanced(const Mat3b img, const float confBndbox_threshold, const float confMask_threshold, \
						const string model_path, const string config_path, vector<Mat1b> &out_vecMask)
{
	// init.
	out_vecMask.clear();
	int total_numObj = 0;

	const float thres_part = 0.5;
	Rect whole_rect        = Rect((int)0,                              (int)0,                              (int)img.cols               , (int)img.rows);
	Rect leftTop_rect      = Rect((int)0,                              (int)0,                              (int)(img.cols*thres_part-1), (int)(img.rows*thres_part-1));
	Rect rightTop_rect     = Rect((int)(img.cols-img.cols*thres_part), (int)0,                              (int)(img.cols*thres_part-1), (int)(img.rows*thres_part-1));
	Rect leftBottom_rect   = Rect((int)0,                              (int)(img.rows-img.rows*thres_part), (int)(img.cols*thres_part-1), (int)(img.rows*thres_part-1));
	Rect rightBottom_rect  = Rect((int)(img.cols-img.cols*thres_part), (int)(img.rows-img.rows*thres_part), (int)(img.cols*thres_part-1), (int)(img.rows*thres_part-1));

	Mat1b partial_mask = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));
	// whole
	vector<Mat1b> partial_vecMask;
	total_numObj += fnUNetScan4Segmentation(img(whole_rect), confBndbox_threshold, confMask_threshold, model_path, config_path, partial_vecMask);
	partial_mask = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));
	fnOverlayMasks(partial_vecMask, partial_mask(whole_rect));
	partial_mask /= 2;
	out_vecMask.push_back(partial_mask);
	// imshow("input", img(whole_rect));
	// imshow("whole", partial_mask);
	// waitKey(0);

	// left top
	partial_vecMask.clear();
	total_numObj += fnUNetScan4Segmentation(img(leftTop_rect), confBndbox_threshold, confMask_threshold, model_path, config_path, partial_vecMask);
	partial_mask = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));
	fnOverlayMasks(partial_vecMask, partial_mask(leftTop_rect));
	partial_mask /= 2;
	out_vecMask.push_back(partial_mask);
	// imshow("input", img(leftTop_rect));
	// imshow("leftTop", partial_mask);
	// waitKey(0);

	// right top
	partial_vecMask.clear();
	total_numObj += fnUNetScan4Segmentation(img(rightTop_rect), confBndbox_threshold, confMask_threshold, model_path, config_path, partial_vecMask);
	partial_mask = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));
	fnOverlayMasks(partial_vecMask, partial_mask(rightTop_rect));
	partial_mask /= 2;
	out_vecMask.push_back(partial_mask);
	// imshow("input", img(rightTop_rect));
	// imshow("rightTop", partial_mask);
	// waitKey(0);

	// left bottom
	partial_vecMask.clear();
	total_numObj += fnUNetScan4Segmentation(img(leftBottom_rect), confBndbox_threshold, confMask_threshold, model_path, config_path, partial_vecMask);
	partial_mask = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));
	fnOverlayMasks(partial_vecMask, partial_mask(leftBottom_rect));
	partial_mask /= 2;
	out_vecMask.push_back(partial_mask);
	// imshow("input", img(leftBottom_rect));
	// imshow("leftBottom", partial_mask);
	// waitKey(0);

	// right bottom
	partial_vecMask.clear();
	total_numObj += fnUNetScan4Segmentation(img(rightBottom_rect), confBndbox_threshold, confMask_threshold, model_path, config_path, partial_vecMask);
	partial_mask = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));
	fnOverlayMasks(partial_vecMask, partial_mask(rightBottom_rect));
	partial_mask /= 2;
	out_vecMask.push_back(partial_mask);
	// imshow("input", img(rightBottom_rect));
	// imshow("rightBottom", partial_mask);
	// waitKey(0);

	return total_numObj;
}

// @brief: image segmentation by mask r-cnn on opencv dnn.
// 	ref: https://docs.opencv.org/3.4/d4/d88/samples_2dnn_2segmentation_8cpp-example.html
// @param[in]: image
// @param[in]: bounding box confidence threshold
// @param[in]: mask confidence threshold
// @param[in]: cnn model path.
// @param[in]: cnn config path.
// @param[out]: output of masks.
// @ret: number of mask
// @birth: n/a
int Algo::fnMaskRCNNScan4Segmentation(const Mat3b img, const float confBndbox_threshold, const float confMask_threshold, \
						const int minEdge, const string model_path, const string config_path, vector<Mat1b> &out_vecMask)
{
	out_vecMask.clear();

	// (1) load tflite mobilenet
	static MaskrcnnInterface *p_maskrcnn = new MaskrcnnInterface(model_path, config_path, confBndbox_threshold, confMask_threshold, minEdge);
	if (false == p_maskrcnn->init())
	{
		ecout << "error: cannot load mobilenet from " << endl	\
			  << model_path << endl	\
			  << config_path << endl;
		return 0;
	}

	// (2) inference.
	int numObj = p_maskrcnn->predict(img, out_vecMask);
	
	return numObj;
}

// @brief: object classification by mobilenet of tflite api.
// 	it will call label_image.so created by tensorflow/lite/examples/label_image
// @param[in]: image
// @param[in]: confidence threshold for classification
// @param[in]: cnn lib path.
// @param[in]: cnn model path.
// @param[in]: cnn label path.
// @ret: class id
// @birth: n/a
int Algo::fnTFLiteScanForClassification(const Mat3b img, const float confidenceThreshold, const string lib_path, const string model_path, const string label_path, float &out_conf)
{
	// (1) load tflite mobilenet
	static MobilenetInterface *p_mobilenet = new MobilenetInterface(lib_path, model_path, label_path);
	if (false == p_mobilenet->init())
	{
		ecout << "error: cannot load mobilenet from " << lib_path << endl;
		return -1;
	}

	// (2) inference.
	float confidence = -1.0;
	int classId = p_mobilenet->predict(img, confidence);
	out_conf = confidence;
	return classId;
}

// @brief: object classification by mobilenet, inactive in code.
// 	Invalid, due to new solution: tflite.
// 	Ref: https://github.com/opencv/opencv/blob/master/samples/dnn/classification.cpp
// @param[in]: image
// @param[in]: threshold for classification
// @param[in]: cnn model path.
// @param[in]: cnn config path.
// @param[in]: width of input image
// @param[in]: height of input image
// @param[in]: scale factor for mobilenet
// @param[in]: mean value ofr mobilenet
// @ret: class id
// @birth: n/a
int Algo::fnCNNScanForClassification(const Mat3b img, const float confidenceThreshold, const string model_path, const string config_path, const int inWidth, const int inHeight, const float inScaleFactor, const float meanVal)
{
	static dnn::Net cls_net;
	Mat image = img.clone();

	// (1) Load cnn model.
	// if (cls_net.empty())
	{
		icout << "\t\tLoad CNN Model..." << endl;
		// if (false == CNNLoadModelContent(model_path) || 
		// 	false == CNNLoadConfigContent(config_path) )
		// {
		// 	return -1;
		// }

		cls_net = readNet(model_path, config_path);
		if (cls_net.empty())
		{
			ecout << "Err: fail to load CNN." << endl;
			return -1;
		}

		const int default_backendId = 0;
		const int default_targetId = 0;
    	cls_net.setPreferableBackend(default_backendId);
    	cls_net.setPreferableTarget(default_targetId);
	}

	bool swapRB = false;
	Mat blob;
    blobFromImage(image, blob, inScaleFactor, cv::Size(inWidth, inHeight), Scalar(meanVal, meanVal, meanVal), swapRB, false);

    cls_net.setInput(blob);
    Mat prob = cls_net.forward();
    Point classIdPoint;
    double confidence;
    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;

	// cout << __FUNCTION__ << ", " << __LINE__ << ": " << "classId is " << classId << ", confidence is " << confidence << endl;

    // Put efficiency information.
    // std::vector<double> layersTimes;
    // double freq = getTickFrequency() / 1000;
    // double t = cls_net.getPerfProfile(layersTimes) / freq;
    // std::string label = format("Inference time: %.2f ms", t);

	/* implement if necessary. */

	return classId;
}

// @brief: object detection by mobilenet-ssd, inactive in code.
// @param[in]: roi image
// @param[in]: x positoin of top left corner of roi image in original image
// @param[in]: y positoin of top left corner of roi image in origianl image
// @param[in]: threshold for bounding box confidence
// @param[in]: cnn model path.
// @param[in]: cnn config path.
// @param[in]: width of input image
// @param[in]: height of input image
// @param[in]: scale factor for mobilenet
// @param[in]: mean value ofr mobilenet
// @param[out]: bounding box info results.
// @ret: number of bounding boxes detected.
// @birth: n/a
int Algo::fnCNNScanForBndbox(const Mat3b img, const size_t pos_x, const size_t pos_y, const float confidenceThreshold, \
                             const string model_path, const string config_path, \
							 const int inWidth, const int inHeight, const float inScaleFactor, const float meanVal, std::vector<Bndbox> &vecBndbox)
{
	static dnn::Net net;
	Mat image = img.clone();

	// (1) Load cnn model.
	// if (net.empty())
	{
		icout << "\t\tLoad CNN Model..." << endl;
		if (false == CNNLoadModelContent(model_path) || 
			false == CNNLoadConfigContent(config_path) )
		{
			return -1;
		}

		net = readNetFromTensorflow(model_path, config_path);
		if (net.empty())
		{
			ecout << "Err: fail to load CNN." << endl;
			return -1;
		}
	}

	// (2) Process image and get bounding box by cnn model
	Mat inputBlob = blobFromImage(image, inScaleFactor, cv::Size(inWidth, inHeight),
								  Scalar(meanVal, meanVal, meanVal), false, false);
	net.setInput(inputBlob);
	Mat detection = net.forward();

	// (3) Save bnd box position in vector
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > confidenceThreshold)
		{
			// (3.1) Load info of bounding box.
			size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));
			int left = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
			int top = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
			int right = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
			int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);

			// (3.2) Filter out invalid bnd box
			if (left < 0 || bottom < 0 || right < 0 || top < 0 ||
				left > image.cols || bottom > image.rows || right > image.cols || top > image.rows)
			{
				continue;
			}

			// (3.3) Filter out ghost bnd box
			if (abs(right - left) * abs(bottom - top) * 3 > image.rows * image.cols)
			{
				// Ghost bnd box normally is very large.
				// 3 is an empirical value.
				continue;
			}

			Bndbox newBndbox;

			// (3.4) Transform relative position (in left/right/top/bottom frame) to absolute position ( in whole frame)
			newBndbox.objectClass = objectClass;
			newBndbox.fTagScore = confidence;
			newBndbox.left = left + pos_x;
			newBndbox.right = right + pos_x;
			newBndbox.top = top + pos_y;
			newBndbox.bottom = bottom + pos_y;

			vecBndbox.push_back(newBndbox);
		}
	}

	// sort by confidence.
	if (vecBndbox.size() > 1)
	{
		sort(vecBndbox.begin(), vecBndbox.end(), 
 			[](const Bndbox& a, const Bndbox& b)
            {
                return a.fTagScore > b.fTagScore;
            });
	}
	if (vecBndbox.size() > 0)
	{
		ncout << "\t\t\t" << __FUNCTION__ << "() thinks it is probably --> " << vecBndbox.at(0).objectClass << endl;
	}

	return vecBndbox.size();
}

// @brief: binary classifier by opencv dnn
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[in]: threshold for bounding box confidence
// @param[in]: cnn model path.
// @param[in]: cnn config path.
// @param[in]: width of input image
// @param[in]: height of input image
// @param[in]: scale factor for mobilenet
// @param[in]: mean value ofr mobilenet
// @ret: if bndbox detected, return true;
// @birth: n/a
bool Algo::isCNNInterestObject(const Mat1b roiMask, const Mat3b roiImg1, const Mat3b roiImg2, const float confidenceThreshold, \
	const string model_path, const string config_path, const int inWidth, const int inHeight, const float inScaleFactor, const float meanVal)
{
	dcout << "\t\t" << __FUNCTION__ << "(): start." << endl;

	size_t pos_x = 0;	// as default.
	size_t pos_y = 0;

	std::vector<Bndbox> vecBndbox;

#ifdef DEEP_DEBUG
	imshow("CNN: roiMask", roiMask);
	imshow("CNN: roiImg1", roiImg1);
	imshow("CNN: roiImg2", roiImg2);
#endif


	vecBndbox.clear();
	if (0 < Algo::fnCNNScanForBndbox(roiImg1, pos_x, pos_y, confidenceThreshold, model_path, config_path, inWidth, inHeight, inScaleFactor, meanVal, vecBndbox))
	{
		return true;
	}

	vecBndbox.clear();
	if (0 < Algo::fnCNNScanForBndbox(roiImg2, pos_x, pos_y, confidenceThreshold, model_path, config_path, inWidth, inHeight, inScaleFactor, meanVal, vecBndbox))
	{
		return true;
	}

	return false;
}

// @brief: image segmentation by opencv dnn
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[in]: threshold for bounding box confidence
// @param[in]: threshold for mask
// @param[in]: min edge for prediction input limitation.
// @param[in]: cnn model path.
// @param[in]: cnn config path.
// @param[out]: mask result
// @ret: if mask and bndbox detected, return true;
// @birth: n/a
bool Algo::MASKRCNNDetectSegment(const Mat1b roiMask, const Mat3b roiImg1, const Mat3b roiImg2, const float fConfBndbox_threshold, 	\
	const float fConfMask_threshold, const int iMinEdge, const string sModel_path, const string sConfig_path, Mat1b& output)
{
	dcout << "\t\t" << __FUNCTION__ << "(): start." << endl;

#ifdef DEEP_DEBUG
	imshow("CNN: roiMask", roiMask);
	imshow("CNN: roiImg1", roiImg1);
	imshow("CNN: roiImg2", roiImg2);
#endif

	bool bRst = false;
	Mat1b output_unionsection = Mat(roiMask.rows, roiMask.cols, CV_8UC1, Scalar(0));

	vector<Mat1b> vMask_img1, vMask_img2;
	Algo::fnMaskRCNNScan4Segmentation(roiImg1, fConfBndbox_threshold, fConfMask_threshold, iMinEdge, sModel_path, sConfig_path, vMask_img1);
	for (auto img: vMask_img1)
	{
		bRst = true;
		bitwise_or(img, output_unionsection, output_unionsection);
	}
	Algo::fnMaskRCNNScan4Segmentation(roiImg2, fConfBndbox_threshold, fConfMask_threshold, iMinEdge, sModel_path, sConfig_path, vMask_img2);
	for (auto img: vMask_img2)
	{
		bRst = true;
		bitwise_or(img, output_unionsection, output_unionsection);
	}

	bitwise_or(output_unionsection, output, output);
	
	// post process.
	Algo::fnAutoThreshold(output, output);
	bitwise_and(roiMask, output, output);

	return bRst;
}

// @brief: image segmentation by tf c++
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[in]: threshold for bounding box confidence
// @param[in]: threshold for mask
// @param[in]: cnn model path.
// @param[in]: cnn config path.
// @param[out]: mask result
// @ret: if mask and bndbox detected, return true;
// @birth: n/a
bool Algo::UNetDetectSegment(const Mat1b roiMask, const Mat3b roiImg1, const Mat3b roiImg2, const float fConfBndbox_threshold, 	\
	const float fConfMask_threshold, const string sModel_path, const string sConfig_path, Mat1b& output)
{
	dcout << "\t\t" << __FUNCTION__ << "(): start." << endl;

#ifdef DEEP_DEBUG
	imshow("CNN: roiMask", roiMask);
	imshow("CNN: roiImg1", roiImg1);
	imshow("CNN: roiImg2", roiImg2);
#endif

	bool bRst = false;
	Mat1b output_unionsection = Mat(roiMask.rows, roiMask.cols, CV_8UC1, Scalar(0));

	vector<Mat1b> vMask_img1, vMask_img2; 	

	Algo::fnUNetScan4Segmentation(roiImg1, fConfBndbox_threshold, fConfMask_threshold, sModel_path, sConfig_path, vMask_img1);
	for (auto img: vMask_img1)
	{
		bRst = true;
		bitwise_or(img, output_unionsection, output_unionsection);
	}
	Algo::fnUNetScan4Segmentation(roiImg2, fConfBndbox_threshold, fConfMask_threshold, sModel_path, sConfig_path, vMask_img2);
	for (auto img: vMask_img2)
	{
		bRst = true;
		bitwise_or(img, output_unionsection, output_unionsection);
	}

	bitwise_or(output_unionsection, output, output);

	// post process.
	// Algo::fnAutoThreshold(output, output);
	bitwise_and(roiMask, output, output);

	return bRst;
}


// @brief: binary classifier by opencv dnn, [inactive].
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[in]: threshold for classification
// @param[in]: cnn model path.
// @param[in]: cnn config path.
// @param[in]: width of input image
// @param[in]: height of input image
// @param[in]: scale factor for mobilenet
// @param[in]: mean value ofr mobilenet
// @ret: yes or no.
// @birth: n/a
bool Algo::isCNNSingleColorArea(Mat1b roiMask, Mat3b roiImg1, Mat3b roiImg2, float confidenceThreshold, \
	string model_path, string config_path, int inWidth, int inHeight, float inScaleFactor, float meanVal)
{
	dcout << "\t\t" << __FUNCTION__ << "(): start." << endl;

#ifdef DEEP_DEBUG
	imshow("CNN: roiMask", roiMask);
	imshow("CNN: roiImg1", roiImg1);
	imshow("CNN: roiImg2", roiImg2);
#endif

	//
	// normally, 0 means background.
	// one of them is background, it will be identified as background.
	// so, < 0, will be 'what we what'
	//
	if (0 == Algo::fnCNNScanForClassification(roiImg1, confidenceThreshold, model_path, config_path, inWidth, inHeight, inScaleFactor, meanVal))
	{
		return false;
	}

	if (0 == Algo::fnCNNScanForClassification(roiImg2, confidenceThreshold, model_path, config_path, inWidth, inHeight, inScaleFactor, meanVal))
	{
		return false;
	}

	return true;
}

// @brief: binary classifier by tflite api.
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[in]: threshold for classification
// @param[in]: cnn lib path.
// @param[in]: cnn model path.
// @param[in]: cnn label path.
// @ret: yes or no.
// @birth: n/a
bool Algo::isTFLiteSingleColorArea(const Mat1b roiMask, const Mat3b roiImg1, const Mat3b roiImg2, const float confThreshold, 	\
								   const string lib_path, const string model_path, const string label_path)
{
	dcout << "\t\t" << __FUNCTION__ << "(): start." << endl;


#ifdef DEEP_DEBUG
	imshow("CNN: roiMask", roiMask);
	imshow("CNN: roiImg1", roiImg1);
	imshow("CNN: roiImg2", roiImg2);
#endif

	//
	// normally, 0 means background.
	//
	// == Rule == 
	// either of them (image and its pair) is background, it will be identified as background
	//
	// so, 'larger than 0' will be 'what we what'
	//
	float conf_img1 = -1.0, conf_img2 = -1.0;
	if (0 == Algo::fnTFLiteScanForClassification(roiImg1, confThreshold, lib_path, model_path, label_path, conf_img1))
	{
 		icout << "\t\t\tAlgo --> TFLite mobilenet: is single color? [false]: roiImg1 --> (confidence) " << 1-conf_img1 << endl;
		return false;
	}

	if (0 == Algo::fnTFLiteScanForClassification(roiImg2, confThreshold, lib_path, model_path, label_path, conf_img2))
	{
 		icout << "\t\t\tAlgo --> TFLite mobilenet: is single color? [false]: roiImg2 --> (confidence) " << 1-conf_img2 << endl;
		return false;
	}

 	icout << "\t\t\tAlgo --> TFLite mobilenet: is single color? [true]: roiImg1 --> (conf) " << conf_img1 << "; roiImg2 --> (conf) " << conf_img2 << endl;
	return true;
}

// @brief: binary classifier by tflite api.
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[in]: threshold for classification
// @param[in]: cnn lib path.
// @param[in]: cnn model path.
// @param[in]: cnn label path.
// @ret: yes or no.
// @birth: n/a
bool Algo::isTFLiteDemageArea(const Mat1b roiMask, const Mat3b roiImg1, const Mat3b roiImg2, const float confThreshold, 	\
								   const string lib_path, const string model_path, const string label_path)
{
	dcout << "\t\t" << __FUNCTION__ << "(): start." << endl;


#ifdef DEEP_DEBUG
	imshow("CNN: roiMask", roiMask);
	imshow("CNN: roiImg1", roiImg1);
	imshow("CNN: roiImg2", roiImg2);
#endif

	//
	// normally, 0 means background.
	// either of them (image and its pair) is demage, it will be identified as demage
	//
	float conf_img1 = -1.0, conf_img2 = -1.0;
	if (0 < Algo::fnTFLiteScanForClassification(roiImg1, confThreshold, lib_path, model_path, label_path, conf_img1))
	{
 		icout << "\t\t\tAlgo --> TFLite mobilenet: is one type of demage? [true]: roiImg1 --> (confidence) " << conf_img1 << endl;
		return true;
	}

	if (0 < Algo::fnTFLiteScanForClassification(roiImg2, confThreshold, lib_path, model_path, label_path, conf_img2))
	{
 		icout << "\t\t\tAlgo --> TFLite mobilenet: is one type of demage? [true]: roiImg2 --> (confidence) " << conf_img2 << endl;
		return true;
	}

 	icout << "\t\t\tAlgo --> TFLite mobilenet: is one type of demage? [false]: roiImg1 --> (conf) " << 1-conf_img1 << "; roiImg2 --> (conf) " << 1-conf_img2 << endl;
	return false;
}
