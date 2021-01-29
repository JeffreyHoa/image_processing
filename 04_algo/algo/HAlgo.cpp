// Copyright 2020 Jeffrey Hao. All Rights Reserved.
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

#include "HAlgo.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace spek_fs;

////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// initialize /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

//
// Algo.h
//
const float HAlgo::algo_areaBndboxRatio_minThreshold_ = 0.1;	// valid value should be larger than it
const float HAlgo::algo_areaBndboxRatio_maxThreshold_ = 0.5;	// valid value should be smaller than it
const float HAlgo::algo_singleColorArea_var_threshold_ = 15.1;	// larger, more strict
const float HAlgo::algo_singleColorArea_mean_threshold_ = 15.1; // larger, more strict
const float HAlgo::algo_singleHueArea_var_threshold_ = 2.1;		// larger, more strict
const float HAlgo::algo_singleHueArea_mean_threshold_ = 2.1;	// larger, more strict

//
// We implement four api for cnn, and we use two of them.
// (1) mobilenet-ssd by opencv dnn (frozen)
// (2) mobilenet by opencv dnn (frozen)
// (3) mask r-cnn by opencv dnn (active)
// (4) mobilenet by tflite (active)
//

// (1) mobilenet-ssd by opencv dnn (frozen)
// download: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
const std::string HAlgo::ssd_model_path_ = "../data/mobilenet_ssd.pb";
const std::string HAlgo::ssd_config_path_ = "../data/mobilenet_ssd.pbtxt";
const int HAlgo::ssd_inWidth_ = 300;			   /* width of input image of cnn */
const int HAlgo::ssd_inHeight_ = 300;			   /* height of input image of cnn */
const float HAlgo::ssd_inScaleFactor_ = 0.007843f; /* scale factor of input image of cnn */
const float HAlgo::ssd_meanVal_ = 127.5;		   /* mean value of input image of cnn */
const float HAlgo::ssd_confidenceThreshold_ = 0.0; /* criteria for valid bounding box */

// (2) mobilenet by opencv dnn (frozen)
const std::string HAlgo::cnn_model_path_ = "../data/mobilenet.pb";
const std::string HAlgo::cnn_config_path_ = "../data/mobilenet.pbtxt";
const int HAlgo::cnn_inWidth_ = 224;			   /* width of input image of cnn */
const int HAlgo::cnn_inHeight_ = 224;			   /* height of input image of cnn */
const float HAlgo::cnn_inScaleFactor_ = 0.007843f; /* scale factor of input image of cnn */
const float HAlgo::cnn_meanVal_ = 127.5;		   /* mean value of input image of cnn */
const float HAlgo::cnn_confidenceThreshold_ = 0.0; /* criteria for valid bounding box */

// (3) mask r-cnn by opencv dnn (active)
// The predicted mask is only 15 x 15 pixels so we resize the mask back to the original input image dimensions.
const std::string HAlgo::maskrcnn_model_path_ = "../data/maskrcnn.pb";
const std::string HAlgo::maskrcnn_config_path_ = "../data/maskrcnn.pbtxt";
const float HAlgo::maskrcnn_confBndboxThreshold_ = 0.1; // criteria for valid bounding box */
const float HAlgo::maskrcnn_confMaskThreshold_ = 0.1;	// criteria for valid bounding box */
const int HAlgo::maskrcnn_predict_minEdge_ = 1000;		// resize input image to meet the min requirement of maskrcnn

// (4) mobilenet by tflite (active)
const std::string HAlgo::tflite_mobilenet_singleColor_model_path_ = "../data/mobilenet_singleColor.tflite";
const std::string HAlgo::tflite_mobilenet_singleColor_label_path_ = "../data/mobilenet_singleColor.txt";
const std::string HAlgo::tflite_mobilenet_singleColor_lib_path_ = "../lib/label_image.so";
const float HAlgo::tflite_mobilenet_singleColor_confThreshold_ = 0.51; // criteria for binary classification

// (5) mobilenet by tflite (active)
const std::string HAlgo::tflite_mobilenet_demage_model_path_ = "../data/mobilenet_demage.tflite";
const std::string HAlgo::tflite_mobilenet_demage_label_path_ = "../data/mobilenet_demage.txt";
const std::string HAlgo::tflite_mobilenet_demage_lib_path_ = "../lib/label_image.so";
const float HAlgo::tflite_mobilenet_demage_confThreshold_ = 0.51; // criteria for binary classification

// (6) unet by tf c++ api (active)
const std::string HAlgo::tfunet_model_path_ = "../data/detect_crack.pb";
const std::string HAlgo::tfunet_config_path_ = "../data/null.pbtxt"; // invalid
const float HAlgo::tfunet_confBndboxThreshold_ = -1.0;				 // invalid
const float HAlgo::tfunet_confMaskThreshold_ = -1.0;				 // invalid
const int HAlgo::tfunet_input_size_ = 256;							 // default value.


///////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// function //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

// @brief: Here we use opensource solution ImageMagick to detect difference for large damages.
// Any other better solution is welcome to replace this.
// @param[in]: image path
// @param[in]: its pair path
// @param[in]: fuzz value of ImageMagick
// @param[in]: output file path.
// @param[out]: binary mask of difference.
// @ret: 0 (success).
// @birth: n/a
int HAlgo::fnImageMagickDetection(const string pristine_path, const string pristine_partner_path, const int fuzz_value,	\
	const string output_diff_path, Mat1b& output_bin)
{
	// (1) Generate diff result.
	string sFuzzValue = to_string(fuzz_value); 
	string s_diffCmd = "magick compare -fuzz " + sFuzzValue + "% " + pristine_path + " " + pristine_partner_path + " " + output_diff_path;
	dcout << "\tInfo: ImageMagick, command: " << s_diffCmd << endl;
	const char *c_diffCmd = s_diffCmd.c_str();

	system(c_diffCmd);

	// (2) Transform to binary image.
	Mat3b diff_result = imread(output_diff_path);
	Mat1b diff_gray, diff_bin;
 	cvtColor(diff_result, diff_gray, COLOR_BGR2GRAY);
	Algo::fnAutoThreshold(diff_gray, diff_bin);

	// (3) Notice: black area represents differenes.
	bitwise_not(diff_bin, diff_bin);

	output_bin = diff_bin.clone();

	return 0;
}

// @brief: edge detection by difference methods.
// @param[in]: input image 
// @param[out]: edge result with rgb channels.
// @param[in]: option for laplacian method.
// @ret: 0 (success).
// @birth: n/a
int HAlgo::fnEdgeDetection(const Mat3b image, Mat3b& output, const bool isLaplacian)
{
	Mat3b img = image.clone();
	Mat1b img_gray, edge;
 	cvtColor(img, img_gray, COLOR_BGR2GRAY);

	// Edge detection.
	if (isLaplacian == false)
	{
		Algo::fnSobelDetection(img_gray, edge);
	}
	else
	{
		Algo::fnLaplacianDetection(img_gray, edge);
	}

	// Threshold to black and white.
	Mat1b bin;
	Mat3b edge_rgb;
	Algo::fnAutoThreshold(edge, bin);
    cvtColor(bin, edge_rgb, COLOR_GRAY2BGR);

	output = edge_rgb.clone();

	return 0;
}


// @brief: extract roi part on images.
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[out]: roi of image
// @param[out]: roi of its pair
// @param[out]: roi of mask
// @param[out]: Rect of roi
// @param[in]: use mask or not.
// @ret: success or fail
// @birth: n/a
bool HAlgo::fnGetRoiPair(const Mat1b mask, const Mat3b img1, const Mat3b img2, 	\
	Mat3b& output_roiImg1, Mat3b& output_roiImg2, Mat1b& output_roiMask, Rect& output_rect, const bool use_mask)
{
	int num_of_rect = Algo::fnGetRoiPair4Classifier(mask, img1, img2, output_roiImg1, output_roiImg2, output_roiMask, output_rect, use_mask);
	if (0 == num_of_rect) {
		icout << "\t\t\tThere's nothing in this pure black mask." << endl;
		return false;

	} else if (1 == num_of_rect) {
		;
	} else {
		wcout << "\t\t\tWarning: should be 1 or 0 from fnGetRoiPair4Classifier()" << endl;
		return false;
	}

	return true;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Simple Classifier to filter out the noisy area in Stage 1.
///////////////////////////////////////////////////////////////////////////////////////////////////

// @brief: binary classification according to its shape
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @ret: yes or no
// @birth: n/a
bool HAlgo::isAvailableArea_edgeDetection(const Mat1b in_mask, const Mat3b in_img1, const Mat3b in_img2)
{
	icout << "\t\t" << __FUNCTION__ << "(): start." << endl;

	// pre processing.
	Mat3b roiImg1 = Mat(in_img1.rows, in_img1.cols, CV_8UC3, Scalar(0,0,0));
	Mat3b roiImg2 = Mat(in_img2.rows, in_img2.cols, CV_8UC3, Scalar(0,0,0));
	Mat1b roiMask = Mat(in_mask.rows, in_mask.cols, CV_8UC1, Scalar(0));
	const bool isRoi = true;
	Rect roiRect;
	if (false == HAlgo::fnGetRoiPair(in_mask, in_img1, in_img2, roiImg1, roiImg2, roiMask, roiRect, isRoi))
	{
		return false;
	}

	//----------------------------------------------------
	// manual binary classification.
	//----------------------------------------------------
	if (false == Algo::isValidAreaBndboxRatio(roiMask, roiImg1, roiImg2, algo_areaBndboxRatio_minThreshold_, algo_areaBndboxRatio_maxThreshold_))
	{
		icout << "\t\t\t" << __FUNCTION__ << "(): false." << endl;
		return false;
	}

	/* implement if necessary */

	icout << "\t\t\t" << __FUNCTION__ << "(): true!" << endl;
	return true;
}

// @brief: binary classification according to both variance and mean of gray channel
//	notice, 'not single color' is available area.
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @ret: yes or no
// @birth: n/a
bool HAlgo::isAvailableArea_areaComparison(const Mat1b in_mask, const Mat3b in_img1, const Mat3b in_img2)
{
	icout << "\t\t" << __FUNCTION__ << "(): start." << endl;

	// pre processing.
	Mat3b roiImg1 = Mat(in_img1.rows, in_img1.cols, CV_8UC3, Scalar(0,0,0));
	Mat3b roiImg2 = Mat(in_img2.rows, in_img2.cols, CV_8UC3, Scalar(0,0,0));
	Mat1b roiMask = Mat(in_mask.rows, in_mask.cols, CV_8UC1, Scalar(0));
	const bool isRoi = true;
	Rect roiRect;
	if (false == HAlgo::fnGetRoiPair(in_mask, in_img1, in_img2, roiImg1, roiImg2, roiMask, roiRect, isRoi))
	{
		return false;
	}
	
	//----------------------------------------------------
	// manual binary classification.
	//----------------------------------------------------
	if (true == Algo::isSingleColorArea_var(roiMask, roiImg1, roiImg2, algo_singleColorArea_var_threshold_))
	{
		icout << "\t\t\t" << __FUNCTION__ << "() var: false." << endl;
		return false;
	}

	if (true == Algo::isSingleColorArea_mean(roiMask, roiImg1, roiImg2, algo_singleColorArea_mean_threshold_))
	{
		icout << "\t\t\t" << __FUNCTION__ << "() mean: false." << endl;
		return false;
	}
	
	/* implement if necessary */

	icout << "\t\t\t" << __FUNCTION__ << "(): true!" << endl;
	return true;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Weak Classifiers to do the binary classification and detection in Stage 3.
///////////////////////////////////////////////////////////////////////////////////////////////////

// @brief: save the voting results of a func.
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[in]: voting index
// @param[in]: voting result of a func.
// @param[in]: title of voting as a part of folder name.
// @param[out]: voting reuslt chain of all func.
// @ret: void
// @birth: n/a
void voting_job(const Mat1b roiMask, const Mat3b roiImg1,const Mat3b roiImg2, const size_t voting_idx, 	\
				const bool b_rst, const string title, vector<int> &vec_voteRecord, int value_per_vote=1) {

	int vote = (b_rst == true) ? value_per_vote : -value_per_vote;
	vec_voteRecord.push_back(vote);
	ncout << "\t\t\t\t[" << voting_idx << "] The voter says: " + to_string(vote) + ", " + title << endl;

	string name = title + "_" + to_string(vote);
	spek_fs::fnSaveVoteOutput(name, roiMask, roiImg1, roiImg2);
}

// @brief: binary classification, if not single color, we will reserve it.
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @ret: voting result.
// @birth: n/a
bool HAlgo::boosting_isSingleColor(const Mat1b in_mask, const Mat3b in_img1, const Mat3b in_img2)
{
	icout << "\t\t" << __FUNCTION__ << "(), voting time starts." << endl;

	//
	// pre-processing
	//
	Mat3b roiImg1 = Mat(in_img1.rows, in_img1.cols, CV_8UC3, Scalar(0,0,0));
	Mat3b roiImg2 = Mat(in_img2.rows, in_img2.cols, CV_8UC3, Scalar(0,0,0));
	Mat1b roiMask = Mat(in_mask.rows, in_mask.cols, CV_8UC1, Scalar(0));
	const bool isRoi = true;
	Rect roiRect;
	if (false == HAlgo::fnGetRoiPair(in_mask, in_img1, in_img2, roiImg1, roiImg2, roiMask, roiRect, isRoi))
	{
		return false;
	}


	//
	// Init before voting.
	// 
	// This is the record of voting result for each weak classifier.
	// Notice, it is allowed that one votor has more important vote.
	//
	vector<int> vec_voteRecord;
	static int voting_idx = -1;
	voting_idx++;


	//----------------------------------------------------
	// voter 0 has one vote.
	//----------------------------------------------------
	{
		const bool bSingleColor = Algo::isSingleHueArea_var(roiMask, roiImg1, roiImg2, algo_singleHueArea_var_threshold_);

		const string title = "isSingleHueArea_var_" + to_string(algo_singleHueArea_var_threshold_);
		voting_job(roiMask, roiImg1, roiImg2, voting_idx, bSingleColor, title, vec_voteRecord);
	}


	//----------------------------------------------------
	// voter 1 has one vote.
	//----------------------------------------------------
	{
		const bool bSingleColor = Algo::isSingleHueArea_mean(roiMask, roiImg1, roiImg2, algo_singleHueArea_mean_threshold_);

		const string title = "isSingleHueArea_mean_" + to_string(algo_singleHueArea_mean_threshold_);
		voting_job(roiMask, roiImg1, roiImg2, voting_idx, bSingleColor, title, vec_voteRecord);
	}


	//----------------------------------------------------
	// we prefer a strict way to identify single color.
	// if it is definately single color, return true.
	//----------------------------------------------------

	int rst_vote_isSingleColor = accumulate(vec_voteRecord.begin(), vec_voteRecord.end(), 0);
	if (rst_vote_isSingleColor == (int)vec_voteRecord.size())
	{
		ncout << "\t\t\tVote statistics: " << rst_vote_isSingleColor << " in " << __FUNCTION__ << "(): true, remove it!" << endl;
		return true;
	}

	// next stage, we will ask cnn to do the final decision independently.
	vec_voteRecord.clear();


	//----------------------------------------------------
	// voter 2 has one vote.
	//----------------------------------------------------
	{
		const bool bSingleColor = Algo::isTFLiteSingleColorArea(roiMask, roiImg1, roiImg2, tflite_mobilenet_singleColor_confThreshold_, \
			tflite_mobilenet_singleColor_lib_path_,
			tflite_mobilenet_singleColor_model_path_,
			tflite_mobilenet_singleColor_label_path_);

		const string title = "isTFLiteSingleColorArea_" + to_string(tflite_mobilenet_singleColor_confThreshold_);
		voting_job(roiMask, roiImg1, roiImg2, voting_idx, bSingleColor, title, vec_voteRecord);
	}


	//----------------------------------------------------
	// statistic result of voting for not_single_color
	//----------------------------------------------------
	int rst_vote = accumulate(vec_voteRecord.begin(), vec_voteRecord.end(), 0);
	if (rst_vote >= 0)
	{
		ncout << "\t\t\tVote statistics: " << rst_vote << " in " << __FUNCTION__ << "(): true, remove it." << endl;
		return true;
	}
	else
	{
		ncout << "\t\t\tVote statistics: " << rst_vote << " in " << __FUNCTION__ << "(): false, reserve it!" << endl;
		return false;
	}
}

// @brief: binary classification, if not single color, we will reserve it.
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @ret: voting result.
// @birth: n/a
bool HAlgo::boosting_notSingleColor(const Mat1b in_mask, const Mat3b in_img1, const Mat3b in_img2)
{
	return !HAlgo::boosting_isSingleColor(in_mask, in_img1, in_img2);
}

// @brief: binary classification, if it is demage, we will reserve it.
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @ret: voting result.
// @birth: n/a
bool HAlgo::boosting_isDemage(const Mat1b in_mask, const Mat3b in_img1, const Mat3b in_img2)
{
	icout << "\t\t" << __FUNCTION__ << "(), voting time starts." << endl;

	//
	// pre-processing
	//
	Mat3b roiImg1 = Mat(in_img1.rows, in_img1.cols, CV_8UC3, Scalar(0,0,0));
	Mat3b roiImg2 = Mat(in_img2.rows, in_img2.cols, CV_8UC3, Scalar(0,0,0));
	Mat1b roiMask = Mat(in_mask.rows, in_mask.cols, CV_8UC1, Scalar(0));
	const bool isRoi = true;
	Rect roiRect;
	if (false == HAlgo::fnGetRoiPair(in_mask, in_img1, in_img2, roiImg1, roiImg2, roiMask, roiRect, isRoi))
	{
		return false;
	}


	//
	// Init before voting.
	// 
	// This is the record of voting result for each weak classifier.
	// Notice, it is allowed that one votor has more important vote.
	//
	vector<int> vec_voteRecord;
	static int voting_idx = -1;
	voting_idx++;


	// Here, we will ask cnn to do the decision directly.

	//----------------------------------------------------
	// voter 0 has one vote.
	//----------------------------------------------------
	{
		const bool bDemage = Algo::isTFLiteDemageArea(roiMask, roiImg1, roiImg2, tflite_mobilenet_demage_confThreshold_, \
			tflite_mobilenet_demage_lib_path_,
			tflite_mobilenet_demage_model_path_,
			tflite_mobilenet_demage_label_path_);

		const string title = "isTFLiteDemageArea_" + to_string(tflite_mobilenet_demage_confThreshold_);
		voting_job(roiMask, roiImg1, roiImg2, voting_idx, bDemage, title, vec_voteRecord);
	}


	//----------------------------------------------------
	// statistic result of voting for not_single_color
	//----------------------------------------------------
	int rst_vote = accumulate(vec_voteRecord.begin(), vec_voteRecord.end(), 0);
	if (rst_vote >= 0)
	{
		ncout << "\t\t\tVote statistics: " << rst_vote << " in " << __FUNCTION__ << "(): true, reserve it." << endl;
		return true;
	}
	else
	{
		ncout << "\t\t\tVote statistics: " << rst_vote << " in " << __FUNCTION__ << "(): false, remove it!" << endl;
		return false;
	}
}

// @brief: using mask-rcnn or unet to scan whether some objects can be detected.
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @param[out]: mask result.
// @ret: yes or no.
// @birth: n/a
bool HAlgo::boosting_interestObject(const Mat1b mask, const Mat3b img1, const Mat3b img2, Mat1b& output_mask)
{
	icout << "\t\t" << __FUNCTION__ << "(), voting time starts." << endl;

#ifdef DEEP_DEBUG
	imshow("CNN: mask", mask);
	imshow("CNN: img1", img1);
	imshow("CNN: img2", img2);
#endif

	//
	// pre-processing
	//
	Mat3b roiImg1 = Mat(img1.rows, img1.cols, CV_8UC3, Scalar(0,0,0));
	Mat3b roiImg2 = Mat(img2.rows, img2.cols, CV_8UC3, Scalar(0,0,0));
	Mat1b roiMask = Mat(mask.rows, mask.cols, CV_8UC1, Scalar(0));
	const bool isRoi = false;
	Rect roiRect;
	if (false == HAlgo::fnGetRoiPair(mask, img1, img2, roiImg1, roiImg2, roiMask, roiRect, isRoi))
	{
		return false;
	}

	// debug algo api test.
	// imshow("roiImg1", roiImg1);
	// imshow("roiImg2", roiImg2);
	// imshow("roiMask", roiMask);
	// waitKey(0);

	//
	// Init before voting.
	// 
	// This is the record of voting result for each weak classifier.
	// Notice, it is allowed that one votor has more important vote.
	//
	vector<int> vec_voteRecord;
	static int voting_idx = 0;
	voting_idx++;


	//----------------------------------------------------
	// voter 0 has one vote.
	//----------------------------------------------------
	{
#if 0
		// unet maybe enough, comment out this to faster performance.
		Mat1b output_roi_mask = Mat(roiMask.rows, roiMask.cols, CV_8UC1, Scalar(0));
		Mat1b output_mask_local = Mat(mask.rows, mask.cols, CV_8UC1, Scalar(0));
		const bool b_rst = Algo::MASKRCNNDetectSegment(roiMask, roiImg1, roiImg2, \
				maskrcnn_confBndboxThreshold_, maskrcnn_confMaskThreshold_, maskrcnn_predict_minEdge_,	\
				maskrcnn_model_path_, maskrcnn_config_path_, output_roi_mask);
#ifdef DEEP_INFO_FILTER
		imshow("MASKRCNN", output_roi_mask);
#endif
		output_roi_mask.copyTo(output_mask_local(roiRect));
		bitwise_or(output_mask, output_mask_local, output_mask);

		const string title = "MASKRCNNDetectSegment_" + to_string(maskrcnn_confBndboxThreshold_) + "_" +	\
											   to_string(maskrcnn_confMaskThreshold_);
		voting_job(roiMask, roiImg1, roiImg2, voting_idx, b_rst, title, vec_voteRecord);
#endif
	}


	//----------------------------------------------------
	// voter 1 has one vote.
	//----------------------------------------------------
	{
		Mat1b output_roi_mask = Mat(roiMask.rows, roiMask.cols, CV_8UC1, Scalar(0));
		Mat1b output_mask_local = Mat(mask.rows, mask.cols, CV_8UC1, Scalar(0));
		const bool b_rst = Algo::UNetDetectSegment(roiMask, roiImg1, roiImg2, \
				tfunet_confBndboxThreshold_, tfunet_confMaskThreshold_,	\
				tfunet_model_path_, tfunet_config_path_, output_roi_mask);
#ifdef DEEP_INFO_FILTER
		imshow("UNet", output_roi_mask);
		imshow("output_mask", output_mask);
#endif
		output_roi_mask.copyTo(output_mask_local(roiRect));

		// bitwise_or(output_mask, output_mask_local, output_mask);
		output_mask_local.copyTo(output_mask);

		const string title = "UNetDetectSegment_" + to_string(tfunet_confBndboxThreshold_) + "_" +	\
											   to_string(tfunet_confMaskThreshold_);
		voting_job(roiMask, roiImg1, roiImg2, voting_idx, b_rst, title, vec_voteRecord);
	}


	//----------------------------------------------------
	// statistic result of voting.
	//----------------------------------------------------

	int rst_vote = 0;
	for (auto x: vec_voteRecord)
	{
		// once one element > 0, the segmentation has been found.
		if (x > 0)
		{
			rst_vote = 1;
			break;
		}
	}

	if (rst_vote > 0)
	{
		ncout << "\t\t\tVote statistics: " << rst_vote << " in " << __FUNCTION__ << "(): true, remain it." << endl;
		return true;
	}
	else
	{
		// only reach here when rst_vote == 0.
		ncout << "\t\t\tVote statistics: " << rst_vote << " in " << __FUNCTION__ << "(): false, remove it!!!" << endl;
		return false;
	}
}

// @brief: using mask-rcnn or unet to scan whether some objects can be detected, only for one image.
// @param[in]: mask
// @param[in]: image
// @param[out]: mask result.
// @ret: yes or no.
// @birth: v0.98
bool HAlgo::detect_interestObject(const Mat1b mask, const Mat3b img1, Mat1b& output_mask)
{
	icout << "\t\t" << __FUNCTION__ << "(), voting time starts." << endl;

#ifdef DEEP_DEBUG
	imshow("CNN: mask", mask);
	imshow("CNN: img1", img1);
#endif

	//
	// pre-processing
	//
	Mat3b roiImg1 = Mat(img1.rows, img1.cols, CV_8UC3, Scalar(0,0,0));
	Mat1b roiMask = Mat(mask.rows, mask.cols, CV_8UC1, Scalar(0));
	const bool isRoi = false;
	Rect roiRect;
	if (false == HAlgo::fnGetRoiPair(mask, img1, img1, roiImg1, roiImg1, roiMask, roiRect, isRoi))
	{
		return false;
	}

	//
	// detection
	//
	Mat1b output_unionsection = Mat(mask.rows, mask.cols, CV_8UC1, Scalar(0));
	vector<Mat1b> vMask_img;

	Algo::fnUNetScan4Segmentation_advanced(img1, tfunet_confBndboxThreshold_, tfunet_confMaskThreshold_, tfunet_model_path_, tfunet_config_path_, vMask_img);
	for (auto img: vMask_img)
	{
		bitwise_or(img, output_unionsection, output_unionsection);
	}

	// post process.
	// Algo::fnAutoThreshold(output_unionsection, output_unionsection);
	output_unionsection.copyTo(output_mask);
	// imshow("segmentation result", output_unionsection);
	// waitKey(0);

	if (vMask_img.size() > 0) 
		return true;
	return false;
}
