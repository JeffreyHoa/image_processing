#include "FileSystem.h"


namespace spek_fs {

using namespace std;
using namespace cv;


// @brief:  Extract name of file or folder.
// @param[in]: path
// @ret: base name
// @birth: v0.94
struct MatchPathSeparator
{
	bool operator()( char ch ) const
	{
		return ch == '/';
	}
};

std::string fnGetBaseName( std::string const& c_pathname)
{
	string pathname = (c_pathname.substr(c_pathname.size() - 1) == "/") ? c_pathname.substr(0, c_pathname.size() - 1) : c_pathname; 
	return std::string( 
			std::find_if( pathname.rbegin(), pathname.rend(),
				MatchPathSeparator() ).base(),
			pathname.end() );
}

// @brief: check the folder path and return a valid one.
// @param[in]: old folder path
// @ret: new folder path
// @birth: v0.95
string fnGetFolderPath(const string str)
{
	string sPath = str;
	if (sPath[sPath.length()-1] != '/')
	{
		sPath = sPath + '/';
	}
	return sPath;
}

// @brief: check the path of file.
// @param[in]: file path
// @ret: bool
// @birth: v0.92
bool fn_isFileExists(const std::string &s)
{
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}

// @brief: remove the folder.
// @param[in]: folder path
// @ret: error no.
// @birth: v0.92
int fnRmFolder(const char *dir)
{
	char cur_dir[] = ".";
	char up_dir[] = "..";
	string dir_name;
	DIR *dirp;
	struct dirent *dp;
	struct stat dir_stat;

	if ( 0 != access(dir, F_OK) ) {
		return -1;
	}


	if ( 0 > stat(dir, &dir_stat) ) {
		perror("get directory stat error");
		return -1;
	}

	if ( S_ISREG(dir_stat.st_mode) ) {
		remove(dir);
	} else if ( S_ISDIR(dir_stat.st_mode) ) {
		dirp = opendir(dir);
		while ( (dp=readdir(dirp)) != NULL ) {

			if ( (0 == strcmp(cur_dir, dp->d_name)) || (0 == strcmp(up_dir, dp->d_name)) ) {
				continue;
			}

			// sprintf(dir_name, "%s/%s", dir, dp->d_name);
			dir_name = string(dir) + "/" + string(dp->d_name);
			const char* ptr_dir_name = dir_name.data();
			fnRmFolder(ptr_dir_name);
		}
		closedir(dirp);

		rmdir(dir);
	} else {
		perror("unknow file type!");    
	}
	return 0;
}

// @brief: check the path of folder.
// @param[in]: folder path
// @ret: bool
// @birth: v0.92
bool fn_isDirectoryExists( const char* pzPath )
{
    if ( pzPath == NULL) return false;

    DIR *pDir;
    bool bExists = false;

    pDir = opendir (pzPath);

    if (pDir != NULL)
    {
        bExists = true;    
        (void) closedir (pDir);
    }

    return bExists;
}

// @brief: Remove old folder and create a new one.
// @param[in]: folder path
// @ret: void
// @birth: v0.92
void fnCreateNewFolder(const std::string &str) 
{
	const string sPath = fnGetFolderPath(str);
	// icout << "output path: " << output_dirPath << endl;

	const char* dir = (char*)sPath.data();

	if(true == fn_isDirectoryExists(dir))
	{
		fnRmFolder(dir);
	}

	mkdir(dir, 0755);
}

// @brief: create a new one if there is no one.
// @param[in]: folder path
// @ret: void
// @birth: v0.97
void fnCreateFolder(const std::string &str) 
{
	const string sPath = fnGetFolderPath(str);
	// icout << "output path: " << output_dirPath << endl;

	const char* dir = (char*)sPath.data();

	if(true == fn_isDirectoryExists(dir))
	{
		return;
	}

	mkdir(dir, 0755);
}

// @brief: check the validity of path.
// @param[in]: img path
// @ret: Mat (move)
// @birth: v0.95
cv::Mat fn_imread(string path)
{
	// icout << __FUNCTION__ << "path: " << path << endl;

	if (false == fn_isFileExists(path))
	{
		cerr << "Warning: fail to load image." << endl
			 << "path: " << path << endl;
		return cv::Mat{};
	}

	Mat img = imread(path);
	if (img.empty())
	{
		cerr << "Error: image is empty." << endl
			 << "path: " << path << endl;
		return cv::Mat{};
	}
	
	return img;
}

// @brief: complete a path.
// @param[in]: root path
// @param[in]: file path
// @ret: new complete path
// @birth: v0.98
string fnPathJoin(const string root, const string path)
{
	if ('/' == root.back())
	{
		return root+path;
	}
	else
	{
		return root+"/"+path;
	}
}

// @brief: [debug] global variables for onTrackbar() and fnThermodynamicChart()
// @birth: v0.98
int maskThreshold = 1;
Mat ori_mask, ori_left, ori_right, mask_image, left_image, right_image, img_merge;
Mat1b thres_mask;
const char* window_name = "debug: img_merge interest areas.";

// @brief: [debug] define a trackbar callback
// @param[in]: default
// @param[in]: default
// @ret: void
// @birth: v0.98
static void onTrackbar(int, void*)
{
	// Here, the size of mask_image is the standard.
	// Thermodynamic Chart
	const int max_limit = 800;
	float scaling_init = 1.0;
	int longest_edge = ori_mask.cols > ori_mask.rows ? ori_mask.cols : ori_mask.rows;
	if (longest_edge > max_limit)
	{
		scaling_init = (float) max_limit / longest_edge;
	}

	resize(ori_mask, mask_image, cv::Size(), scaling_init, scaling_init);
	Mat3b resultRGB = Mat(mask_image.rows, mask_image.cols, CV_8UC3, Scalar(0,0,0));

	const float photo_alpha = 0.5;
	const float mask_alpha = (float) maskThreshold / 100;
	const int mask_threshold = (int) (255 * mask_alpha);

	// Mat1b thres_mask;	// it has become a global variable.
	threshold(mask_image, thres_mask, mask_threshold, 255, THRESH_BINARY);

	// scan every image pixel, white color to labeling color.
	for (int i = 0; i < thres_mask.rows; i++)
	{
		for (int j = 0; j < thres_mask.cols; j++)
		{
			int grayPixel = thres_mask.at<uchar>(i, j);
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

	resize(ori_left,  left_image,  cv::Size(resultRGB.cols, resultRGB.rows));
	addWeighted(resultRGB, 1-photo_alpha, left_image, photo_alpha, 0, left_image);

	resize(ori_right, right_image, cv::Size(resultRGB.cols, resultRGB.rows));
	// addWeighted(resultRGB, 1-photo_alpha, right_image, photo_alpha, 0, right_image);
 
	Mat outImg_left, outImg_right;
	cv::Size size(left_image.cols + right_image.cols, MAX(left_image.rows, right_image.rows));
 
	img_merge = Mat(size, CV_8UC3);
	outImg_left  = img_merge(Rect(0,               0, left_image.cols,  left_image.rows));
	outImg_right = img_merge(Rect(left_image.cols, 0, right_image.cols, right_image.rows));
 
	left_image.copyTo(outImg_left);
	right_image.copyTo(outImg_right);

	imshow(window_name, img_merge);
}

// @brief: [debug] show the debugging charts by controlling threshold.
// @param[in]: image 1
// @param[in]: image 2
// @param[in]: contourLine
// @param[in]: save folder path.
// @param[in/out]: iLabel in option
// @param[in]: fixed mask threshold
// @param[in]: manually flexible mask threshold setting.
// @ret: void
// @birth: v0.98
Mat1b fnThermodynamicChart(const Mat3b image1, const Mat3b image2, const Mat1b prob_mask, string const sSaveFolderPath, int *p_index, const int in_maskThreshold, const bool adjustMask_isAvailable)
{
	ori_left  = image1.clone();
	ori_right = image2.clone();
	ori_mask = prob_mask.clone();
  equalizeHist(ori_mask, ori_mask);
	maskThreshold = in_maskThreshold;


	if (true == adjustMask_isAvailable)
	{
		// Create a window
		namedWindow(window_name, 1);

		// create a toolbar
		createTrackbar("Display mask", window_name, &maskThreshold, 100, onTrackbar);

		// Show the image
		onTrackbar(0, 0);

		// Wait for a key stroke; the same function arranges events processing
		waitKey(0);
	}
	else
	{
		onTrackbar(0, 0);
	}

	if (sSaveFolderPath != "" && p_index != nullptr)
	{
		*p_index += 1;

		string savePath = spek_fs::fnPathJoin(sSaveFolderPath, to_string(*p_index) + "_ThermodynamicChart_thres" + to_string(maskThreshold) + ".jpg");
		imwrite(savePath, img_merge);
	}

	// Notice: return a global variable.
	Mat1b output; 
	resize(thres_mask, output, cv::Size(prob_mask.cols, prob_mask.rows));
	return output;
}


///////////////////////////////////////////////////////////////////////////////
// voting output.
///////////////////////////////////////////////////////////////////////////////

// @brief: save the voting results
// @param[in]: title of voting as a part of folder name.
// @param[in]: mask
// @param[in]: image
// @param[in]: its pair
// @ret: void
// @birth: n/a
void fnSaveVoteOutput(string title, Mat1b roiMask, Mat3b roiImg1, Mat3b roiImg2)
{
	// dcout << __FUNCTION__ << ", " << __LINE__ << endl;

	// 1. get a time frame id as name.
    // struct timeval tv;
    // gettimeofday(&tv,NULL);
    // printf("second:%ld\n",tv.tv_sec); 
    // printf("millisecond:%ld\n",tv.tv_sec*1000 + tv.tv_usec/1000);
    // printf("microsecond:%ld\n",tv.tv_sec*1000000 + tv.tv_usec);
	static int idxFile = 0;
	string file_id = to_string(idxFile++);

	// 2. create path
	string path_output_vote = "./OUTPUT_VOTE";
	fnCreateFolder(path_output_vote);

	string path_folder = path_output_vote + "/" + title;
	fnCreateFolder(path_folder);

	string path_roiMask = path_folder + "/" + file_id + "_roiMask.jpg";
	string path_roiImg1 = path_folder + "/" + file_id + "_roiImg1.jpg";
	string path_roiImg2 = path_folder + "/" + file_id + "_roiImg2.jpg";
	// dcout << "imwrite: " << path_roiMask << endl;
	// dcout << "imwrite: " << path_roiImg1 << endl;
	// dcout << "imwrite: " << path_roiImg2 << endl;

	// 3. save file.
#ifdef DEEP_DEBUG 
	imshow("roiMask", roiMask);
	imshow("roiImg1", roiImg1);
	imshow("roiImg2", roiImg2);
	moveWindow("roiMask", gapCol4imshow*0, gapRow4imshow*0);
	moveWindow("roiImg1", gapCol4imshow*2, gapRow4imshow*0);
	moveWindow("roiImg2", gapCol4imshow*4, gapRow4imshow*0);
#else
	imwrite(path_roiMask, roiMask);
	imwrite(path_roiImg1, roiImg1); 
	imwrite(path_roiImg2, roiImg2);
#endif

}


///////////////////////////////////////////////////////////////////////////////
// command line parser.
///////////////////////////////////////////////////////////////////////////////

// @brief: options
// @birth: v0.94
map<string, string> code_options = {
	{ "--pairDiffer_fuzzValue",              			"int" 	},		
	{ "--areaCounter_scaleRatio",           			"float" },		
	{ "--areaCounter_minArea",              			"int" 	},	
	{ "--areaCounter_maxArea",               			"int"	},	 
	{ "--areaCounter_default_sepRows",       			"int"	},	 
	{ "--areaCounter_default_sepCols",       			"int"	},	 
	{ "--areaFilter_classifier_isAvailable", 			"bool" 	},	        
	{ "--areaFilter_scaleRatio",           				"float" },		
	{ "--areaFilter_boosting_notSingleColor_minArea",	"int" 	},	       
	{ "--areaFilter_boosting_notSingleColor_maxArea",	"int" 	},	        
	{ "--areaFilter_boosting_notSingleColor_sepRows",	"int" 	},	          
	{ "--areaFilter_boosting_notSingleColor_sepCols",	"int" 	},	        
	{ "--areaFilter_boosting_demage_minArea",			"int" 	},	       
	{ "--areaFilter_boosting_demage_maxArea",			"int" 	},	        
	{ "--areaFilter_boosting_demage_sepRows",			"int" 	},	          
	{ "--areaFilter_boosting_demage_sepCols",			"int" 	},
	{ "--areaFilter_detector_isAvailable",   			"bool" 	},	       
	{ "--areaFilter_boosting_interestObject_minArea",	"int" 	},
	{ "--areaFilter_adjustMask_isAvailable",   			"bool" 	},	       
	{ "--areaFilter_display_maskThreshold",				"int" 	},
	{ "--crackFilter_scaleRatio",           			"float" },		
	{ "--crackFilter_classifier_isAvailable", 			"bool" 	},	        
	{ "--crackFilter_boosting_demage_minArea",			"int" 	},	       
	{ "--crackFilter_boosting_demage_maxArea",			"int" 	},	        
	{ "--crackFilter_boosting_demage_sepRows",			"int" 	},	          
	{ "--crackFilter_boosting_demage_sepCols",			"int" 	},
	{ "--crackFilter_adjustMask_isAvailable",  			"bool" 	},	       
	{ "--crackFilter_display_maskThreshold",			"int" 	}	
};

// @brief: print help info.
// @ret: void
// @birth: v0.92
void help()
{
    cout << "\n" 
	        "******\n"
    		" help \n"
			"******\n"
			"\n"
	        "This program demonstrated the SpekScan() function\n"
            "i.e.:\n"
            "./spekscan --image1 ./image1.jpg --image2 ./image2.jpg --output ./OUTPUT\n" << endl;

	for(auto x: code_options) {
		cout << x.first << " (" << x.second << ")" << endl;
	}
}

// @brief: print version.
// @param[in]: label
// @ret: void
// @birth: v0.92
void print_version(string label)
{
    cout << "\nSpekScan: " << label << endl;
}

// @brief: CommandLineParser
// @param[in]: argc
// @param[in]: argv
// @ret: structure DetectorProgramOptions
// @birth: v0.92
DetectorProgramOptions parse_detector_program_options(int argc, const char **argv)
{
	DetectorProgramOptions results;
	 
	for (int i = 1; i < argc; i++)
	{
		// icout << "parse: " << argv[i] << endl;
		if (strcmp(argv[i], "--help") == 0)
		{
			results.print_help_flag = true;
			return results;
		}
		else if (strcmp(argv[i], "--version") == 0)
		{
			results.print_version_flag = true;
			return results;
		}
		else if (strcmp(argv[i], "--image1") == 0)
		{
			if (i + 1 >= argc)
			{
				results.error_parsing_flag = true;
				cerr << "Error: --image1 requires a path after it" << endl;
				return results;
			}

			results.detector_options["image1"] = argv[++i];
		}
		else if (strcmp(argv[i], "--image2") == 0)
		{
			if (i + 1 >= argc)
			{
				results.error_parsing_flag = true;
				cerr << "Error: --image2 requires a path after it" << endl;
				return results;
			}

			results.detector_options["image2"] = argv[++i];
		}
		else if (strcmp(argv[i], "--output") == 0)
		{
			if (i + 1 >= argc)
			{
				results.error_parsing_flag = true;
				cerr << "Error: --output requires a folder path after it" << endl;
				return results;
			}

			results.detector_options["output"] = argv[++i];
		}
		else if (strcmp(argv[i], "--unittest") == 0)
		{
			// this is not compulsory option.
			if (i + 1 >= argc)
			{
				results.error_parsing_flag = true;
				cerr << "Error: --unittest requires a boolean after it" << endl;
				return results;
			}

			results.detector_options["unittest"] = argv[++i];
		}
		else if (strcmp(argv[i], "--mask") == 0)
		{
			// this is not compulsory option.
			if (i + 1 >= argc)
			{
				results.error_parsing_flag = true;
				cerr << "Error: --mask requires a path after it" << endl;
				return results;
			}

			results.detector_options["mask"] = argv[++i];
		}
		else
		{
			string cur_argv = string(argv[i]);
			if (code_options.find(cur_argv) == code_options.end()) 
			{
				results.error_parsing_flag = true;
				cerr << "Error: unknown arguments" << endl;
				return results;
			}

			if (i + 1 >= argc)
			{
				results.error_parsing_flag = true;
				cerr << "Error: " << argv[i] << "requires a value after it." << endl;
				return results;
			}

			results.detector_options[cur_argv] = argv[++i];
		}
	}

	//
	// this are compulsory parses.
	//
	if (results.detector_options.find("image1") == results.detector_options.end()) 
	{
		// not found.
		results.error_parsing_flag = true;
		cerr << "Error: cannot find --image1" << endl;
		return results;

	} 
	else if (results.detector_options.find("image2") == results.detector_options.end()) 
	{
		// not found.
		results.error_parsing_flag = true;
		cerr << "Error: cannot find --image2" << endl;
		return results;
	} 
	else if (results.detector_options.find("output") == results.detector_options.end()) 
	{
		// not found.
		results.error_parsing_flag = true;
		cerr << "Error: cannot find --output" << endl;
		return results;
	}
	else if (results.detector_options.find("unittest") != results.detector_options.end())
	{
		// can find unittest, but cannot find mask.
		if (results.detector_options.find("mask") == results.detector_options.end())
		{
			results.error_parsing_flag = true;
			cerr << "Error: cannot find --mask" << endl;
			return results;
		}
	}

	return results;
}

// @brief: check unit test option
// @param[in]: detector options
// @ret: true/false
// @birth: v0.95
bool fn_isUnitTest(map<string, const char *> detector_options)
{
	if (detector_options.find("unittest") != detector_options.end())
	{
		// find unittest.
		string unittest = string(detector_options["unittest"]);
		if (0 == unittest.compare("true"))
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}

	return false;
}

}
