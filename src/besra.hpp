#ifndef __BESRA_H_INCLUDED__
#define __BESRA_H_INCLUDED__ 


#include <iostream>
#include <utility>
#include "boost/log/trivial.hpp"
#include "boost/log/utility/setup.hpp"
#include "boost/filesystem.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#ifdef USE_GPU
#include "opencv2/gpu/gpu.hpp"
#endif

namespace fs = boost::filesystem;

namespace besra {

    void init_log();

    cv::Mat extract_features_from_image(fs::path image_path, cv::BOWImgDescriptorExtractor *bow = NULL);

    cv::Mat extract_features_from_dir(fs::path dir, int desc_sz, int limit, cv::BOWImgDescriptorExtractor *bow = NULL);
}

#endif
