/**
 * Copyright (C) 2014 Andrew E. Bruno
 *
 * This file is part of Besra
 *
 * Besra is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef __BESRA_H_INCLUDED__
#define __BESRA_H_INCLUDED__ 


#include <iostream>
#include <fstream>
#include <utility>
#include <string>
#include <queue>
#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#ifdef USE_GPU
#include <opencv2/gpu/gpu.hpp>
#endif
#ifdef OPENMP_FOUND
#include <omp.h>
#endif

namespace fs = boost::filesystem;

namespace besra {

    void init_log(bool verbose=false);

    class Besra {
        private:
            bool processLine(std::string line, cv::Mat &descriptors, int &label,
                              cv::Ptr<cv::BOWImgDescriptorExtractor> bow);

            std::pair<cv::Mat, cv::Mat> processImages(const fs::path &file, int threads = 0, 
                                                      cv::Ptr<cv::BOWImgDescriptorExtractor> bow = cv::Ptr<cv::BOWImgDescriptorExtractor>());

        public:
            cv::Ptr<cv::DescriptorMatcher> matcher;
            cv::Ptr<cv::DescriptorExtractor> extractor;
            cv::Ptr<cv::FeatureDetector> detector;
#ifdef USE_GPU
            cv::Ptr<cv::gpu::SURF_GPU> gpu_surf;
#endif

            Besra(int minHessian = 600, std::string extractor="SURF", std::string detector="SURF");

            cv::Mat readImage(const fs::path &file);

            std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat &img);
            cv::Mat detectAndCompute(const cv::Mat &img);
            cv::Mat detectAndCompute(const cv::Mat &img, cv::Ptr<cv::BOWImgDescriptorExtractor> bow);

            cv::Mat buildVocabulary(const fs::path &input_file, int clusterCount = 150, int threads = 0);
            cv::Ptr<cv::BOWImgDescriptorExtractor> loadBOW(const cv::Mat &vocabulary);
            cv::Ptr<cv::ml::StatModel> train(const fs::path &input_file, const cv::Mat &vocabulary, int threads = 0);
            cv::Ptr<cv::ml::StatModel> loadStatModel(const fs::path &cache, const cv::Mat &vocabulary);
            float classify(const fs::path &path, cv::Ptr<cv::BOWImgDescriptorExtractor> bow, cv::Ptr<cv::ml::StatModel> model);
    };

}

#endif
