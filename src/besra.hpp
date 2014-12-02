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
#include <boost/algorithm/string.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup.hpp>
#include <boost/filesystem.hpp>
#include <boost/atomic.hpp>
#include <boost/thread/thread.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#ifdef USE_GPU
#include <opencv2/gpu/gpu.hpp>
#endif

namespace fs = boost::filesystem;

namespace besra {

    void init_log();

    class Besra {
        private:
            cv::Ptr<cv::DescriptorMatcher> matcher;
            cv::Ptr<cv::DescriptorExtractor> extractor;
            cv::Ptr<cv::FeatureDetector> detector;
#ifdef USE_GPU
            cv::Ptr<cv::gpu::SURF_GPU> gpu_surf;
#endif

            std::pair<cv::Mat, cv::Mat> processImages(const fs::path &file, int limit = 0, int threads = 0, 
                                                      cv::Ptr<cv::BOWImgDescriptorExtractor> bow = NULL);

        public:
            Besra(int minHessian = 600);

            cv::Mat readImage(const fs::path &file);

            std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat &img);
            cv::Mat detectAndCompute(const cv::Mat &img);
            cv::Mat detectAndCompute(const cv::Mat &img, cv::Ptr<cv::BOWImgDescriptorExtractor> bow);

            cv::Mat buildVocabulary(const fs::path &input_file, int clusterCount = 150, int limit = 0, int threads = 0);
            cv::Ptr<cv::BOWImgDescriptorExtractor> loadBOW(const cv::Mat &vocabulary);
            cv::Ptr<CvSVM> train(const fs::path &input_file, const cv::Mat &vocabulary, int limit = 0, int threads = 0);
            cv::Ptr<CvSVM> loadStatModel(const fs::path &cache, const cv::Mat &vocabulary);
            float classify(const fs::path &path, cv::Ptr<cv::BOWImgDescriptorExtractor> bow, cv::Ptr<CvSVM> model);
    };

    class ImageRecord {
        public:
            float label;
            fs::path path;
            ImageRecord();
            ImageRecord(float label, fs::path path);
    };

    class ImageQueue {
        private:
            std::queue<ImageRecord> queue;
            mutable boost::mutex mutex;
            boost::condition_variable waitCondition;
            bool done;
        public:
            ImageQueue();
            void markDone();
            bool isDone();
            bool empty();
            void push(const ImageRecord &rec);
            bool pop(ImageRecord &rec);
    };

    class ImageConsumer {
        private:
            int count;
            cv::Ptr<ImageQueue> queue;
            cv::Mat descriptors;
            cv::Mat labels;

        public:
            int id;
            ImageConsumer(int id, cv::Ptr<ImageQueue> queue);
            void operator () (besra::Besra &besra, cv::Ptr<cv::BOWImgDescriptorExtractor> bow = NULL);
            cv::Mat getDescriptors();
            cv::Mat getLabels();
    };

    class ImageProducer {
        private:
            cv::Ptr<ImageQueue> queue;

        public:
            int id;
            ImageProducer(int id, cv::Ptr<ImageQueue> queue);
            void operator () (const fs::path &path, int limit);
    };
}

#endif
