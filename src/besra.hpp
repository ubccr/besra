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
#include <utility>
#include <queue>
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

            cv::Mat processPath(const fs::path &file, int limit = 0, int threads = 0, cv::Ptr<cv::BOWImgDescriptorExtractor> bow = NULL);

        public:
            Besra(int minHessian = 600);

            cv::Mat readImage(const fs::path &file);

            std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat &img);
            cv::Mat detectAndCompute(const cv::Mat &img);
            cv::Mat detectAndCompute(const cv::Mat &img, cv::Ptr<cv::BOWImgDescriptorExtractor> bow);

            cv::Mat buildVocabulary(std::vector<fs::path> paths, int clusterCount = 150, int limit = 0, int threads = 0);
            cv::Ptr<cv::BOWImgDescriptorExtractor> loadBOW(const cv::Mat &vocabulary);
            cv::Ptr<CvSVM> train(const fs::path &positive, const fs::path &negative, 
                                 const cv::Mat &vocabulary, int limit = 0, int threads = 0);
            cv::Ptr<CvSVM> loadStatModel(const fs::path &cache, const cv::Mat &vocabulary);
    };

    class PathQueue {
        private:
            std::queue<fs::path> queue;
            mutable boost::mutex mutex;
            boost::condition_variable waitCondition;
            bool done;
        public:
            PathQueue();
            void markDone();
            bool isDone();
            bool empty();
            void push(const fs::path &path);
            bool pop(fs::path &path);
    };

    class ImageConsumer {
        private:
            cv::Ptr<PathQueue> queue;
            cv::Mat descriptors;

        public:
            int id;
            ImageConsumer(int id, cv::Ptr<PathQueue> queue);
            void operator () (besra::Besra &besra, cv::Ptr<cv::BOWImgDescriptorExtractor> bow = NULL);
            cv::Mat getDescriptors();
    };

    class PathProducer {
        private:
            cv::Ptr<PathQueue> queue;

        public:
            int id;
            PathProducer(int id, cv::Ptr<PathQueue> queue);
            void operator () (const fs::path &path, int limit);
    };
}

#endif
