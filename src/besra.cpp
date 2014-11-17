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
#include "besra.hpp"

namespace besra {

    void init_log() {
        boost::log::register_simple_formatter_factory< boost::log::trivial::severity_level, char >("Severity");
        boost::log::add_console_log(
            std::cout, 
            boost::log::keywords::format = "%TimeStamp% [%Severity%]: %Message%",
            boost::log::keywords::auto_flush = true
        );

        boost::log::add_common_attributes();

        boost::log::core::get()->set_filter(
            boost::log::trivial::severity >= boost::log::trivial::info
        );
    }

    Besra::Besra() {
        //matcher = new cv::BruteForceMatcher< cv::L2<float> >();
        matcher = new cv::FlannBasedMatcher();
        extractor = new cv::SurfDescriptorExtractor();
        detector = new cv::SurfFeatureDetector(1000);
#ifdef USE_GPU
        gpu_surf = new cv::gpu::SURF_GPU(1000);
#endif
    }

    cv::Mat Besra::readImage(const fs::path &file) {
        return cv::imread(file.string(), CV_LOAD_IMAGE_GRAYSCALE);
    }

    std::vector<cv::KeyPoint> Besra::detectKeypoints(const cv::Mat &img) {
        std::vector<cv::KeyPoint> keypoints; 

#ifdef USE_GPU
        cv::gpu::GpuMat gpu_keypoints;
        cv::gpu::GpuMat gpu_descriptors;
        cv::gpu::GpuMat gpu_img;
        gpu_img.upload(img);
        (*gpu_surf)(gpu_img, cv::gpu::GpuMat(), gpu_keypoints, gpu_descriptors);
        gpu_surf->downloadKeypoints(gpu_keypoints, keypoints); 
#else
        detector->detect(img, keypoints);
#endif

        return keypoints;
    }

    cv::Mat Besra::detectAndCompute(const cv::Mat &img) {
        cv::Mat descriptors;

#ifdef USE_GPU
        cv::gpu::GpuMat gpu_keypoints;
        cv::gpu::GpuMat gpu_descriptors;
        cv::gpu::GpuMat gpu_img;
        gpu_img.upload(img);
        (*gpu_surf)(gpu_img, cv::gpu::GpuMat(), gpu_keypoints, gpu_descriptors);
        gpu_descriptors.download(descriptors);
#else
        std::vector<cv::KeyPoint> keypoints = detectKeypoints(img);
        extractor->compute(img, keypoints, descriptors);
#endif

        return descriptors;
    }

    cv::Mat Besra::detectAndCompute(const cv::Mat &img, cv::Ptr<cv::BOWImgDescriptorExtractor> bow) {
        cv::Mat descriptors;
            
        std::vector<cv::KeyPoint> keypoints = detectKeypoints(img);
        bow->compute(img, keypoints, descriptors);

        return descriptors;
    }

    cv::Mat Besra::buildVocabulary(std::vector<fs::path> dirs, int clusterCount, int limit) {
        cv::BOWKMeansTrainer bowtrainer(clusterCount);

        for(std::vector<fs::path>::iterator d = dirs.begin(); d != dirs.end(); ++d) {
            int count = 0;
            fs::directory_iterator end;
            for(fs::directory_iterator iter(*d) ; iter != end ; ++iter) {
                if(!fs::is_regular_file(iter->status())) continue;
                cv::Mat img = readImage(iter->path());
                cv::Mat descriptors = detectAndCompute(img);
                if(descriptors.empty()) {
                    BOOST_LOG_TRIVIAL(warning) << "<buildVocabulary> Error computing descriptors for image: " << iter->path().string();
                    continue;
                }

                bowtrainer.add(descriptors);
                count++;
                if(count % 100 == 0) {
                    BOOST_LOG_TRIVIAL(info) << "<buildVocabulary> Processed: " << count;
                }

                if(limit > 0 && count > limit) break;
            }
        }

        BOOST_LOG_TRIVIAL(warning) << "<buildVocabulary> Clustering..";
        cv::Mat vocabulary = bowtrainer.cluster();

        return vocabulary;
    }

    cv::Ptr<cv::BOWImgDescriptorExtractor> Besra::loadBOW(const cv::Mat &vocabulary) {
        cv::Ptr<cv::BOWImgDescriptorExtractor> bow = new cv::BOWImgDescriptorExtractor(extractor, matcher);
        bow->setVocabulary(vocabulary);
        return bow;
    }

    cv::Ptr<CvSVM> Besra::train(const fs::path &positive_dir, const fs::path &negative_dir, 
                                const cv::Mat &vocabulary, int limit) {
        cv::Ptr<cv::BOWImgDescriptorExtractor> bow = loadBOW(vocabulary);

        cv::Mat samples;
        cv::Mat labels;

        int count = 0;
        fs::directory_iterator end;
        for(fs::directory_iterator iter(positive_dir) ; iter != end ; ++iter) {
            if(!fs::is_regular_file(iter->status())) continue;

            cv::Mat img = readImage(iter->path());
            cv::Mat descriptors = detectAndCompute(img, bow);
            if(descriptors.empty()) {
                BOOST_LOG_TRIVIAL(warning) << "<trainPos> Error computing descriptors for image: " << iter->path().string();
                continue;
            }

            samples.push_back(descriptors);
            cv::Mat ones = cv::Mat::ones(descriptors.rows, 1, bow->descriptorType());
            labels.push_back(ones);
            count++;
            
            if(count % 100 == 0) {
                BOOST_LOG_TRIVIAL(info) << "<train> Positive images processed: " << count;
            }

            if(limit > 0 && count > limit) break;
        }

        count = 0;
        for(fs::directory_iterator iter(negative_dir) ; iter != end ; ++iter) {
            if(!fs::is_regular_file(iter->status())) continue;

            cv::Mat img = readImage(iter->path());
            cv::Mat descriptors = detectAndCompute(img, bow);
            if(descriptors.empty()) {
                BOOST_LOG_TRIVIAL(warning) << "<trainNeg> Error computing descriptors for image: " << iter->path().string();
                continue;
            }

            samples.push_back(descriptors);
            cv::Mat zeros = cv::Mat::zeros(descriptors.rows, 1, bow->descriptorType());
            labels.push_back(zeros);
            count++;
            
            if(count % 100 == 0) {
                BOOST_LOG_TRIVIAL(info) << "<train> Negative images processed: " << count;
            }

            if(limit > 0 && count > limit) break;
        }

        BOOST_LOG_TRIVIAL(info) << "Training SVM";

        CvSVMParams params;
        params.svm_type = CvSVM::C_SVC;
        params.kernel_type = CvSVM::LINEAR;
        params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);

        cv::Ptr<CvSVM> svm = new CvSVM();
        svm->train(samples, labels, cv::Mat(), cv::Mat(), params);

        return svm;
    }

    cv::Ptr<CvSVM> Besra::loadStatModel(const fs::path &cache, const cv::Mat &vocabulary) {
        cv::Ptr<CvSVM> svm = new CvSVM();
        svm->load(cache.string().c_str());
        return svm;
    }
}
