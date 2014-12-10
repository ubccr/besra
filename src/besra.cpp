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

    void init_log(bool verbose) {
        boost::log::register_simple_formatter_factory< boost::log::trivial::severity_level, char >("Severity");
        boost::log::add_console_log(
            std::cerr, 
            boost::log::keywords::format = "%TimeStamp% [%Severity%]: %Message%",
            boost::log::keywords::auto_flush = true
        );

        boost::log::add_common_attributes();

        if(verbose) {
            boost::log::core::get()->set_filter(
                boost::log::trivial::severity >= boost::log::trivial::info
            );
        } else {
            boost::log::core::get()->set_filter(
                boost::log::trivial::severity >= boost::log::trivial::error
            );
        }

    }

    Besra::Besra(int minHessian, std::string matcher, std::string detector) {
        this->matcher = cv::DescriptorMatcher::create(matcher);
        this->extractor = cv::xfeatures2d::SURF::create(minHessian);
        this->detector = cv::xfeatures2d::SURF::create(minHessian);
#ifdef USE_GPU
        gpu_surf = new cv::gpu::SURF_GPU(minHessian);
#endif
    }

    cv::Mat Besra::readImage(const fs::path &file) {
        return cv::imread(file.string(), cv::IMREAD_GRAYSCALE);
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

    cv::Mat Besra::buildVocabulary(const fs::path &input_file, int clusterCount, int threads) {
        cv::BOWKMeansTrainer bowtrainer(clusterCount);

        std::pair<cv::Mat, cv::Mat> result = processImages(input_file, threads);
        cv::Mat descriptors = result.first;
        for(int i = 0; i < descriptors.rows; i++) {
            bowtrainer.add(descriptors.row(i));
        }

        if(bowtrainer.descriptorsCount() == 0) {
            //TODO: throw exception or bail here
            BOOST_LOG_TRIVIAL(error) << "<buildVocabulary> No descriptors found! Can't perform clustering..";
        }
        
        BOOST_LOG_TRIVIAL(info) << "<buildVocabulary> Clustering.." << bowtrainer.getDescriptors().size();
        cv::Mat vocabulary = bowtrainer.cluster();

        return vocabulary;
    }

    bool Besra::processLine(std::string line, cv::Mat &descriptors, int &label,
                            cv::Ptr<cv::BOWImgDescriptorExtractor> bow) {
        std::vector<std::string> tokens;  
        boost::split(tokens, line, boost::is_any_of("\t,"));
        if(tokens.size() != 2) {
            BOOST_LOG_TRIVIAL(error) <<  "Invalid record format for line: " << line;
            return false;
        }

        fs::path img_path(tokens[0]);
        if(!fs::is_regular_file(img_path)) return false;

        label = 0;
        try {
            label = std::stoi(tokens[1]);
        } catch ( ... ) {
            BOOST_LOG_TRIVIAL(error) <<  "Invalid label for image file: " << img_path;
            return false;
        }

        cv::Mat img = readImage(img_path);
        if(bow == NULL) { 
            descriptors = detectAndCompute(img);
        } else {
            descriptors = detectAndCompute(img, bow);
        }

        if(descriptors.empty()) {
            BOOST_LOG_TRIVIAL(warning) <<  "Empty descriptors for image: " << img_path;
            return false;
        }

        return true;
    }

    std::pair<cv::Mat, cv::Mat> Besra::processImages(const fs::path &path, int threads, 
                                                     cv::Ptr<cv::BOWImgDescriptorExtractor> bow) {
        cv::Mat descriptors;
        cv::Mat labels;

        if(threads <= 0) {
            threads = 1;
        }

#ifdef OPENMP_FOUND
        omp_set_num_threads(threads);
#endif

        int count = 0;
        std::ifstream ifs(path.c_str());
        #pragma omp parallel
        {
            int tcount = 0;
            std::string line;
            while(true) {

                #pragma omp critical(input)
                {
                    std::getline(ifs, line);
                }
                if(ifs.eof()) break;


                cv::Mat d;
                int label;
                if(processLine(line, d, label, bow)) {
                    tcount++;
                    #pragma omp critical
                    {
                        descriptors.push_back(d);
                        labels.push_back(label);
                        count++;
                    }
                }

                if(tcount % 100 == 0) {
#ifdef OPENMP_FOUND
                    BOOST_LOG_TRIVIAL(info) <<  "Thread " << omp_get_thread_num() << " progress: " << tcount;
#else
                    BOOST_LOG_TRIVIAL(info) <<  "progress: " << tcount;
#endif
                }
            }
        }

        return std::make_pair(descriptors, labels);
    }

    cv::Ptr<cv::BOWImgDescriptorExtractor> Besra::loadBOW(const cv::Mat &vocabulary) {
        cv::Ptr<cv::BOWImgDescriptorExtractor> bow = cv::makePtr<cv::BOWImgDescriptorExtractor>(extractor, matcher);
        bow->setVocabulary(vocabulary);
        return bow;
    }

    cv::Ptr<cv::ml::StatModel> Besra::train(const fs::path &input_file, const cv::Mat &vocabulary, int threads) {
        cv::Ptr<cv::BOWImgDescriptorExtractor> bow = loadBOW(vocabulary);

        cv::Mat samples;
        cv::Mat labels;

        std::pair<cv::Mat, cv::Mat> result = processImages(input_file, threads, bow);
        samples = result.first;
        labels = result.second;

        BOOST_LOG_TRIVIAL(info) << "Training SVM";

        cv::ml::SVM::Params params;
        params.svmType = cv::ml::SVM::C_SVC;
        params.kernelType = cv::ml::SVM::LINEAR;
        params.termCrit = cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 1000, FLT_EPSILON);

        return cv::ml::StatModel::train<cv::ml::SVM>(samples, cv::ml::ROW_SAMPLE, labels, params);
    }

    float Besra::classify(const fs::path &path, cv::Ptr<cv::BOWImgDescriptorExtractor> bow, cv::Ptr<cv::ml::StatModel> model) {
        cv::Mat img = readImage(path);
        cv::Mat features = detectAndCompute(img, bow);
        float res = model->predict(features);
        return res;
    }

    cv::Ptr<cv::ml::StatModel> Besra::loadStatModel(const fs::path &cache, const cv::Mat &vocabulary) {
        return cv::ml::StatModel::load<cv::ml::SVM>(cache.string());
    }

}
