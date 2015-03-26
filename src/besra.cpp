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

    Besra::Besra(int minHessian, std::string extractor, std::string detector, bool bayes) {
        this->bayes = bayes;

        if(extractor == "SURF") {
            this->extractor = cv::xfeatures2d::SURF::create(minHessian);
        } else if(extractor == "FREAK") {
            this->extractor = cv::xfeatures2d::FREAK::create();
        } else if(extractor == "BRISK") {
            this->extractor = cv::BRISK::create();
        } else if(extractor == "ORB") {
            this->extractor = cv::ORB::create();
        } else if(extractor == "KAZE") {
            this->extractor = cv::KAZE::create();
        } else if(extractor == "AKAZE") {
            this->extractor = cv::AKAZE::create();
        } else if(extractor == "BRIEF") {
            this->extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        } else if(extractor == "SIFT") {
            this->extractor = cv::xfeatures2d::SIFT::create();
        }

        if(detector == "SURF") {
            this->detector = cv::xfeatures2d::SURF::create(minHessian);
        } else if(detector == "BRISK") {
            this->detector = cv::BRISK::create();
        } else if(detector == "ORB") {
            this->detector = cv::ORB::create();
        } else if(detector == "KAZE") {
            this->detector = cv::KAZE::create();
        } else if(detector == "AKAZE") {
            this->detector = cv::AKAZE::create();
        } else if(detector == "MSER") {
            this->detector = cv::MSER::create();
        } else if(detector == "SIFT") {
            this->detector = cv::xfeatures2d::SIFT::create();
        } else if(detector == "FAST") {
            this->detector = cv::FastFeatureDetector::create();
        } else if(detector == "GFTT") {
            this->detector = cv::GFTTDetector::create();
        } else if(detector == "BLOB") {
            this->detector = cv::SimpleBlobDetector::create();
        }

        if(this->detector != NULL) {
            BOOST_LOG_TRIVIAL(info) << "Detector has descriptor type: " << this->detector->descriptorType();
            if(this->detector->descriptorType() == CV_32F) {
                BOOST_LOG_TRIVIAL(info) << "Float descriptor. Using BruteForce matcher";
                this->matcher = cv::DescriptorMatcher::create("BruteForce");
                //this->matcher = cv::DescriptorMatcher::create("FlannBased");
            } else {
                // assume binary?
                BOOST_LOG_TRIVIAL(info) << "Binary descriptor. Using BruteForce-Hamming matcher";
                this->matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
                /*
                cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::AutotunedIndexParams>();
                cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>();
                indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
                searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
                this->matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
                */
            }
        }

        if(this->detector != NULL && this->extractor != NULL) {
            if(this->detector->descriptorType() != this->extractor->descriptorType()) {
                throw std::invalid_argument("Invalid detector/extractor combination");
            }
        }


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

        int type = descriptors.type();
        if(type != CV_32F) {
            descriptors.convertTo(descriptors, CV_32F);
        }

        for(int i = 0; i < descriptors.rows; i++) {
            bowtrainer.add(descriptors.row(i));
        }

        if(bowtrainer.descriptorsCount() == 0) {
            //TODO: throw exception or bail here
            BOOST_LOG_TRIVIAL(error) << "<buildVocabulary> No descriptors found! Can't perform clustering..";
        }
        
        BOOST_LOG_TRIVIAL(info) << "<buildVocabulary> Clustering.." << bowtrainer.getDescriptors().size();
        cv::Mat vocabulary = bowtrainer.cluster();

        if(type != CV_32F) {
            vocabulary.convertTo(vocabulary, type);
        }

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

        int count = 0;
        std::ifstream ifs(path.c_str());
        #pragma omp parallel num_threads(threads)
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
        if(samples.type() != CV_32F) {
            samples.convertTo(samples, CV_32F);
        }

        labels = result.second;

        BOOST_LOG_TRIVIAL(info) << "Training StatsModel";

        if(this->bayes) {
            cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(samples, cv::ml::ROW_SAMPLE, labels);
            return cv::ml::StatModel::train<cv::ml::NormalBayesClassifier>(trainData);
        }

        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::LINEAR);
        svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 1000, FLT_EPSILON));

        svm->train(samples, cv::ml::ROW_SAMPLE, labels);
        cv::Ptr<cv::ml::StatModel> mod = svm;
        return mod;
    }

    float Besra::classify(const fs::path &path, cv::Ptr<cv::BOWImgDescriptorExtractor> bow, cv::Ptr<cv::ml::StatModel> model) {
        cv::Mat img = readImage(path);
        cv::Mat features = detectAndCompute(img, bow);

        if(features.empty()) {
            throw std::out_of_range("Empty features");
        }

        float res = model->predict(features);
        return res;
    }

    cv::Ptr<cv::ml::StatModel> Besra::loadStatModel(const fs::path &cache, const cv::Mat &vocabulary) {
        if(this->bayes) {
            return cv::ml::StatModel::load<cv::ml::NormalBayesClassifier>(cache.string());
        }
        return cv::ml::StatModel::load<cv::ml::SVM>(cache.string());
    }

}
