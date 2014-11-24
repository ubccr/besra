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

    Besra::Besra(int minHessian) {
        //matcher = new cv::BruteForceMatcher< cv::L2<float> >();
        matcher = new cv::FlannBasedMatcher();
        extractor = new cv::SurfDescriptorExtractor();
        detector = new cv::SurfFeatureDetector(minHessian);
#ifdef USE_GPU
        gpu_surf = new cv::gpu::SURF_GPU(minHessian);
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

    cv::Mat Besra::buildVocabulary(std::vector<fs::path> paths, int clusterCount, int limit, int threads) {
        cv::BOWKMeansTrainer bowtrainer(clusterCount);

        for(std::vector<fs::path>::iterator d = paths.begin(); d != paths.end(); ++d) {
            cv::Mat descriptors = processPath(*d, limit, threads);
            for(int i = 0; i < descriptors.rows; i++) {
                bowtrainer.add(descriptors.row(i));
            }
        }
        
        BOOST_LOG_TRIVIAL(warning) << "<buildVocabulary> Clustering.." << bowtrainer.getDescriptors().size();
        cv::Mat vocabulary = bowtrainer.cluster();

        return vocabulary;
    }

    cv::Mat Besra::processPath(const fs::path &path, int limit, int threads, cv::Ptr<cv::BOWImgDescriptorExtractor> bow) {
        cv::Mat descriptors;

        if(threads > 0) {
            cv::Ptr<besra::PathQueue> queue = new besra::PathQueue();
            besra::PathProducer pp(1, queue);
            boost::thread pt(boost::ref(pp), path, limit);

            besra::ImageConsumer *cons[threads];
            boost::thread_group g;
            for(int i = 0; i < threads; i++) {
                besra::ImageConsumer *ic = new besra::ImageConsumer(i, queue);
                g.add_thread(new boost::thread(boost::ref(*ic), *this, bow));
                cons[i] = ic;
            }

            pt.join();
            queue->markDone();
            g.join_all();

            for(int i = 0; i < threads; i++) {
                BOOST_LOG_TRIVIAL(info) << "ImageConsumer Thread " << cons[i]->id << " rows: " 
                                        << cons[i]->getDescriptors().rows;
                cv::Mat d = cons[i]->getDescriptors();
                descriptors.push_back(d);
                delete cons[i];
            }
        } else {
            int count = 0;
            fs::directory_iterator end;
            for(fs::directory_iterator iter(path) ; iter != end ; ++iter) {
                if(!fs::is_regular_file(iter->status())) continue;

                cv::Mat img = readImage(iter->path());
                cv::Mat d;
                if(bow == NULL) { 
                    d = detectAndCompute(img);
                } else {
                    d = detectAndCompute(img, bow);
                }

                if(d.empty()) {
                    BOOST_LOG_TRIVIAL(warning) << "Empty descriptors for image: " << iter->path().string();
                    continue;
                }

                descriptors.push_back(d);
                count++;
                
                if(count % 100 == 0) {
                    BOOST_LOG_TRIVIAL(info) << "Proccessed images processed: " << count;
                }

                if(limit > 0 && count >= limit) break;
            }
        }

        return descriptors;
    }

    cv::Ptr<cv::BOWImgDescriptorExtractor> Besra::loadBOW(const cv::Mat &vocabulary) {
        cv::Ptr<cv::BOWImgDescriptorExtractor> bow = new cv::BOWImgDescriptorExtractor(extractor, matcher);
        bow->setVocabulary(vocabulary);
        return bow;
    }

    cv::Ptr<CvSVM> Besra::train(const fs::path &positive, const fs::path &negative, 
                                const cv::Mat &vocabulary, int limit, int threads) {
        cv::Ptr<cv::BOWImgDescriptorExtractor> bow = loadBOW(vocabulary);

        cv::Mat samples;
        cv::Mat labels;

        cv::Mat pos_descriptors = processPath(positive, limit, threads, bow);
        samples.push_back(pos_descriptors);
        cv::Mat ones = cv::Mat::ones(pos_descriptors.rows, 1, bow->descriptorType());
        labels.push_back(ones);

        cv::Mat neg_descriptors = processPath(negative, limit, threads, bow);
        samples.push_back(neg_descriptors);
        cv::Mat zeros = cv::Mat::zeros(neg_descriptors.rows, 1, bow->descriptorType());
        labels.push_back(zeros);

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

    ImageConsumer::ImageConsumer(int id, cv::Ptr<PathQueue> queue) {
        this->id = id;
        this->queue = queue;
    }

    cv::Mat ImageConsumer::getDescriptors() {
        return descriptors;
    }

    void ImageConsumer::operator () (besra::Besra &besra, cv::Ptr<cv::BOWImgDescriptorExtractor> bow) {
        fs::path path;
        while(!queue->isDone()) {
            while(queue->pop(path)) {
                cv::Mat img = besra.readImage(path);
                cv::Mat d;
                if(bow == NULL) { 
                    d = besra.detectAndCompute(img);
                } else {
                    d = besra.detectAndCompute(img, bow);
                }

                if(d.empty()) {
                    BOOST_LOG_TRIVIAL(warning) << "ImageConsumer Thread " << id 
                                               <<  " empty descriptors for image: " << path;
                    continue;
                }
                descriptors.push_back(d);
            }
        }
        while(queue->pop(path)) {
            cv::Mat img = besra.readImage(path);
            cv::Mat d;
            if(bow == NULL) { 
                d = besra.detectAndCompute(img);
            } else {
                d = besra.detectAndCompute(img, bow);
            }

            if(d.empty()) {
                BOOST_LOG_TRIVIAL(warning) << "ImageConsumer Thread " << id 
                                           <<  " empty descriptors for image: " << path;
                continue;
            }

            descriptors.push_back(d);
        }
    }

    PathProducer::PathProducer(int id, cv::Ptr<PathQueue> queue) {
        this->id = id;
        this->queue=queue;
    }

    void PathProducer::operator () (const fs::path &path, int limit) {
        int count = 0;
        fs::directory_iterator end;
        for(fs::directory_iterator iter(path) ; iter != end ; ++iter) {
            if(!fs::is_regular_file(iter->status())) continue;
            queue->push(iter->path());
            count++;
            if(limit > 0 && count >= limit) break;
        }
    }

    bool PathQueue::empty() {
        boost::mutex::scoped_lock lock(mutex);
        return queue.empty();
    }

    void PathQueue::push(const fs::path &path) {
        boost::mutex::scoped_lock lock(mutex);
        queue.push(path);
        lock.unlock();
        waitCondition.notify_one();
    }

    bool PathQueue::pop(fs::path &path) {
        boost::mutex::scoped_lock lock(mutex);
        if(queue.empty()) {
            return false;
        }

        path = queue.front();
        queue.pop();

        return true;
    }

    void PathQueue::markDone() {
        boost::mutex::scoped_lock lock(mutex);
        done = true;
    }

    bool PathQueue::isDone() {
        return done;
    }

    PathQueue::PathQueue() {
        done = false;
    }

}
