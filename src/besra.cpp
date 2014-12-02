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

    cv::Mat Besra::buildVocabulary(const fs::path &input_file, int clusterCount, int limit, int threads) {
        cv::BOWKMeansTrainer bowtrainer(clusterCount);

        std::pair<cv::Mat, cv::Mat> result = processImages(input_file, limit, threads);
        cv::Mat descriptors = result.first;
        for(int i = 0; i < descriptors.rows; i++) {
            bowtrainer.add(descriptors.row(i));
        }

        if(bowtrainer.descripotorsCount() == 0) {
            //TODO: throw exception or bail here
            BOOST_LOG_TRIVIAL(error) << "<buildVocabulary> No descriptors found! Can't perform clustering..";
        }
        
        BOOST_LOG_TRIVIAL(info) << "<buildVocabulary> Clustering.." << bowtrainer.getDescriptors().size();
        cv::Mat vocabulary = bowtrainer.cluster();

        return vocabulary;
    }

    std::pair<cv::Mat, cv::Mat> Besra::processImages(const fs::path &path, int limit, int threads, cv::Ptr<cv::BOWImgDescriptorExtractor> bow) {
        cv::Mat descriptors;
        cv::Mat labels;

        if(threads <= 0) {
            threads = 1;
        }

        cv::Ptr<besra::ImageQueue> queue = new besra::ImageQueue();
        besra::ImageProducer pp(1, queue);
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
            cv::Mat l = cons[i]->getLabels();
            descriptors.push_back(d);
            labels.push_back(l);
            delete cons[i];
        }

        return std::make_pair(descriptors, labels);
    }

    cv::Ptr<cv::BOWImgDescriptorExtractor> Besra::loadBOW(const cv::Mat &vocabulary) {
        cv::Ptr<cv::BOWImgDescriptorExtractor> bow = new cv::BOWImgDescriptorExtractor(extractor, matcher);
        bow->setVocabulary(vocabulary);
        return bow;
    }

    cv::Ptr<CvSVM> Besra::train(const fs::path &input_file, const cv::Mat &vocabulary, int limit, int threads) {
        cv::Ptr<cv::BOWImgDescriptorExtractor> bow = loadBOW(vocabulary);

        cv::Mat samples;
        cv::Mat labels;

        std::pair<cv::Mat, cv::Mat> result = processImages(input_file, limit, threads, bow);
        samples = result.first;
        labels = result.second;

        BOOST_LOG_TRIVIAL(info) << "Training SVM";

        CvSVMParams params;
        params.svm_type = CvSVM::C_SVC;
        params.kernel_type = CvSVM::LINEAR;
        //params.svm_type = CvSVM::C_SVC;
        //params.kernel_type = CvSVM::RBF;
        params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);

        cv::Ptr<CvSVM> svm = new CvSVM();
        svm->train(samples, labels, cv::Mat(), cv::Mat(), params);

        return svm;
    }

    float Besra::classify(const fs::path &path, cv::Ptr<cv::BOWImgDescriptorExtractor> bow, cv::Ptr<CvSVM> model) {
        cv::Mat img = readImage(path);
        cv::Mat features = detectAndCompute(img, bow);
        float res = model->predict(features);
        return res;
    }

    cv::Ptr<CvSVM> Besra::loadStatModel(const fs::path &cache, const cv::Mat &vocabulary) {
        cv::Ptr<CvSVM> svm = new CvSVM();
        svm->load(cache.string().c_str());
        return svm;
    }

    ImageConsumer::ImageConsumer(int id, cv::Ptr<ImageQueue> queue) {
        this->count = 0;
        this->id = id;
        this->queue = queue;
    }

    cv::Mat ImageConsumer::getDescriptors() {
        return descriptors;
    }

    cv::Mat ImageConsumer::getLabels() {
        return labels;
    }

    void ImageConsumer::operator () (besra::Besra &besra, cv::Ptr<cv::BOWImgDescriptorExtractor> bow) {
        ImageRecord rec;
        while(!queue->isDone()) {
            while(queue->pop(rec)) {
                cv::Mat img = besra.readImage(rec.path);
                cv::Mat d;
                if(bow == NULL) { 
                    d = besra.detectAndCompute(img);
                } else {
                    d = besra.detectAndCompute(img, bow);
                }

                count++;
                if(count % 100 == 0) {
                    BOOST_LOG_TRIVIAL(info) << "ImageConsumer Thread " << id 
                                               <<  " progress: " << count;
                }

                if(d.empty()) {
                    BOOST_LOG_TRIVIAL(warning) << "ImageConsumer Thread " << id 
                                               <<  " empty descriptors for image: " << rec.path;
                    continue;
                }
                descriptors.push_back(d);
                labels.push_back(rec.label);
            }
        }
        while(queue->pop(rec)) {
            cv::Mat img = besra.readImage(rec.path);
            cv::Mat d;
            if(bow == NULL) { 
                d = besra.detectAndCompute(img);
            } else {
                d = besra.detectAndCompute(img, bow);
            }

            count++;
            if(count % 100 == 0) {
                BOOST_LOG_TRIVIAL(info) << "ImageConsumer Thread " << id 
                                           <<  " progress: " << count;
            }

            if(d.empty()) {
                BOOST_LOG_TRIVIAL(warning) << "ImageConsumer Thread " << id 
                                           <<  " empty descriptors for image: " << rec.path;
                continue;
            }

            descriptors.push_back(d);
            labels.push_back(rec.label);
        }
    }

    ImageProducer::ImageProducer(int id, cv::Ptr<ImageQueue> queue) {
        this->id = id;
        this->queue=queue;
    }

    void ImageProducer::operator () (const fs::path &path, int limit) {
        int count = 0;
        if(fs::is_regular_file(path)) {
            std::ifstream ifs(path.c_str());
            std::string line;
            while(std::getline(ifs, line)) {
                std::vector<std::string> tokens;  
                boost::split(tokens, line, boost::is_any_of("\t,"));
                if(tokens.size() != 2) {
                    BOOST_LOG_TRIVIAL(error) << "ImageProducer Thread " << id 
                                               <<  " invalid format for line: " << line;
                    continue;
                }

                fs::path img_path(tokens[0]);
                if(!fs::is_regular_file(img_path)) continue;

                float label = 0;
                try {
                    label = std::stof(tokens[1]);
                } catch ( ... ) {
                    BOOST_LOG_TRIVIAL(error) << "ImageProducer Thread " << id 
                                               <<  " invalid label for image file: " << img_path;
                    continue;
                }

                ImageRecord rec(label, img_path);
                queue->push(rec);
                count++;
                if(limit > 0 && count >= limit) break;
            }
        } else {
            BOOST_LOG_TRIVIAL(warning) << "ImageProducer Thread " << id 
                                       <<  " invalid path: " << path;
        }
    }

    bool ImageQueue::empty() {
        boost::mutex::scoped_lock lock(mutex);
        return queue.empty();
    }

    void ImageQueue::push(const ImageRecord &rec) {
        boost::mutex::scoped_lock lock(mutex);
        queue.push(rec);
        lock.unlock();
        waitCondition.notify_one();
    }

    bool ImageQueue::pop(ImageRecord &rec) {
        boost::mutex::scoped_lock lock(mutex);
        if(queue.empty()) {
            return false;
        }

        rec = queue.front();
        queue.pop();

        return true;
    }

    void ImageQueue::markDone() {
        boost::mutex::scoped_lock lock(mutex);
        done = true;
    }

    bool ImageQueue::isDone() {
        return done;
    }

    ImageQueue::ImageQueue() {
        done = false;
    }

    ImageRecord::ImageRecord() {
    }
    ImageRecord::ImageRecord(float label, fs::path path) {
        this->label = label;
        this->path = path;
    }

}
