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

    cv::Mat extract_features_from_image(fs::path image_path, cv::BOWImgDescriptorExtractor *bow) {
        std::vector<cv::KeyPoint> keypoints; 
        cv::Mat descriptors;
        cv::Mat img = cv::imread(image_path.string(), CV_LOAD_IMAGE_GRAYSCALE);

#ifdef USE_GPU
        cv::gpu::SURF_GPU surf(1000, 4, 2, false, 0.01f, false);
        cv::gpu::GpuMat gpu_keypoints;
        cv::gpu::GpuMat gpu_descriptors;
        cv::gpu::GpuMat gpu_img;
        gpu_img.upload(img);
        surf(gpu_img, cv::gpu::GpuMat(), gpu_keypoints, gpu_descriptors);
        surf.downloadKeypoints(gpu_keypoints, keypoints); 
        if(bow != NULL) {
            bow->compute(gpu_img, keypoints, descriptors);
        } else {
            gpu_descriptors.download(descriptors);
        }
#else
        cv::SurfDescriptorExtractor extractor;
        cv::SurfFeatureDetector detector(1000, 4, 2);
        detector.detect(img, keypoints);
        if(bow != NULL) {
            bow->compute(img, keypoints, descriptors);
        } else {
            extractor.compute(img, keypoints, descriptors);
        }
#endif

        return descriptors;
    }

    cv::Mat extract_features_from_dir(fs::path dir, int desc_sz, int limit, cv::BOWImgDescriptorExtractor *bow) {
        cv::Mat features(0, desc_sz, CV_32F);

        int count = 0;
        fs::directory_iterator end;
        for(fs::directory_iterator iter(dir) ; iter != end ; ++iter) {
            if(!fs::is_regular_file(iter->status())) continue;

            features.push_back(extract_features_from_image(iter->path(), bow));

            count++;
            
            if(count % 100 == 0) {
                BOOST_LOG_TRIVIAL(info) << "Processed: " << count;
            }

            if(count > limit) break;
        }

        return features;
    }

}
