#include "besra.hpp"
#include "boost/program_options.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    besra::init_log();

    fs::path cwd = fs::initial_path();
    int desc_sz;
    int dict_sz;
    int limit;
    std::string positive_path;
    std::string negative_path;
    std::string output_path;

    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "help message")
        ("positive,p", po::value<std::string>(&positive_path)->required(), "path to directory of positive images (crystals)")
        ("negative,n", po::value<std::string>(&negative_path)->required(), "path to directory of negative images (no crystals)")
        ("output,o", po::value<std::string>(), "path to output directory")
        ("limit,l", po::value<int>(&limit)->default_value(5000), "max number of images to process")
        ("descriptor-size,s", po::value<int>(&desc_sz)->default_value(64), "descriptor size")
        ("dictionary-size,d", po::value<int>(&dict_sz)->default_value(150), "dictionary size")
    ;

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        po::notify(vm);    
    } catch(po::error& e) { 
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
      std::cerr << desc << std::endl; 
      return 1; 
    } 

    fs::path positive_dir(positive_path);
    fs::path negative_dir(negative_path);
    fs::path output_dir(cwd);

    if(!fs::exists(positive_dir) || !fs::is_directory(positive_dir)) {
      std::cerr << "Invalid directory: " << positive_dir << std::endl; 
      return 1; 
    }

    if(!fs::exists(negative_dir) || !fs::is_directory(negative_dir)) {
      std::cerr << "Invalid directory: " << negative_dir << std::endl; 
      return 1; 
    }

    if(vm.count("output")) {
        output_dir = fs::path(vm["output"].as<std::string>());
        if( !fs::exists(output_dir) || !fs::is_directory(output_dir)) {
            std::cerr << "Invalid output directory: " << output_dir << std::endl; 
            return 1; 
        }
    }

    fs::path svm_cache_file(output_dir / fs::path("svm-train.xml"));
    fs::path vocab_cache_file(output_dir / fs::path("bow-vocab.yml"));

#ifdef USE_GPU
    int gpus = cv::gpu::getCudaEnabledDeviceCount();
    if(gpus > 0) {
        BOOST_LOG_TRIVIAL(info) << "Found " <<  gpus << " GPUs";
    } else {
        std::cerr << "USE_GPU is enabled and no GPUs found!" << std::endl; 
        return 1; 
    }
#endif
    BOOST_LOG_TRIVIAL(info) << "Extracting features";
    cv::BOWKMeansTrainer bowtrainer(dict_sz);
    bowtrainer.add(besra::extract_features_from_dir(positive_dir, desc_sz, limit));
    bowtrainer.add(besra::extract_features_from_dir(negative_dir, desc_sz, limit));

    BOOST_LOG_TRIVIAL(info) << "Building vocab";
    cv::Mat vocabulary = bowtrainer.cluster();

    BOOST_LOG_TRIVIAL(info) << "Storing vocab to file";
    cv::FileStorage vocab_fs(vocab_cache_file.string(), cv::FileStorage::WRITE);
    vocab_fs << "vocabulary" << vocabulary;
    vocab_fs.release();

    cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BruteForceMatcher<cv::L2<float> >());
    cv::Ptr<cv::DescriptorExtractor> extractor(new cv::SurfDescriptorExtractor(4,2,false));
    cv::BOWImgDescriptorExtractor bow(extractor,matcher);
    bow.setVocabulary(vocabulary);

    BOOST_LOG_TRIVIAL(info) << "Building training data";
    cv::Mat positive_training_data = besra::extract_features_from_dir(positive_dir, desc_sz, limit, &bow);
    cv::Mat negative_training_data = besra::extract_features_from_dir(negative_dir, desc_sz, limit, &bow);

    // Create labels
    cv::Mat samples(0, positive_training_data.cols, positive_training_data.type());
    cv::Mat labels(0, 1, CV_32FC1);

    samples.push_back(positive_training_data);
    cv::Mat crystals_yes = cv::Mat::ones(positive_training_data.rows, 1, CV_32FC1);
    labels.push_back(crystals_yes);

    samples.push_back(negative_training_data);
    cv::Mat crystals_no = cv::Mat::zeros(negative_training_data.rows, 1, CV_32FC1);
    labels.push_back(crystals_no);

    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);

    // Train the SVM
    CvSVM svm;

    BOOST_LOG_TRIVIAL(info) << "Training SVM";
    svm.train(samples, labels, cv::Mat(), cv::Mat(), params);

    BOOST_LOG_TRIVIAL(info) << "Saving training data";
    svm.save(svm_cache_file.string().c_str());

    BOOST_LOG_TRIVIAL(info) << "Done!";

    return 0;
}
