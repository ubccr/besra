#include "besra.hpp"
#include <fstream>
#include "boost/program_options.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    besra::init_log();

    int desc_sz;
    int dict_sz;
    int limit;
    std::string input_path;
    std::string svm_cache_path;
    std::string vocab_cache_path;
    std::string output_path;

    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "help message")
        ("input,i", po::value<std::string>(&input_path)->required(), "path to input directory")
        ("svm-cache,j", po::value<std::string>(&svm_cache_path)->required(), "path to SVM cache file")
        ("vocab-cache,v", po::value<std::string>(&vocab_cache_path)->required(), "path to vocab cache file")
        ("output,o", po::value<std::string>(), "path to output file")
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

    fs::path input_dir(input_path);
    fs::path svm_cache_file(svm_cache_path);
    fs::path vocab_cache_file(vocab_cache_path);
    fs::path output_file(output_path / fs::path("besra-results.tsv"));

    if(!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
      std::cerr << "Invalid input directory: " << input_dir << std::endl; 
      return 1; 
    }

    if(!fs::exists(svm_cache_file)) {
      std::cerr << "Invalid svm cache file: " << svm_cache_file << std::endl; 
      std::cerr << desc << std::endl; 
      return 1; 
    }

    if(!fs::exists(vocab_cache_file)) {
      std::cerr << "Invalid vocab cache file: " << vocab_cache_file << std::endl; 
      std::cerr << desc << std::endl; 
      return 1; 
    }

    if(vm.count("output")) {
        output_file = fs::path(vm["output"].as<std::string>());
    }

    std::ofstream fout;
    fout.open(output_file.string().c_str());

#ifdef USE_GPU
    int gpus = cv::gpu::getCudaEnabledDeviceCount();
    if(gpus > 0) {
        BOOST_LOG_TRIVIAL(info) << "Found " <<  gpus << " GPUs";
    } else {
        std::cerr << "USE_GPU is enabled and no GPUs found!" << std::endl; 
        return 1; 
    }
#endif

    BOOST_LOG_TRIVIAL(info) << "Loading vocab";
    cv::Mat vocabulary;
    cv::FileStorage fs(vocab_cache_file.string(), cv::FileStorage::READ);
    fs["vocabulary"] >> vocabulary;
    fs.release();   

    cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BruteForceMatcher<cv::L2<float> >());
    cv::Ptr<cv::DescriptorExtractor> extractor(new cv::SurfDescriptorExtractor(4,2,false));
    cv::BOWImgDescriptorExtractor bow(extractor,matcher);
    bow.setVocabulary(vocabulary);

    BOOST_LOG_TRIVIAL(info) << "Loading training data";

    CvSVM svm;
    svm.load(svm_cache_file.string().c_str());

    BOOST_LOG_TRIVIAL(info) << "Classifying data";

    int count = 0;
    fs::directory_iterator end;
    for(fs::directory_iterator iter(input_dir) ; iter != end ; ++iter) {
        if(!fs::is_regular_file(iter->status())) continue;
        std::string filepath = iter->path().string();

        try{
            cv::Mat features = besra::extract_features_from_image(filepath, &bow);
            float res = svm.predict(features);
            BOOST_LOG_TRIVIAL(info) << "Class: " << res << " Image: " << filepath;
            fout << res << "\t" << filepath << std::endl;
            count++;
        } catch(...) { 
            BOOST_LOG_TRIVIAL(error) << "Failed to classify image: " << filepath;
        }

        if(count > limit) break;
    }

    fout.close();
    BOOST_LOG_TRIVIAL(info) << "Done!";

    return 0;
}
