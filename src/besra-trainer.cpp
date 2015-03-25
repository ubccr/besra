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
#include "boost/program_options.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv) {

    fs::path cwd = fs::initial_path();
    int clusters;
    int threads;
    int minHessian;
    bool verbose;
    std::string classifier;
    std::string input_path;
    std::string output_path;
    std::string detector_str;
    std::string extractor_str;

    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "help message")
        ("input,i", po::value<std::string>(&input_path)->required(), "path to input file")
        ("output,o", po::value<std::string>(), "path to output directory")
        ("threads,t", po::value<int>(&threads)->default_value(1), "number of threads to spawn")
        ("clusters,c", po::value<int>(&clusters)->default_value(150), "clusters")
        ("hessian,k", po::value<int>(&minHessian)->default_value(600), "hessian threshold")
        ("vocab,b", po::value<std::string>(), "path to vocabulary cache file")
        ("verbose,v", po::bool_switch(&verbose)->default_value(false), "verbose output")
        ("classifier,j", po::value<std::string>(&classifier)->default_value("svm"), "classifier (svn, bayes)")
        ("detector,d", po::value<std::string>(&detector_str)->default_value("SURF"), "feature detector (SURF, BRISK, ORB, KAZE, AKAZE, MSER, SIFT, FAST, GFTT, BLOB)")
        ("extractor,e", po::value<std::string>(&extractor_str)->default_value("SURF"), "descriptor extractor (SURF, FREAK, BRISK, ORB, KAZE, AKAZE, BRIEF, SIFT)")
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

    besra::init_log(verbose);

    fs::path input_file(input_path);
    fs::path output_dir(cwd);

    if(!fs::is_regular_file(input_file)) {
      std::cerr << "Invalid input file: " << input_file << std::endl; 
      return 1; 
    }

    if(vm.count("output")) {
        output_dir = fs::path(vm["output"].as<std::string>());
        if( !fs::exists(output_dir) || !fs::is_directory(output_dir)) {
            std::cerr << "Invalid output directory: " << output_dir << std::endl; 
            return 1; 
        }
    }

    fs::path model_cache_file(output_dir / fs::path("stats-model.yml"));
    fs::path vocab_cache_file(output_dir / fs::path("bow-vocab.yml"));
    if(vm.count("vocab")) {
        vocab_cache_file = fs::path(vm["vocab"].as<std::string>());
        if( !fs::exists(vocab_cache_file) ) {
            std::cerr << "Invalid vocabulary cache file: " << vocab_cache_file << std::endl; 
            return 1; 
        }
    }


#ifdef USE_GPU
    int gpus = cv::gpu::getCudaEnabledDeviceCount();
    if(gpus > 0) {
        BOOST_LOG_TRIVIAL(info) << "Found " <<  gpus << " GPUs";
    } else {
        std::cerr << "USE_GPU is enabled and no GPUs found!" << std::endl; 
        return 1; 
    }
#endif
    besra::Besra besra(minHessian, extractor_str, detector_str); 
    if(besra.extractor == NULL) {
        std::cerr << "Invalid extractor: " << extractor_str << std::endl; 
        return 1; 
    }
    if(besra.detector == NULL) {
        std::cerr << "Invalid detector: " << detector_str << std::endl; 
        return 1; 
    }

    if(extractor_str != "SURF" || detector_str != "SURF") {
        // Only SURF appears to be thread safe at the moment
        threads = 1;
    }

    if(classifier == "bayes") {
        besra.bayes = true;
    }

    cv::Mat vocabulary;

    if(vm.count("vocab")) {
        BOOST_LOG_TRIVIAL(info) << "Loading vocab from file: " << vocab_cache_file.string();
        cv::FileStorage fs(vocab_cache_file.string(), cv::FileStorage::READ);
        fs["vocabulary"] >> vocabulary;
        fs.release();   
    } else {
        BOOST_LOG_TRIVIAL(info) << "Building vocab..";
        vocabulary = besra.buildVocabulary(input_file, clusters, threads);

        BOOST_LOG_TRIVIAL(info) << "Saving vocab to cache file: " << vocab_cache_file.string();
        cv::FileStorage vocab_fs(vocab_cache_file.string(), cv::FileStorage::WRITE);
        vocab_fs << "vocabulary" << vocabulary;
        vocab_fs.release();
    }


    BOOST_LOG_TRIVIAL(info) << "Building stats model..";
    cv::Ptr<cv::ml::StatModel> model = besra.train(input_file, vocabulary, threads);

    BOOST_LOG_TRIVIAL(info) << "Saving stats model to cache file: " << model_cache_file.string();
    model->save(model_cache_file.string());

    BOOST_LOG_TRIVIAL(info) << "Done!";

    return 0;
}
