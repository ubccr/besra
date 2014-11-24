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
#include <fstream>
#include "boost/program_options.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    besra::init_log();

    fs::path cwd = fs::initial_path();
    std::string input_path;
    std::string model_cache_path;
    std::string vocab_cache_path;
    int limit;
    int minHessian;

    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "help message")
        ("input,i", po::value<std::string>(&input_path)->required(), "path to input directory")
        ("model,m", po::value<std::string>(&model_cache_path)->required(), "path to stats model cache file")
        ("vocab,v", po::value<std::string>(&vocab_cache_path)->required(), "path to vocabulary cache file")
        ("output,o", po::value<std::string>(), "path to output file")
        ("limit,l", po::value<int>(&limit)->default_value(0), "max number of images to process (0 = unlimited)")
        ("hessian,k", po::value<int>(&minHessian)->default_value(600), "hessian threshold")
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
    fs::path model_cache_file(model_cache_path);
    fs::path vocab_cache_file(vocab_cache_path);
    fs::path output_file(cwd / fs::path("besra-results.tsv"));

    if(!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
      std::cerr << "Invalid input directory: " << input_dir << std::endl; 
      return 1; 
    }

    if(!fs::exists(model_cache_file)) {
      std::cerr << "Invalid svm cache file: " << model_cache_file << std::endl; 
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

    besra::Besra besra(minHessian); 

    BOOST_LOG_TRIVIAL(info) << "Loading vocab from file: " << vocab_cache_file.string();
    cv::Mat vocabulary;
    cv::FileStorage fs(vocab_cache_file.string(), cv::FileStorage::READ);
    fs["vocabulary"] >> vocabulary;
    fs.release();   

    cv::Ptr<cv::BOWImgDescriptorExtractor> bow = besra.loadBOW(vocabulary);

    BOOST_LOG_TRIVIAL(info) << "Loading stats model from file: " << model_cache_file.string();

    cv::Ptr<CvSVM> model = besra.loadStatModel(model_cache_file, vocabulary);

    BOOST_LOG_TRIVIAL(info) << "Classifying data..";

    int count = 0;
    fs::directory_iterator end;
    for(fs::directory_iterator iter(input_dir) ; iter != end ; ++iter) {
        if(!fs::is_regular_file(iter->status())) continue;
        std::string filepath = iter->path().string();

        try{
            cv::Mat img = besra.readImage(filepath);
            cv::Mat features = besra.detectAndCompute(img, bow);
            float res = model->predict(features);
            BOOST_LOG_TRIVIAL(info) << "Class: " << res << " Image: " << filepath;
            fout << res << "\t" << filepath << std::endl;
            count++;
        } catch(cv::Exception& e) { 
            const char* err_msg = e.what();
            BOOST_LOG_TRIVIAL(error) << "Failed to classify image: " << filepath << " error: " << err_msg;
        }

        if(limit > 0 && count > limit) break;
    }

    fout.close();
    BOOST_LOG_TRIVIAL(info) << "Done!";

    return 0;
}
