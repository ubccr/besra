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
    std::string input_path;
    std::string model_cache_path;
    std::string vocab_cache_path;
    int limit;
    int threads;
    int minHessian;
    bool verbose;

    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "help message")
        ("input,i", po::value<std::string>(&input_path)->required(), "path to input directory")
        ("model,m", po::value<std::string>(&model_cache_path)->required(), "path to stats model cache file")
        ("vocab,b", po::value<std::string>(&vocab_cache_path)->required(), "path to vocabulary cache file")
        ("output,o", po::value<std::string>(), "path to output file")
        ("limit,l", po::value<int>(&limit)->default_value(0), "max number of images to process (0 = unlimited)")
        ("threads,t", po::value<int>(&threads)->default_value(1), "number of threads to spawn")
        ("hessian,k", po::value<int>(&minHessian)->default_value(600), "hessian threshold")
        ("verbose,v", po::bool_switch(&verbose)->default_value(false), "verbose output")
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

    fs::path input_dir(input_path);
    fs::path model_cache_file(model_cache_path);
    fs::path vocab_cache_file(vocab_cache_path);
    fs::path output_file(cwd / fs::path("besra-results.tsv"));

    if(!fs::exists(input_dir)) {
      std::cerr << "Invalid input file/directory: " << input_dir << std::endl; 
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
    if(fs::is_directory(input_dir)) {
        fs::directory_iterator end;
        for(fs::directory_iterator iter(input_dir) ; iter != end ; ++iter) {
            if(!fs::is_regular_file(iter->status())) continue;
            std::string filepath = iter->path().string();

            try{
                float res = besra.classify(filepath, bow, model);
                BOOST_LOG_TRIVIAL(info) << "Class: " << res << " Image: " << filepath;
                fout << filepath << "\t" << res << std::endl;
                count++;
            } catch(cv::Exception& e) { 
                const char* err_msg = e.what();
                BOOST_LOG_TRIVIAL(error) << "Failed to classify image: " << filepath << " error: " << err_msg;
            }

            if(limit > 0 && count > limit) break;
        }
    } else if(fs::is_regular_file(input_dir)) {

#ifdef OPENMP_FOUND
        if(threads > limit) {
            threads = limit;
        }
        omp_set_num_threads(threads);
#endif

        std::ifstream ifs(input_dir.c_str());
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

                std::vector<std::string> tokens;  
                boost::split(tokens, line, boost::is_any_of("\t,"));
                if(tokens.size() == 0) {
                    continue;
                }


                fs::path img_path(tokens[0]);
                if(!fs::is_regular_file(img_path)) continue;
                std::string filepath = img_path.string();

                try{
                    float res = besra.classify(filepath, bow, model);
                    tcount++;
                    #pragma omp critical
                    {
                        fout << filepath << "\t" << res << std::endl;
                        count++;
                    }
                } catch(cv::Exception& e) { 
                    const char* err_msg = e.what();
                    BOOST_LOG_TRIVIAL(error) << "Failed to classify image: " << filepath << " error: " << err_msg;
                }

                if(tcount % 100 == 0) {
#ifdef OPENMP_FOUND
                    BOOST_LOG_TRIVIAL(info) <<  "Thread " << omp_get_thread_num() << " progress: " << tcount;
#else
                    BOOST_LOG_TRIVIAL(info) <<  "progress: " << tcount;
#endif
                }

                if(limit > 0 && count > limit) break;
            }
        }
    } else {
        BOOST_LOG_TRIVIAL(error) << "Invalid input file/directory: " << input_dir;
    }

    fout.close();
    BOOST_LOG_TRIVIAL(info) << "Done!";

    return 0;
}
