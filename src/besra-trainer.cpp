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
    besra::init_log();

    fs::path cwd = fs::initial_path();
    int clusters;
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
        ("limit,l", po::value<int>(&limit)->default_value(0), "max number of images to process (0 = unlimited)")
        ("clusters,c", po::value<int>(&clusters)->default_value(150), "clusters")
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

    fs::path model_cache_file(output_dir / fs::path("stats-model.xml"));
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
    besra::Besra besra; 

    std::vector<fs::path> dirs;
    dirs.push_back(positive_dir);
    dirs.push_back(negative_dir);

    BOOST_LOG_TRIVIAL(info) << "Building vocab..";
    cv::Mat vocabulary = besra.buildVocabulary(dirs, clusters, limit);

    BOOST_LOG_TRIVIAL(info) << "Saving vocab to cache file: " << vocab_cache_file.string();
    cv::FileStorage vocab_fs(vocab_cache_file.string(), cv::FileStorage::WRITE);
    vocab_fs << "vocabulary" << vocabulary;
    vocab_fs.release();

    BOOST_LOG_TRIVIAL(info) << "Building stats model..";
    cv::Ptr<CvSVM> model = besra.train(positive_dir, negative_dir, vocabulary, limit);

    BOOST_LOG_TRIVIAL(info) << "Saving stats model to cache file: " << model_cache_file.string();
    model->save(model_cache_file.string().c_str());

    BOOST_LOG_TRIVIAL(info) << "Done!";

    return 0;
}
