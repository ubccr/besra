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
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>

namespace po = boost::program_options;

int main(int argc, char** argv) {
    besra::init_log();

    std::string img_path;
    std::string detector_str;
    int minHessian;
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "help message")
        ("hessian,k", po::value<int>(&minHessian)->default_value(600), "hessian threshold")
        ("image,i", po::value<std::string>(&img_path)->required(), "path to image file")
        ("detector,d", po::value<std::string>(&detector_str)->default_value("SURF"), "feature detector (SURF, BRISK, ORB, KAZE, AKAZE, MSER, SIFT, FAST, GFTT, BLOB)")
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

    fs::path img_file(img_path);

    if(!fs::exists(img_file)) {
      std::cerr << "Invalid image file: " << img_file << std::endl; 
      std::cerr << desc << std::endl; 
      return 1; 
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

    besra::Besra besra(minHessian, "SURF", detector_str); 
    if(besra.detector == NULL) {
        std::cerr << "Invalid detector: " << detector_str << std::endl; 
        return 1; 
    }

    cv::Mat img = besra.readImage(img_file);
    std::vector<cv::KeyPoint> keypoints = besra.detectKeypoints(img);

    cv::Mat img_keypoints;
      
    cv::drawKeypoints(img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("Keypoints: " + img_file.filename().string(), img_keypoints);

    cv::waitKey(0);

    return 0;
}
