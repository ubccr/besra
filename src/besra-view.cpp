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
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "help message")
        ("image,i", po::value<std::string>(&img_path)->required(), "path to image file")
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

    besra::Besra besra; 

    /* TODO: display keypoints in image */

    std::string line = "hello,world,\"bob\",a,looski";

    boost::escaped_list_separator<char> sep('\\', ',', '\"');
    boost::tokenizer<boost::escaped_list_separator<char> > tk(line, sep);
    for(boost::tokenizer<boost::escaped_list_separator<char> >::iterator i(tk.begin()); i!=tk.end();++i) {
        std::cout << *i << std::endl;
    }

    cv::Ptr<besra::PathQueue> queue = new besra::PathQueue();
    besra::PathProducer pp(1, queue);

    std::vector<fs::path> dirs;
    fs::path p1("/ifs/projects/ccrstaff/aebruno2/hwi/train/crystal/");
    dirs.push_back(p1);

    boost::thread pt(boost::ref(pp), dirs, 0);

    int max_threads = 10;

    besra::ImageConsumer *cons[max_threads];
    boost::thread_group g;
    for(int i = 0; i < max_threads; i++) {
        besra::ImageConsumer *ic = new besra::ImageConsumer(i, queue);
        g.add_thread(new boost::thread(boost::ref(*ic), besra));
        cons[i] = ic;
    }

    pt.join();
    queue->markDone();
    g.join_all();

    for(int i = 0; i < max_threads; i++) {
        std::cout << cons[i]->id << ": " << cons[i]->getDescriptors().rows << std::endl;
    }

    return 0;
}
