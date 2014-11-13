#include "besra.hpp"
#include "boost/program_options.hpp"

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


    return 0;
}
