cmake_minimum_required(VERSION 1.8)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

#set(Boost_USE_STATIC_LIBS ON)
add_definitions(-DBOOST_LOG_DYN_LINK)

option(USE_GPU "Compile to use GPUs" OFF)

if(USE_GPU)
    add_definitions(-DUSE_GPU)
endif()

find_package(Boost 1.55 COMPONENTS log log_setup thread date_time filesystem system program_options REQUIRED)
find_package(OpenCV 2.3 REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIR})

add_executable(besra-trainer src/besra-trainer.cpp src/besra.cpp)
target_link_libraries( besra-trainer ${Boost_LIBRARIES} ${OpenCV_LIBS} )

add_executable(besra-classify src/besra-classify.cpp src/besra.cpp)
target_link_libraries( besra-classify ${Boost_LIBRARIES} ${OpenCV_LIBS} )

add_executable(besra-view src/besra-view.cpp src/besra.cpp)
target_link_libraries( besra-view ${Boost_LIBRARIES} ${OpenCV_LIBS} )
