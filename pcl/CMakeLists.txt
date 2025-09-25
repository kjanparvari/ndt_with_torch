cmake_minimum_required(VERSION 3.10)
project(eval_pcl)

set(CMAKE_CXX_STANDARD 17)

# find_package(Open3D REQUIRED)
find_package(PCL REQUIRED)
find_package(YAML-CPP REQUIRED)
find_package(tensorboard_logger REQUIRED)


include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(eval_pcl eval_pcl.cpp KittiOdometryDataset.cpp)

target_link_libraries(eval_pcl yaml-cpp ${PCL_LIBRARIES} tensorboard_logger)

set_target_properties(eval_pcl PROPERTIES
    BUILD_RPATH "/usr/local/lib"
    INSTALL_RPATH "/usr/local/lib")


# To build the project, run the following commands in ./pcl directory:
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# cmake --build build --target eval_pcl -j