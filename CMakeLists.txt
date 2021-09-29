# Copyright: 2012 Takato Horii
# Author: Takato Horii
# 

cmake_minimum_required(VERSION 2.6)
set(PROJECTNAME MultimodalDeepBeliefNet)
project(${PROJECTNAME})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
#find_package( Eigen3 REQUIRED )

set(folder_source src/main.cpp src/RBMBase.cpp src/ConditionalRBMBase.cpp src/RecurrentTemporalRBMBase.cpp src/GBRTRBM.cpp src/BBRTRBM.cpp)

set(folder_header src/RBMBase.h src/ConditionalRBMBase.h src/RecurrentTemporalRBMBase.h src/GBRTRBM.h src/BBRTRBM.h)

SOURCE_GROUP("Source Files" FILES ${folder_source})
SOURCE_GROUP("Header Files" FILES ${folder_header})

include_directories(C:/eigen/eigen_3_2_7)
add_executable(${PROJECTNAME} ${folder_source} ${folder_header})
install(TARGETS ${PROJECTNAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin)
