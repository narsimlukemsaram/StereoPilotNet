# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.

project(dw_samples_stereo_common C CXX)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES
    stereoCommon.hpp
    stereoCommon.cpp
    utils.hpp
    utils.cu
)

set(LIBRARIES
    dw_samples_framework
)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
cuda_add_library(${PROJECT_NAME} ${SOURCES} STATIC)
target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Samples")
