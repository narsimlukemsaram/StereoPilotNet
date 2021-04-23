# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

project(sample_pilotnet C CXX)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(PUBLIC_DOCS
    README.md
)
set(SOURCES
    main.cpp
)

set(LIBRARIES
    dw_samples_pilotnet_common
    dw_samples_stereo_common
    ${Driveworks_LIBRARIES}
)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
cuda_add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Samples")

# ------------------------------------------------------------------------------
# Install target
# ------------------------------------------------------------------------------
sdk_add_sample(${PROJECT_NAME})
