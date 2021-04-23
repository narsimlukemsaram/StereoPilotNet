/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2015-2017 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef SAMPLES_STEREO_COMMON_STEREOCOMMON_HPP__
#define SAMPLES_STEREO_COMMON_STEREOCOMMON_HPP__

// Driveworks
#include <dw/Driveworks.h>

// Sample framework
#include <framework/DriveWorksSample.hpp>
#include <framework/SimpleStreamer.hpp>
#include <framework/SimpleRenderer.hpp>
#include <framework/SimpleCamera.hpp>

// Pyramids
#include <dw/sfm/SFM.h>

// Stereo
#include <dw/stereo/Stereo.h>

// Stereo sample common
#include "utils.hpp"

/**
 * Class that holds functions and variables common to all stereo samples
 */
using namespace dw_samples::common;

class StereoApp: public DriveWorksSample
{
public:
    explicit StereoApp(const ProgramArguments& args);

    bool onInitialize() override;
    void onRelease() override;

    bool initSDK();
    virtual bool initRenderer();

    void setInputProperties(dwImageProperties imageProperties);
    bool readStereoImages(dwImageCUDA *gStereoImages[DW_STEREO_SIDE_BOTH]);

    // Create calibrated cameras based on the stereo rig (assumed the stereo cameras are the first)
    bool getCameraFromRig();

    dwCalibratedCameraHandle_t m_cameraModel[DW_STEREO_SIDE_COUNT];
    dwRigConfigurationHandle_t m_rigConfiguration = DW_NULL_HANDLE;

    float32_t m_invalidThreshold = -1.0f;

    // for color code rendering
    float32_t m_colorGain;

    // properties of the input image coming from the sensor
    dwImageProperties m_imageProperties{};
    dwImageProperties m_inputCameraImageProperties{};
protected:
    // ------------------------------------------------
    // Driveworks Context and SAL
    // ------------------------------------------------
    dwContextHandle_t m_context = DW_NULL_HANDLE;
    dwSALHandle_t m_sal = DW_NULL_HANDLE;
    dwRendererHandle_t m_renderer = DW_NULL_HANDLE;


    // initialize stereo input from two videos
    std::unique_ptr<SimpleCamera> initFromVideo(const std::string &videoFName);
    // initialize stereo video from a USB camera
    std::unique_ptr<SimpleCamera> initFromCamera();
    // creates camera sensor based on one of the functions above
    std::unique_ptr<SimpleCamera> initInput(const dwSensorParams &params);

    // camera can have single image input side-by-side (ZEN camera) or two separate inputs
    bool m_inputTypeCamera = false;
    bool m_inputSingleImage = false;
    std::unique_ptr<SimpleCamera> m_stereoCamera[DW_STEREO_SIDE_BOTH];

    // sample helper class for rendering operations commonly used in the samples
    std::unique_ptr<dw_samples::common::SimpleRenderer> m_simpleRenderer;
};

#endif
