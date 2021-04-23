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

#include "stereoCommon.hpp"

//#######################################################################################
StereoApp::StereoApp(const ProgramArguments& args) : DriveWorksSample(args)
{
    m_imageProperties.type = DW_IMAGE_CUDA;
    m_imageProperties.planeCount = 1;
    m_imageProperties.pxlFormat = DW_IMAGE_RGBA;
    m_imageProperties.pxlType = DW_TYPE_UINT8;
    m_inputSingleImage = (args.get("single-input") == std::string("1"));

    for (int i = 0; i < DW_STEREO_SIDE_BOTH; ++i) {
        m_cameraModel[i] = DW_NULL_HANDLE;
    }
}

//#######################################################################################
bool StereoApp::onInitialize()
{
    if (m_inputTypeCamera) {
        // for now assume ZED camera (side-by-side images)
        m_stereoCamera[0] = initFromCamera();

        m_imageProperties.width = m_stereoCamera[0]->getImageProperties().width / 2;
        m_imageProperties.height = m_stereoCamera[0]->getImageProperties().height / 2;
    } else {

        m_stereoCamera[0] = initFromVideo(m_args.get("video0"));
        if (!m_inputSingleImage) {
            m_stereoCamera[1] = initFromVideo(m_args.get("video1"));
        }
    }

    if (!m_stereoCamera[0]) {
        return false;
    }

    return true;
}

//#######################################################################################
bool StereoApp::initSDK()
{
    // initialize logger to print verbose message on console in color
    CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
    CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));

    // initialize SDK context, using data folder
    dwContextParameters sdkParams = {};
    sdkParams.dataPath = DataPath::get_cstr();

#ifdef VIBRANTE
    sdkParams.eglDisplay = getEGLDisplay();
#endif

    CHECK_DW_ERROR(dwInitialize(&m_context, DW_VERSION, &sdkParams));
    CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_context));
    return true;
}

//#######################################################################################
bool StereoApp::initRenderer()
{
    CHECK_DW_ERROR_MSG(dwRenderer_initialize(&m_renderer, m_context),
                       "Cannot initialize Renderer, maybe no GL context available?");
    dwRect rect;
    rect.width  = getWindowWidth();
    rect.height = getWindowHeight();
    rect.x      = 0;
    rect.y      = 0;
    CHECK_DW_ERROR(dwRenderer_setRect(rect, m_renderer));

    m_simpleRenderer.reset(new SimpleRenderer(m_renderer, m_context));

    m_simpleRenderer->setRenderBufferNormCoords((float32_t)StereoApp::m_window->width(),
                                                (float32_t)StereoApp::m_window->height(),
                                                DW_RENDER_PRIM_LINELIST);
    return true;
}

//#######################################################################################
void StereoApp::onRelease()
{
    for(int i=0; i<DW_STEREO_SIDE_BOTH; ++i)
        m_stereoCamera[i].reset();

    m_simpleRenderer.reset();

    for (int i = 0; i < DW_STEREO_SIDE_BOTH; ++i) {
        if(m_cameraModel[i] != DW_NULL_HANDLE) {
            dwCalibratedCamera_release(&m_cameraModel[i]);
        }
    }
    if(m_rigConfiguration != DW_NULL_HANDLE) {
        dwRigConfiguration_release(&m_rigConfiguration);
    }

    dwRenderer_release(&m_renderer);
    dwSAL_release(&m_sal);
    dwRelease(&m_context);
}

//#######################################################################################
void StereoApp::setInputProperties(dwImageProperties imageProperties)
{
    m_imageProperties = imageProperties;
}

//#######################################################################################
bool StereoApp::getCameraFromRig()
{
    if (m_args.get("rigconfig").empty()) {
        throw std::runtime_error("Rig configuration file not specified, please provide a rig "
                                 "configuration file with the calibration of the stereo camera");
    }

    CHECK_DW_ERROR(dwRigConfiguration_initializeFromFile(&m_rigConfiguration, m_context,
                                                         m_args.get("rigconfig").c_str()));


    // check if 2 cameras are in there (only the stereo cameras are present for demonstrative purposes)
    uint32_t cameraCount = 0;
    CHECK_DW_ERROR(dwRigConfiguration_getSensorCount(&cameraCount, m_rigConfiguration));
    if (cameraCount != 2) {
        throw std::runtime_error("Wrong number of cameras in rig file.");
    }

    // get the camera models stored in the rig (contain the intrinsics)
    dwCameraRigHandle_t rig;
    CHECK_DW_ERROR(dwCameraRig_initializeFromConfig(&rig, &cameraCount, m_cameraModel, 2,
                                                    m_context, m_rigConfiguration));
    CHECK_DW_ERROR(dwCameraRig_release(&rig));

    uint32_t totalWidth = 0;
    for (int32_t i = 0; i < DW_STEREO_SIDE_BOTH; ++i) {
        dwPinholeCameraConfig pinholeConfig;
        CHECK_DW_ERROR(dwRigConfiguration_getPinholeCameraConfig(&pinholeConfig, i, m_rigConfiguration));

        if (pinholeConfig.height != m_imageProperties.height) {
            throw std::runtime_error("Camera height and height in rig.xml don't match.");
        }

        totalWidth += pinholeConfig.width;
    }

    if (totalWidth != (m_imageProperties.width * 2) ) {
        throw std::runtime_error("Camera width is invalid, are you using a stereo camera?");
    }

    return true;
}

//#######################################################################################
bool StereoApp::readStereoImages(dwImageCUDA *gStereoImages[DW_STEREO_SIDE_BOTH])
{
    bool endStatus = true;

    dwImageProperties cameraProperties = m_stereoCamera[0]->getImageProperties();

    if (m_inputTypeCamera) {
        // for now the sample illustrates only USB cameras with a single input
        dwImageCUDA *stereoFrame = GenericImage::toDW<dwImageCUDA>(m_stereoCamera[0]->readFrame());

        if (!stereoFrame) {
            m_stereoCamera[0]->resetCamera();
            return false;
        }

        dwRect roi[2];
        roi[0].height = cameraProperties.height;
        roi[0].width = cameraProperties.width / 2;
        roi[0].x = 0;
        roi[0].y = 0;
        roi[1].height = cameraProperties.height;
        roi[1].width = cameraProperties.width / 2;
        roi[1].x = cameraProperties.width / 2;
        roi[1].y = 0;

        // image is siplit in two by remapping to two separate dwImageCUDA
        dwImageCUDA_mapToROI(gStereoImages[0], stereoFrame, roi[0]);
        dwImageCUDA_mapToROI(gStereoImages[1], stereoFrame, roi[1]);

    } else {
        dwImageGeneric *stereoInput[2];
        for (int32_t i = 0; i < DW_STEREO_SIDE_BOTH; ++i){
            stereoInput[i] = m_stereoCamera[i]->readFrame();
            if (!stereoInput[i]) {
                endStatus &= false;
                m_stereoCamera[i]->resetCamera();
            }
        }

        if (endStatus) {
            for (int32_t i = 0; i < DW_STEREO_SIDE_BOTH; ++i){
                gStereoImages[i] = GenericImage::toDW<dwImageCUDA>(stereoInput[i]);
            }
        }
    }
    return true;
}

//#######################################################################################
std::unique_ptr<SimpleCamera> StereoApp::initFromVideo(const std::string &videoFName)
{
    std::string arguments = "video=" + videoFName;
    dwSensorParams params;
    params.parameters = arguments.c_str();
    params.protocol   = "camera.virtual";

    return initInput(params);

}

//#######################################################################################
std::unique_ptr<SimpleCamera> StereoApp::initFromCamera()
{
    dwSensorParams params;
    params.protocol = "camera.usb";
    std::string parameters = "device=" + m_args.get("device");
    params.parameters      = parameters.c_str();

    return initInput(params);
}

//#######################################################################################
std::unique_ptr<SimpleCamera> StereoApp::initInput(const dwSensorParams &params)
{
    std::unique_ptr<SimpleCamera> camera(new SimpleCamera(m_imageProperties, params, m_sal, m_context));

    dwCameraProperties cameraProperties = camera->getCameraProperties();
    dwImageProperties cameraImageProperties = camera->getImageProperties();
    m_imageProperties.width = cameraImageProperties.width;
    m_imageProperties.height = cameraImageProperties.height;

    {
        std::stringstream ss;
        ss << "Camera image with " << cameraImageProperties.width << "x" << cameraImageProperties.height
           << " at " << cameraProperties.framerate << " FPS" << std::endl;
        log("%s", ss.str().c_str());
    }

    return camera;
}
