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
// Copyright (c) 2017 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <stdexcept>
#include <iostream>
#include <cmath>

#include "DriveNetApp.hpp"

//------------------------------------------------------------------------------
void DriveNetApp::getNextFrame(dwImageCUDA** nextFrameCUDA, dwImageGL** nextFrameGL)
{
    *nextFrameCUDA = GenericImage::toDW<dwImageCUDA>(camera->readFrame());
    if (*nextFrameCUDA == nullptr) {
        camera->resetCamera();
    } else {
        *nextFrameGL = streamerCUDA2GL->post(GenericImage::toDW<dwImageCUDA>(converterToRGBA->convert(GenericImage::fromDW(*nextFrameCUDA))));
    }
}

//------------------------------------------------------------------------------
bool DriveNetApp::initSDK()
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

    CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));
    CHECK_DW_ERROR(dwSAL_initialize(&sal, context));
    return true;
}

//------------------------------------------------------------------------------
bool DriveNetApp::initRenderer()
{
    // -----------------------------
    // Initialize Renderer
    // -----------------------------
    CHECK_DW_ERROR_MSG(dwRenderer_initialize(&renderer, context),
                       "Cannot initialize Renderer, maybe no GL context available?");
    dwRect rect;
    rect.width  = getWindowWidth();
    rect.height = getWindowHeight();
    rect.x      = 0;
    rect.y      = 0;
    CHECK_DW_ERROR(dwRenderer_setRect(rect, renderer));

    simpleRenderer.reset(new SimpleRenderer(renderer, context));

    // set the normalization coordinates for boxes rendering
    simpleRenderer->setRenderBufferNormCoords(getImageProperties().width,
                                              getImageProperties().height,
                                              DW_RENDER_PRIM_LINELIST);
    return true;
}

//------------------------------------------------------------------------------
bool DriveNetApp::initSensors()
{

    dwSensorParams params;
    {
#ifdef VIBRANTE
        if (m_args.get("input-type").compare("camera") == 0) {
            std::string parameterString = "camera-type=" + m_args.get("camera-type");
            parameterString += ",csi-port=" + m_args.get("csi-port");
            parameterString += ",slave=" + m_args.get("slave");
            parameterString += ",serialize=false,output-format=raw,camera-count=4";
            std::string cameraMask[4] = {"0001", "0010", "0100", "1000"};
            uint32_t cameraIdx = std::stoi(m_args.get("camera-index"));
            if(cameraIdx < 0 || cameraIdx > 3){
                std::cerr << "Error: camera index must be 0, 1, 2 or 3" << std::endl;
                return false;
            }
            parameterString += ",camera-mask=" + cameraMask[cameraIdx];

            params.parameters           = parameterString.c_str();
            params.protocol             = "camera.gmsl";

            camera.reset(new RawSimpleCamera(params, sal, context, cudaStream, DW_CAMERA_PROCESSED_IMAGE));
        }else
#endif
        {
            std::string parameterString = m_args.parameterString();
            params.parameters           = parameterString.c_str();
            params.protocol             = "camera.virtual";

            std::string videoFormat = m_args.get("video");
            std::size_t found = videoFormat.find_last_of(".");

            if (videoFormat.substr(found+1).compare("h264") == 0) {
                camera.reset(new SimpleCamera(params, sal, context));
                dwImageProperties outputProperties = camera->getOutputProperties();
                outputProperties.type = DW_IMAGE_CUDA;
                outputProperties.pxlFormat = DW_IMAGE_RGB;
                outputProperties.pxlType = DW_TYPE_FLOAT16;
                outputProperties.planeCount = 3;
                camera->setOutputProperties(outputProperties);
            } else {
                camera.reset(new RawSimpleCamera(params, sal, context, cudaStream, DW_CAMERA_PROCESSED_IMAGE));
            }
        }
    }


    if (camera == nullptr) {
        throw std::runtime_error("Camera could not be created");
    }

#ifdef VIBRANTE
    if(m_args.get("input-type").compare("camera") == 0){
        dwCameraRawFormat rawFormat = camera->getCameraProperties().rawFormat;
        if(rawFormat != DW_CAMERA_RAW_FORMAT_RCCB &&
                rawFormat != DW_CAMERA_RAW_FORMAT_BCCR &&
                rawFormat != DW_CAMERA_RAW_FORMAT_CRBC &&
                rawFormat != DW_CAMERA_RAW_FORMAT_CBRC){
            throw std::runtime_error("Camera not supported, see documentation");
        }
    }
#endif

    std::cout << "Camera image with " << camera->getCameraProperties().resolution.x << "x"
              << camera->getCameraProperties().resolution.y << " at "
              << camera->getCameraProperties().framerate << " FPS" << std::endl;

    dwImageProperties displayProperties = DriveNetApp::getImageProperties();
    displayProperties.pxlType = DW_TYPE_UINT8;
    displayProperties.planeCount = 1;
    displayProperties.pxlFormat = DW_IMAGE_RGBA;

    dwImageProperties tonemappedProps = DriveNetApp::getImageProperties();
    tonemappedProps.pxlFormat = DW_IMAGE_RGB;
    tonemappedProps.pxlType = DW_TYPE_UINT8;

    converterToRGBA.reset(new GenericSimpleFormatConverter(tonemappedProps,
                                                           displayProperties,
                                                           context));

    streamerCUDA2GL.reset(new SimpleImageStreamer<dwImageCUDA, dwImageGL>(displayProperties, 1000,
                                                                          context));

    return true;
}

//------------------------------------------------------------------------------
void DriveNetApp::resetTracker()
{
    CHECK_DW_ERROR(dwObjectTracker_reset(objectTracker));
}

//------------------------------------------------------------------------------
void DriveNetApp::resetDetector()
{
    for (uint32_t classIdx = 0U; classIdx < classLabels.size(); ++classIdx) {
        numClusters[classIdx] = 0U;
        numMergedObjects[classIdx] = 0U;
        numTrackedObjects[classIdx] = 0U;
    }
}

//------------------------------------------------------------------------------
bool DriveNetApp::initTracker(const dwImageProperties& rcbProperties, cudaStream_t stream)
{
    // initialize ObjectTracker - it will be required to track detected instances over multiple frames
    // for better understanding how ObjectTracker works see sample_object_tracker

    dwObjectFeatureTrackerParams featureTrackingParams;
    dwObjectTrackerParams objectTrackingParams[DW_OBJECT_MAX_CLASSES];
    CHECK_DW_ERROR(dwObjectTracker_initDefaultParams(&featureTrackingParams, objectTrackingParams,
                                                     numDriveNetClasses));
    featureTrackingParams.maxFeatureCount = 8000;
    featureTrackingParams.detectorScoreThreshold = 0.0001f;
    featureTrackingParams.iterationsLK = 10;
    featureTrackingParams.windowSizeLK = 8;

    for (uint32_t classIdx = 0U; classIdx < numDriveNetClasses; ++classIdx) {
        objectTrackingParams[classIdx].confRateTrackMax = 0.05f;
        objectTrackingParams[classIdx].confRateTrackMin = 0.01f;
        objectTrackingParams[classIdx].confRateDetect = 0.5f;
        objectTrackingParams[classIdx].confThreshDiscard = 0.0f;
        objectTrackingParams[classIdx].maxFeatureCountPerBox = 200;
    }

    {
        CHECK_DW_ERROR(dwObjectTracker_initialize(&objectTracker, context,
                                                  &rcbProperties, &featureTrackingParams,
                                                  objectTrackingParams, numDriveNetClasses));
    }

    CHECK_DW_ERROR(dwObjectTracker_setCUDAStream(stream, objectTracker));

    return true;
}

//------------------------------------------------------------------------------
bool DriveNetApp::initDetector(const dwImageProperties& rcbProperties, cudaStream_t cudaStream)
{
    // Initialize DriveNet network
    CHECK_DW_ERROR(dwDriveNet_initDefaultParams(&driveNetParams));
    // Set up max number of proposals and clusters
    driveNetParams.maxClustersPerClass = maxClustersPerClass;
    driveNetParams.maxProposalsPerClass = maxProposalsPerClass;
    // Set batch size to 2 for foveal.
    driveNetParams.networkBatchSize = DW_DRIVENET_BATCHSIZE_2;
    driveNetParams.networkPrecision = DW_DRIVENET_PRECISION_FP32;
    CHECK_DW_ERROR(dwDriveNet_initialize(&driveNet, &objectClusteringHandles, &driveNetClasses,
                                         &numDriveNetClasses, context, &driveNetParams));

    // Initialize Objec Detector from DriveNet
    dwObjectDetectorDNNParams tmpDNNParams;
    CHECK_DW_ERROR(dwObjectDetector_initDefaultParams(&tmpDNNParams, &detectorParams));
    // Enable fusing objects from different ROIs
    detectorParams.enableFuseObjects = DW_TRUE;
    // Two images will be given as input. Each image is a region on the image received from camera.
    detectorParams.maxNumImages = 2U;
    CHECK_DW_ERROR(dwObjectDetector_initializeFromDriveNet(&driveNetDetector, context, driveNet,
                                                           &detectorParams));
    CHECK_DW_ERROR(dwObjectDetector_setCUDAStream(cudaStream, driveNetDetector));

    // since our input images might have a different aspect ratio as the input to drivenet
    // we setup the ROI such that the crop happens from the top of the image
    float32_t aspectRatio = 1.0f;
    {
        dwBlobSize inputBlob;
        CHECK_DW_ERROR(dwDriveNet_getInputBlobsize(&inputBlob, driveNet));

        aspectRatio = static_cast<float32_t>(inputBlob.height) / static_cast<float32_t>(inputBlob.width);
    }

    // 1st image is a full resolution image as it comes out from the RawPipeline (cropped to DriveNet aspect ratio)
    dwRect fullROI;
    {
        fullROI = {0, 0, static_cast<int32_t>(rcbProperties.width),
                         static_cast<int32_t>(rcbProperties.width * aspectRatio)};
        dwTransformation2D transformation = {{1.0f, 0.0f, 0.0f,
                                              0.0f, 1.0f, 0.0f,
                                              0.0f, 0.0f, 1.0f}};

        CHECK_DW_ERROR(dwObjectDetector_setROI(0, &fullROI, &transformation, driveNetDetector));
    }

    // 2nd image is a cropped out region within the 1/4-3/4 of the original image in the center
    {
        dwRect ROI = {fullROI.width/4, fullROI.height/4,
                      fullROI.width/2, fullROI.height/2};
        dwTransformation2D transformation = {{1.0f, 0.0f, 0.0f,
                                              0.0f, 1.0f, 0.0f,
                                              0.0f, 0.0f, 1.0f}};

        CHECK_DW_ERROR(dwObjectDetector_setROI(1, &ROI, &transformation, driveNetDetector));
    }

    // fill out member structure according to the ROIs
    CHECK_DW_ERROR(dwObjectDetector_getROI(&detectorParams.ROIs[0],
                   &detectorParams.transformations[0], 0, driveNetDetector));
    CHECK_DW_ERROR(dwObjectDetector_getROI(&detectorParams.ROIs[1],
                   &detectorParams.transformations[1], 1, driveNetDetector));

    // Get which label name for each class id
    classLabels.resize(numDriveNetClasses);
    for (uint32_t classIdx = 0U; classIdx < numDriveNetClasses; ++classIdx) {
        const char *classLabel;
        CHECK_DW_ERROR(dwDriveNet_getClassLabel(&classLabel, classIdx, driveNet));
        classLabels[classIdx] = classLabel;
    }

    // Initialize arrays for the pipeline
    objectProposals.resize(numDriveNetClasses, std::vector<dwObject>(maxProposalsPerClass));
    objectClusters.resize(numDriveNetClasses, std::vector<dwObject>(maxClustersPerClass));
    objectsTracked.resize(numDriveNetClasses, std::vector<dwObject>(maxClustersPerClass));
    objectsMerged.resize(numDriveNetClasses, std::vector<dwObject>(maxClustersPerClass));

    numTrackedObjects.resize(numDriveNetClasses, 0);
    numMergedObjects.resize(numDriveNetClasses, 0);
    numClusters.resize(numDriveNetClasses, 0);
    numProposals.resize(numDriveNetClasses, 0);

    dnnBoxList.resize(numDriveNetClasses);

    return true;
}

//------------------------------------------------------------------------------
const std::vector<std::pair<dwBox2D,std::string>>& DriveNetApp::getResult(uint32_t classIdx)
{
    return dnnBoxList[classIdx];
}

//------------------------------------------------------------------------------
void DriveNetApp::inferDetectorAsync(const dwImageCUDA* rcbImage)
{
    // we feed two images to the DriveNet module, the first one will have full ROI
    // the second one, is the same image, however with an ROI cropped in the center
    const dwImageCUDA* rcbImagePtr[2] = {rcbImage, rcbImage};
    CHECK_DW_ERROR(dwObjectDetector_inferDeviceAsync(rcbImagePtr, 2U, driveNetDetector));
}

//------------------------------------------------------------------------------
void DriveNetApp::inferTrackerAsync(const dwImageCUDA* rcbImage)
{
    // track feature points on the rcb image
    CHECK_DW_ERROR(dwObjectTracker_featureTrackDeviceAsync(rcbImage, objectTracker));
}

//------------------------------------------------------------------------------
void DriveNetApp::processResults()
{
    CHECK_DW_ERROR(dwObjectDetector_interpretHost(2U, driveNetDetector));

    // for each detection class, we do
    for (uint32_t classIdx = 0U; classIdx < classLabels.size(); ++classIdx) {

        // track detection from last frame given new feature tracker responses
        CHECK_DW_ERROR(dwObjectTracker_boxTrackHost(objectsTracked[classIdx].data(), &numTrackedObjects[classIdx],
                                                    objectsMerged[classIdx].data(), numMergedObjects[classIdx],
                                                    classIdx, objectTracker));

        // extract new detections from DriveNet
        CHECK_DW_ERROR(dwObjectDetector_getDetectedObjects(objectProposals[classIdx].data(),
                                                           &numProposals[classIdx],
                                                           0U, classIdx, driveNetDetector));

        // cluster proposals
        CHECK_DW_ERROR(dwObjectClustering_cluster(objectClusters[classIdx].data(),
                                                  &numClusters[classIdx],
                                                  objectProposals[classIdx].data(),
                                                  numProposals[classIdx],
                                                  objectClusteringHandles[classIdx]));


        // the new response should be at new location as detected by DriveNet
        // in addition we have previously tracked response from last time
        // we hence now merge both detections to find the actual response for the current frame
        const dwObject *toBeMerged[2] = {objectsTracked[classIdx].data(),
                                         objectClusters[classIdx].data()};
        const size_t sizes[2] = {numTrackedObjects[classIdx], numClusters[classIdx]};
        CHECK_DW_ERROR(dwObject_merge(objectsMerged[classIdx].data(), &numMergedObjects[classIdx],
                       maxClustersPerClass, toBeMerged, sizes, 2U, 0.1f, 0.1f, context));

        // extract now the actual bounding box of merged response in pixel coordinates to render on screen
        dnnBoxList[classIdx].resize(numMergedObjects[classIdx]);

        for (uint32_t objIdx = 0U; objIdx < numMergedObjects[classIdx]; ++objIdx) {
            const dwObject &obj = objectsMerged[classIdx][objIdx];
            dwBox2D &box = dnnBoxList[classIdx][objIdx].first;
            box.x = static_cast<int32_t>(std::round(obj.box.x));
            box.y = static_cast<int32_t>(std::round(obj.box.y));
            box.width = static_cast<int32_t>(std::round(obj.box.width));
            box.height = static_cast<int32_t>(std::round(obj.box.height));

            dnnBoxList[classIdx][objIdx].second = classLabels[classIdx];
        }
    }
}
