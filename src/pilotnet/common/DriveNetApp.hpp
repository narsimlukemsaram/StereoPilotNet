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

#ifndef DRIVENET_COMNMON_H_
#define DRIVENET_COMNMON_H_

#include <vector>
#include <string>

// DriveNet
#include <dw/object/DriveNet.h>
#include <dw/object/Detector.h>
#include <dw/object/Tracker.h>

#include <framework/DriveWorksSample.hpp>
#include <framework/SimpleCamera.hpp>
#include <framework/SimpleStreamer.hpp>
#include <framework/SimpleFormatConverter.hpp>
#include <framework/SimpleRenderer.hpp>
#include <framework/Checks.hpp>

using namespace dw_samples::common;

class DriveNetApp : public DriveWorksSample
{
public:
    DriveNetApp(const ProgramArguments &args) : DriveWorksSample(args) {}

    // gets the next frame from the camera sensor
    void getNextFrame(dwImageCUDA **nextFrameCUDA, dwImageGL **nextFrameGL);
    const dwImageProperties getImageProperties() {return camera->getOutputProperties();}

    bool initSDK();
    bool initRenderer();
    bool initSensors();
    bool initTracker(const dwImageProperties& rcbProperties, cudaStream_t stream);
    bool initDetector(const dwImageProperties& rcbProperties, cudaStream_t stream);

    void resetDetector();
    void resetTracker();

    void inferDetectorAsync(const dwImageCUDA* rcbImage);
    void inferTrackerAsync(const dwImageCUDA* rcbImage);
    void processResults();

    size_t getNumClasses() { return classLabels.size(); }

    const std::vector<std::pair<dwBox2D,std::string>>& getResult(uint32_t classIdx);

    // Maximum number of proposals per class object class
    static const uint32_t maxProposalsPerClass = 1000U;
    // Maximum number of objects (clustered proposals) per object class
    static const uint32_t maxClustersPerClass = 400U;

protected:
    // ------------------------------------------------
    // Driveworks Context and SAL
    // ------------------------------------------------
    dwContextHandle_t context = DW_NULL_HANDLE;
    dwSALHandle_t sal = DW_NULL_HANDLE;
    dwRendererHandle_t renderer = DW_NULL_HANDLE;

    // DriveNet
    dwDriveNetHandle_t driveNet = DW_NULL_HANDLE;
    dwDriveNetParams driveNetParams{};
    const dwDriveNetClass *driveNetClasses = nullptr;
    uint32_t numDriveNetClasses = 0;
    // Detector
    dwObjectDetectorParams detectorParams{};
    dwObjectDetectorHandle_t driveNetDetector = DW_NULL_HANDLE;
    // Clustering
    dwObjectClusteringHandle_t *objectClusteringHandles = nullptr;
    // Tracker
    dwObjectTrackerHandle_t objectTracker = DW_NULL_HANDLE;

    cudaStream_t cudaStream = 0;

    // Colors for rendering bounding boxes
    static const uint32_t MAX_BOX_COLORS = DW_DRIVENET_NUM_CLASSES;
    float32_t boxColors[MAX_BOX_COLORS][4] = {{1.0f, 0.0f, 0.0f, 1.0f},
                                                {0.0f, 1.0f, 0.0f, 1.0f},
                                                {0.0f, 0.0f, 1.0f, 1.0f},
                                                {1.0f, 0.0f, 1.0f, 1.0f},
                                                {1.0f, 0.647f, 0.0f, 1.0f}};

    // Number of proposals per object class
    std::vector<size_t> numProposals;
    // List of proposals per object class
    std::vector<std::vector<dwObject>> objectProposals;

    // Number of clusters per object class
    std::vector<size_t> numClusters;
    // List of clusters per object class
    std::vector<std::vector<dwObject>> objectClusters;

    // Number of merged objects per object class
    std::vector<size_t> numMergedObjects;
    // List of merged objects per object class
    std::vector<std::vector<dwObject>> objectsMerged;

    // Number of tracked objects per object class
    std::vector<size_t> numTrackedObjects;
    // List of tracked objects per object class
    std::vector<std::vector<dwObject>> objectsTracked;

    // Labels of each class
    std::vector<std::string> classLabels;

    // Vector of pairs of boxes and class label ids
    std::vector<std::vector<std::pair<dwBox2D,std::string>>> dnnBoxList;

    std::unique_ptr<SimpleCamera> camera;
    std::unique_ptr<SimpleRenderer> simpleRenderer;

    // used for conversion to RGBA and streaming to GL for display
    std::unique_ptr<GenericSimpleFormatConverter> converterToRGBA;
    std::unique_ptr<SimpleImageStreamer<dwImageCUDA, dwImageGL>> streamerCUDA2GL;

};

#endif
