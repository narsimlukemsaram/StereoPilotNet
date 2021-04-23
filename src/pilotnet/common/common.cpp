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

#include "common.hpp"
#include <dw/object/DriveNet.h>

cudaStream_t g_cudaStream  = 0;

// Driveworks Handles
dwContextHandle_t gSdk                          = DW_NULL_HANDLE;
dwRendererHandle_t gRenderer                    = DW_NULL_HANDLE;
dwRenderBufferHandle_t gLineBuffer              = DW_NULL_HANDLE;
dwSALHandle_t gSal                              = DW_NULL_HANDLE;
dwSensorHandle_t gCameraSensor                  = DW_NULL_HANDLE;
dwSoftISPHandle_t gSoftISP              = DW_NULL_HANDLE;

// frame processing
dwImageCUDA gRCBImage{};
dwImageCUDA gRGBImage{};
dwImageCUDA gRGBAImage{};
dwImageProperties gRGBProperties{};

dwImageStreamerHandle_t gCuda2gl                = DW_NULL_HANDLE;
dwImageStreamerHandle_t gInput2cuda             = DW_NULL_HANDLE;
dwImageFormatConverterHandle_t gConvertRCB2RGB  = DW_NULL_HANDLE;
dwImageFormatConverterHandle_t gConvertYUV2RGB  = DW_NULL_HANDLE;
dwImageFormatConverterHandle_t gConvertRGB2RGBA = DW_NULL_HANDLE;

// camera output type
dwCameraOutputType gCameraOutputType            = DW_CAMERA_RAW_IMAGE;

// Sample variables
dwRect gScreenRectangle{};
std::string gInputType;

// Colors for rendering bounding boxes
const uint32_t gMaxBoxColors = DW_DRIVENET_NUM_CLASSES;
float32_t gBoxColors[gMaxBoxColors][4] = {{1.0f, 0.0f, 0.0f, 1.0f},
                                          {0.0f, 1.0f, 0.0f, 1.0f},
                                          {0.0f, 0.0f, 1.0f, 1.0f},
                                          {1.0f, 0.0f, 1.0f, 1.0f},
                                          {1.0f, 0.647f, 0.0f, 1.0f}};

//------------------------------------------------------------------------------
void drawROI(dwRect roi, const float32_t color[4], dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer)
{
    float32_t x_start = static_cast<float32_t>(roi.x) ;
    float32_t x_end   = static_cast<float32_t>(roi.x + roi.width);
    float32_t y_start = static_cast<float32_t>(roi.y);
    float32_t y_end   = static_cast<float32_t>(roi.y + roi.height);

    float32_t *coords     = nullptr;
    uint32_t maxVertices  = 0;
    uint32_t vertexStride = 0;
    dwRenderBuffer_map(&coords, &maxVertices, &vertexStride, renderBuffer);
    coords[0]  = x_start;
    coords[1]  = y_start;
    coords    += vertexStride;
    coords[0]  = x_start;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0]  = x_start;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0]  = x_end;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0]  = x_end;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0] = x_end;
    coords[1] = y_start;
    coords    += vertexStride;
    coords[0] = x_end;
    coords[1] = y_start;
    coords    += vertexStride;
    coords[0] = x_start;
    coords[1] = y_start;
    dwRenderBuffer_unmap(8, renderBuffer);
    dwRenderer_setColor(color, renderer);
    dwRenderer_setLineWidth(2, renderer);
    dwRenderer_renderBuffer(renderBuffer, renderer);
}

//------------------------------------------------------------------------------
bool initPipeline(const dwImageProperties &cameraImageProps, const dwCameraProperties &cameraProps,
                  dwDriveNetHandle_t driveNet, dwContextHandle_t ctx)
{
    dwStatus result = DW_SUCCESS;

    // raw image
    if (gCameraOutputType == DW_CAMERA_RAW_IMAGE) {

        // we need to set the softISP, Drivenet can setup the softISP with specific parameters
        dwSoftISPParams softISPParams;
        // camera properties define resolution and raw format, dnnMetaData define the tonemapping type
        dwSoftISP_initParamsFromCamera(&softISPParams, cameraProps);

        result = dwSoftISP_initialize(&gSoftISP, softISPParams, ctx);
        if (result != DW_SUCCESS) {
            std::cerr << "Image streamer initialization failed: " << dwGetStatusName(result) << std::endl;
            return false;
        }

        {
            // this is in case the network requires a different tonemapper than the default one
            dwDNNMetaData dnnMetaData;
            dwDriveNet_getDNNMetaData(&dnnMetaData, driveNet);
            dwSoftISP_setTonemapType(dnnMetaData.tonemapType, gSoftISP);
        }

        // Initialize Raw pipeline
        dwSoftISP_setCUDAStream(g_cudaStream, gSoftISP);

        dwImageProperties rcbImageProperties;
        dwSoftISP_getDemosaicImageProperties(&rcbImageProperties, gSoftISP);

        // RCB image to get output from the SoftISP
        dwImageCUDA_create(&gRCBImage, &rcbImageProperties, DW_IMAGE_CUDA_PITCH);
        dwSoftISP_bindDemosaicOutput(&gRCBImage, gSoftISP);

        // RGB image
        gRGBProperties  = rcbImageProperties;
        gRGBProperties.pxlFormat  = DW_IMAGE_RGB;
        gRGBProperties.pxlType    = DW_TYPE_FLOAT16;
        gRGBProperties.planeCount = 3;

        // Setup streamer to pass input to CUDA
        result = result != DW_SUCCESS ? result : dwImageStreamer_initialize(&gInput2cuda, &cameraImageProps, DW_IMAGE_CUDA, ctx);
    }
    // h264
    else {
        gRGBProperties = cameraImageProps;
        gRGBProperties.type = DW_IMAGE_CUDA;
        gRGBProperties.pxlFormat  = DW_IMAGE_RGB;
        gRGBProperties.pxlType    = DW_TYPE_FLOAT16;
        gRGBProperties.planeCount = 3;

        // init format converter to convert from YUV420->RGB
        result = result != DW_SUCCESS ? result : dwImageFormatConverter_initialize(&gConvertYUV2RGB, gRGBProperties.type, ctx);
        if (result != DW_SUCCESS) {
            std::cerr << "Image format converter initialization failed: " << dwGetStatusName(result) << std::endl;
            return false;
        }

#ifdef VIBRANTE
        // Setup streamer to pass input to CUDA
        result = dwImageStreamer_initialize(&gInput2cuda, &cameraImageProps, DW_IMAGE_CUDA, ctx);
        if (result != DW_SUCCESS) {
            std::cerr << "Image streamer initialization failed: " << dwGetStatusName(result) << std::endl;
            gRun = false;
        }
#endif
    }

    // create RGB image
    dwImageCUDA_create(&gRGBImage, &gRGBProperties, DW_IMAGE_CUDA_PITCH);
    dwSoftISP_bindTonemapOutput(&gRGBImage, gSoftISP);

    // create RGBA image to display over GL
    dwImageProperties rgbaImageProperties = gRGBProperties;
    rgbaImageProperties.pxlFormat         = DW_IMAGE_RGBA;
    rgbaImageProperties.pxlType           = DW_TYPE_UINT8;
    rgbaImageProperties.planeCount        = 1;
    dwImageCUDA_create(&gRGBAImage, &rgbaImageProperties, DW_IMAGE_CUDA_PITCH);

    // init format converter to convert from RGB->RGBA
    result = result != DW_SUCCESS ? result : dwImageFormatConverter_initialize(&gConvertRGB2RGBA, gRGBProperties.type, ctx);
    if (result != DW_SUCCESS) {
        std::cerr << "Image format converter initialization failed: " << dwGetStatusName(result) << std::endl;
        return false;
    }

    // Setup streamer to pass CUDA to GL
    result = result != DW_SUCCESS ? result : dwImageStreamer_initialize(&gCuda2gl, &rgbaImageProperties, DW_IMAGE_GL, ctx);
    if (result != DW_SUCCESS) {
        std::cerr << "Image streamer initialization failed: " << dwGetStatusName(result) << std::endl;
        return false;
    }

    return true;
}

//------------------------------------------------------------------------------
void initSdk(dwContextHandle_t *context, WindowBase *window)
{
    // create a Logger to log to console
    // we keep the ownership of the logger at the application level
    dwLogger_initialize(getConsoleLoggerCallback(true));
    dwLogger_setLogLevel(DW_LOG_VERBOSE);

    // instantiate Driveworks SDK context
    dwContextParameters sdkParams{};

    std::string path = DataPath::get();
    sdkParams.dataPath = path.c_str();

#ifdef VIBRANTE
    sdkParams.eglDisplay = window->getEGLDisplay();
#else
    (void)window;
#endif

    dwInitialize(context, DW_VERSION, &sdkParams);
}

//------------------------------------------------------------------------------
void initRenderer(const dwImageProperties& rcbProperties, dwRendererHandle_t *renderer, dwContextHandle_t context, WindowBase *window)
{
    dwStatus result = dwRenderer_initialize(renderer, context);
    if (result != DW_SUCCESS)
        throw std::runtime_error(std::string("Cannot init renderer: ") + dwGetStatusName(result));

    // Set some renderer defaults
    gScreenRectangle.width  = window->width();
    gScreenRectangle.height = window->height();
    gScreenRectangle.x      = 0;
    gScreenRectangle.y      = 0;

    float32_t rasterTransform[9];
    rasterTransform[0] = 1.0f;
    rasterTransform[3] = 0.0f;
    rasterTransform[6] = 0.0f;
    rasterTransform[1] = 0.0f;
    rasterTransform[4] = 1.0f;
    rasterTransform[7] = 0.0f;
    rasterTransform[2] = 0.0f;
    rasterTransform[5] = 0.0f;
    rasterTransform[8] = 1.0f;

    dwRenderer_set2DTransform(rasterTransform, *renderer);
    float32_t boxColor[4] = {0.0f,1.0f,0.0f,1.0f};
    dwRenderer_setColor(boxColor, *renderer);
    dwRenderer_setLineWidth(2.0f, *renderer);

    dwRenderer_setRect(gScreenRectangle, *renderer);

    uint32_t maxLines = 20000;
    {
        dwRenderBufferVertexLayout layout;
        layout.posFormat   = DW_RENDER_FORMAT_R32G32_FLOAT;
        layout.posSemantic = DW_RENDER_SEMANTIC_POS_XY;
        layout.colFormat   = DW_RENDER_FORMAT_NULL;
        layout.colSemantic = DW_RENDER_SEMANTIC_COL_NULL;
        layout.texFormat   = DW_RENDER_FORMAT_NULL;
        layout.texSemantic = DW_RENDER_SEMANTIC_TEX_NULL;
        dwRenderBuffer_initialize(&gLineBuffer, layout, DW_RENDER_PRIM_LINELIST, maxLines, context);
        dwRenderBuffer_set2DCoordNormalizationFactors((float32_t)rcbProperties.width,
                                                      (float32_t)rcbProperties.height, gLineBuffer);
    }
}

//------------------------------------------------------------------------------
bool initSensors(dwSALHandle_t *sal, dwSensorHandle_t *camera, dwImageProperties *cameraImageProperties,
                 dwCameraProperties* cameraProperties, dwContextHandle_t context)
{
    dwStatus result;

    result = dwSAL_initialize(sal, context);
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot initialize SAL: " << dwGetStatusName(result) << std::endl;
        return false;
    }

    // create GMSL Camera interface
    dwSensorParams params;
#ifdef VIBRANTE
    if (gInputType.compare("camera") == 0) {
        std::string parameterString = "camera-type=" + gArguments.get("camera-type");
        parameterString += ",csi-port=" + gArguments.get("csi-port");
        parameterString += ",slave=" + gArguments.get("slave");
        parameterString += ",serialize=false,output-format=raw,camera-count=4";
        std::string cameraMask[4] = {"0001", "0010", "0100", "1000"};
        uint32_t cameraIdx = std::stoi(gArguments.get("camera-index"));
        if(cameraIdx < 0 || cameraIdx > 3){
            std::cerr << "Error: camera index must be 0, 1, 2 or 3" << std::endl;
            return false;
        }
        parameterString += ",camera-mask=" + cameraMask[cameraIdx];

        params.parameters           = parameterString.c_str();
        params.protocol             = "camera.gmsl";

        result                      = dwSAL_createSensor(camera, params, *sal);
    }else
#endif
    {
        std::string parameterString = gArguments.parameterString();
        params.parameters           = parameterString.c_str();
        params.protocol             = "camera.virtual";
        result                      = dwSAL_createSensor(camera, params, *sal);

        std::string videoFormat = gArguments.get("video");
        std::size_t found = videoFormat.find_last_of(".");

        if (videoFormat.substr(found+1).compare("h264") == 0) {
            gCameraOutputType = DW_CAMERA_PROCESSED_IMAGE;
        }
    }
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot create driver: camera.virtual with params: " << params.parameters << std::endl
                  << "Error: " << dwGetStatusName(result) << std::endl;
        return false;
    }


    dwSensorCamera_getImageProperties(cameraImageProperties, gCameraOutputType, *camera);
    dwSensorCamera_getSensorProperties(cameraProperties, *camera);

#ifdef VIBRANTE
    if(gInputType.compare("camera") == 0){
        if(cameraProperties->rawFormat == DW_CAMERA_RAW_FORMAT_RCCB ||
           cameraProperties->rawFormat == DW_CAMERA_RAW_FORMAT_BCCR ||
           cameraProperties->rawFormat == DW_CAMERA_RAW_FORMAT_CRBC ||
           cameraProperties->rawFormat == DW_CAMERA_RAW_FORMAT_CBRC){

           std::cout << "Camera image with " << cameraProperties->resolution.x << "x"
                << cameraProperties->resolution.y << " at " << cameraProperties->framerate << " FPS" << std::endl;

           return true;
        }
        else{
            std::cerr << "Camera is not supported" << std::endl;

            return false;
        }
    }
#endif

    return true;
}

//------------------------------------------------------------------------------
void resizeWindowCallback(int width, int height) {
   gScreenRectangle.width = width;
   gScreenRectangle.height = height;
   gScreenRectangle.x = 0;
   gScreenRectangle.y = 0;
   dwRenderer_setRect(gScreenRectangle, gRenderer);
}



//------------------------------------------------------------------------------
bool getNextFrameImages(dwImageCUDA** rgbCudaImageOut, dwImageGL** rgbaGLImageOut, dwCameraFrameHandle_t frameHandle)
{
    dwStatus result = DW_SUCCESS;

    // raw image
    if (gCameraOutputType == DW_CAMERA_RAW_IMAGE) {

        const dwImageDataLines* dataLines;
        dwImageCPU  *rawImageCPU;
        dwImageCUDA *rawImageCUDA;

#ifdef VIBRANTE
        dwImageNvMedia *rawImageNvMedia = nullptr;

        if (gInputType.compare("camera") == 0) {
            result = dwSensorCamera_getImageNvMedia(&rawImageNvMedia, DW_CAMERA_RAW_IMAGE, frameHandle);
            rawImageNvMedia->prop.pxlFormat = DW_IMAGE_RAW;
        }else
#endif
        {
            result = dwSensorCamera_getImageCPU(&rawImageCPU, gCameraOutputType, frameHandle);
        }

        if (result != DW_SUCCESS) {
            std::cerr << "Cannot get raw image: " << dwGetStatusName(result) << std::endl;
            return false;
        }
        dwSensorCamera_getDataLines(&dataLines, frameHandle);

        // process
#ifdef VIBRANTE
        if (gInputType.compare("camera") == 0) {
            result = dwImageStreamer_postNvMedia(rawImageNvMedia, gInput2cuda);
        }else
#endif
        {
            result = dwImageStreamer_postCPU(rawImageCPU, gInput2cuda);
        }
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot post raw image: " << dwGetStatusName(result) << std::endl;
            return false;
        }

        // input image was posted, get now CUDA image out of the streamer
        dwImageStreamer_receiveCUDA(&rawImageCUDA, 10000, gInput2cuda);
        {
            // run softISP, get RCB and RGB images
            {
                dwSoftISP_bindRawInput(rawImageCUDA, gSoftISP);
                CHECK_DW_ERROR(dwSoftISP_processDeviceAsync(DW_SOFT_ISP_PROCESS_TYPE_DEMOSAIC | DW_SOFT_ISP_PROCESS_TYPE_TONEMAP,
                                                            gSoftISP));
            }

            // return used RAW image, we do not need it anymore, as we now have a copy through the RawPipeline
            dwImageStreamer_returnReceivedCUDA(rawImageCUDA, gInput2cuda);
        }

        // wait
#ifdef VIBRANTE
        if (gInputType.compare("camera") == 0) {
            dwImageStreamer_waitPostedNvMedia(&rawImageNvMedia, 10000, gInput2cuda);
        }else
#endif
        {
            dwImageStreamer_waitPostedCPU(&rawImageCPU, 10000, gInput2cuda);
        }

    }
    // h264 image
    else {
        dwImageCUDA *yuvImageCUDA = nullptr;

#ifdef VIBRANTE
        dwImageNvMedia *imageNvMedia = nullptr;
        result = dwSensorCamera_getImageNvMedia(&imageNvMedia, DW_CAMERA_PROCESSED_IMAGE, frameHandle);
        result = dwImageStreamer_postNvMedia(imageNvMedia, gInput2cuda);
        result = dwImageStreamer_receiveCUDA(&yuvImageCUDA, 60000, gInput2cuda);
#else
        result = dwSensorCamera_getImageCUDA(&yuvImageCUDA, DW_CAMERA_PROCESSED_IMAGE, frameHandle);
#endif
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot get h264 image: " << dwGetStatusName(result) << std::endl;
            return false;
        }
        result = dwImageFormatConverter_copyConvertCUDA(&gRGBImage, yuvImageCUDA, gConvertYUV2RGB, g_cudaStream);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot convert to rgb: " << dwGetStatusName(result) << std::endl;
            return false;
        }

#if VIBRANTE
        dwImageStreamer_returnReceivedCUDA(yuvImageCUDA, gInput2cuda);
        dwImageNvMedia *tempNvMedia;
        result = dwImageStreamer_waitPostedNvMedia(&tempNvMedia, 33000, gInput2cuda);
#endif
    }

    // RGB -> RGBA
    {
        dwImageFormatConverter_copyConvertCUDA(&gRGBAImage, &gRGBImage, gConvertRGB2RGBA, g_cudaStream);
    }

    // get GL image
    {
        dwImageStreamer_postCUDA(&gRGBAImage, gCuda2gl);
        dwImageStreamer_receiveGL(rgbaGLImageOut, 10000, gCuda2gl);
    }

    // RCB result
    *rgbCudaImageOut = &gRGBImage;

    return true;
}

//------------------------------------------------------------------------------
void returnNextFrameImages(dwImageCUDA* rgbCudaImageOut, dwImageGL* rgbaGLImage)
{
    (void)rgbCudaImageOut;

    // return GL image
    {
        dwImageStreamer_returnReceivedGL(rgbaGLImage, gCuda2gl);
        dwImageCUDA *returnedFrame;
        dwImageStreamer_waitPostedCUDA(&returnedFrame, 10000, gCuda2gl);
    }
}

//------------------------------------------------------------------------------
void release()
{
    if (gRGBAImage.dptr[0]) dwImageCUDA_destroy(&gRGBAImage);
    if (gRCBImage.dptr[0]) dwImageCUDA_destroy(&gRCBImage);
    if (gRGBImage.dptr[0]) dwImageCUDA_destroy(&gRGBImage);

    if (gConvertRGB2RGBA) {
        dwImageFormatConverter_release(&gConvertRGB2RGBA);
    }
    if (gConvertRCB2RGB) {
        dwImageFormatConverter_release(&gConvertRCB2RGB);
    }
    if (gConvertYUV2RGB) {
        dwImageFormatConverter_release(&gConvertYUV2RGB);
    }
    if (gCuda2gl) {
        dwImageStreamer_release(&gCuda2gl);
    }
    if (gInput2cuda) {
        dwImageStreamer_release(&gInput2cuda);
    }
    if (g_cudaStream) {
        cudaStreamDestroy(g_cudaStream);
    }
    if (gLineBuffer) {
        dwRenderBuffer_release(&gLineBuffer);
    }
    if (gRenderer) {
        dwRenderer_release(&gRenderer);
    }
    if (gCameraSensor) {
        dwSAL_releaseSensor(&gCameraSensor);
    }
    if (gSal) {
        dwSAL_release(&gSal);
    }
    if (gSoftISP) {
        dwSoftISP_release(&gSoftISP);
    }
    dwRelease(&gSdk);
    dwLogger_release();
}
