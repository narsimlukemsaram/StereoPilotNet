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
////////////////////////////////////tereo/////////////////////////////////////////////////////

// SAMPLES COMMON
#include <framework/Grid.hpp>

// DRIVENET COMMON
#include <pilotnet/common/DriveNetApp.hpp>
#include <pilotnet/common/common.hpp>
#include <stereo/stereo_common/stereoCommon.hpp>

// RAD2DEG
#include <framework/MathUtils.hpp>

#include <bitset>

#define MAX_CAMERAS 2

//------------------------------------------------------------------------------
// Variables PilotNet
//------------------------------------------------------------------------------

struct Camera {
	dwSensorHandle_t sensor;
	uint32_t numSiblings;
};
std::vector<Camera> gCameras;
uint32_t gNumCameras = 0;

GridData_t gGrid;
uint32_t gImageWidth = 0;
uint32_t gImageHeight = 0;
uint32_t gFrameInference = 0;

// Setup CUDA->GL streamer for producing images to render on screen
dwImageStreamerHandle_t cuda2gl;

//------------------------------------------------------------------------------
// Variables StereoNet
//------------------------------------------------------------------------------

// handle to the stereo algorithm
dwStereoHandle_t m_stereoAlgorithm;

// handle to pyramids, input to the stereo algorithm
dwPyramidHandle_t m_pyramids[DW_STEREO_SIDE_COUNT];

// handle to stereo rectifier
dwStereoRectifierHandle_t m_stereoRectifier;

// converts RGBA to R to pass to Stereo algorithm
dwImageFormatConverterHandle_t m_RGBA2R;

// output of the stereo rectifier
dwImageCUDA m_outputRectified[DW_STEREO_SIDE_COUNT];

dwCalibratedCameraHandle_t m_cameraModel[DW_STEREO_SIDE_COUNT];
dwRigConfigurationHandle_t m_rigConfiguration = DW_NULL_HANDLE;
dwCalibratedCameraHandle_t m_calibratedCamera;

/*
    images that are undistorted following rectification (due to intrinsics prior distortion and
    additional distortion after stereo rectification) are not shaped like rectangles. This roi contains
    a rectangular are of valid pixels common to both cameras. The roi can end up being very narrow if
    the two cameras have either heavy distortion or very large baseline
 */
dwBox2D m_roi;

// anaglyph (left and right overlapped)
dwImageCUDA m_outputAnaglyph;

// container for color coded disparity map
dwImageCUDA m_colorDisparity;
const dwImageCUDA *disparity;
dwImageCUDA objDisparity;
dwImageProperties props;
const dwImageCUDA *confidence;
dwImageCUDA m_outputRectifiedR;

// ...and we allocate for a gaussian pyramid that will be built with the rectified images
// the number of levels is adjusted to the base resolution of a typical stereo camera
// the default input has a very high resolution so many levels guarantee a better coverage of the image
uint32_t m_levelCount = 6;
uint32_t m_levelStop;
dwStereoSide m_side;
float32_t m_invalidThreshold = -1.0f;
dwBool m_occlusion = DW_TRUE;
dwBool m_occlusionInfill = DW_FALSE;
dwBool m_infill = DW_FALSE;

// for color code rendering
float32_t m_colorGain;
const float32_t COLOR_GAIN = 10.0f; // default 4
const float32_t COLOR_GAIN_MAX = 30.0f; // default 20
const float32_t INV_THR_MAX = 10.0f; // default 6

int minDisp;
int maxDisp;

float32_t depth[100];
float32_t alphaAngle[100];
float32_t betaAngle[100];
uint16_t speed[100];

// setup rendering rectangle for the current camera
dwRect rect{};

// map CUDA RGBA frame to GL space
dwImageGL *frameGL;

dwStatus result = DW_SUCCESS;

// DriveNet detector
dwDriveNetHandle_t m_driveNet;
dwDriveNetParams m_driveNetParams{};
const dwDriveNetClass *driveNetClasses = nullptr;
uint32_t numDriveNetClasses = 0;

// Detector
dwObjectDetectorParams detectorParams{};
dwObjectDetectorParams drivenetParams {};
dwObjectDetectorHandle_t m_driveNetDetector;
// dwDriveNetParams drivenetParams;

// Clustering
dwObjectClusteringHandle_t *objectClusteringHandles = nullptr;

// Tracker
dwObjectTrackerHandle_t m_objectTracker;

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

// Maximum number of proposals per class object class
static const uint32_t maxProposalsPerClass = 1000U;

// Maximum number of objects (clustered proposals) per object class
static const uint32_t maxClustersPerClass = 400U;

// Screen geometry
dwRect m_screenRectangle;
uint32_t m_windowWidth;
uint32_t m_windowHeight;

// Lane detector
dwLaneDetectorHandle_t m_laneDetector;

// Freespace detector
dwFreeSpaceDetectorHandle_t m_freeSpaceDetector;

float32_t m_threshold = 0.3f;
float32_t m_maxDistance = 50.0f;

dwTime_t tNow;
dwTime_t tSend;
dwTime_t lastTime = 0;

uint8_t noOfObjects = 0;
uint8_t sequenceNumber = 0;

// processing time of frameworks
bool driveNetDetector = true;
bool laneDetector = true;
bool freeSpaceDetector = true;

//------------------------------------------------------------------------------
// Method declarations
//------------------------------------------------------------------------------
int main(int argc, const char **argv);
bool initSensors(dwSALHandle_t *sal, std::vector<Camera> *cameras, dwImageProperties *cameraImageProperties,
		dwCameraProperties* cameraProperties, dwContextHandle_t ctx);
bool initNCameras(std::vector<Camera> *cameras, dwSALHandle_t sal);
bool initNVideos(std::vector<Camera> *cameras, dwSALHandle_t sal);
void runPipeline(dwSoftISPHandle_t softISP, const dwImageProperties& rawImageProps,
		const dwImageProperties& rcbImageProps, float32_t framerate, dwContextHandle_t ctx, cudaStream_t m_cudaStream);
void resizeWindowCallbackGrid(int width, int height);
void releaseGrid();

// PilotNet
bool initDriveworks(dwContextHandle_t ctx);
bool initDetector(const dwImageProperties& rcbImageProps, dwContextHandle_t ctx, cudaStream_t m_stream);
bool initTracker(const dwImageProperties& rcbImageProps, dwContextHandle_t ctx, cudaStream_t m_stream);
bool initRenderer(const dwImageProperties& rcbImageProps, dwContextHandle_t ctx);
void inferDetectorAsync(const dwImageCUDA* rcbImage);
void inferTrackerAsync(const dwImageCUDA* rcbImage);
void processResults(dwContextHandle_t ctx);
void drawROIs(dwRect roi, const float32_t color[4], dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer);
void drawLaneMarkings(const dwLaneDetection &lanes, float32_t laneWidth, dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer);
void drawLaneMarkingsCustomColor(float32_t laneColors[][4], uint32_t nColors,
		const dwLaneDetection &lanes, float32_t laneWidth,
		dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer);
void drawLaneDetectionROI(dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer);
void drawFreeSpaceBoundary(dwFreeSpaceDetection* boundary, dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer);
void drawFreeSpaceDetectionROI(dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer);
void setupRenderer(dwRendererHandle_t &renderer, const dwRect &screenRect, dwContextHandle_t ctx);
void setupLineBuffer(const dwImageProperties& rcbImageProps, dwRenderBufferHandle_t &lineBuffer, unsigned int maxLines, dwContextHandle_t ctx);
size_t getNumClasses() { return classLabels.size(); }
const std::vector<std::pair<dwBox2D,std::string>>& getResult(uint32_t classIdx);
void renderCameraTexture(dwImageStreamerHandle_t streamer, dwRendererHandle_t renderer);
void keyPress(int key);
void resetTracker();
void resetDetector();

//------------------------------------------------------------------------------
// Method declarations for StereoNet
//------------------------------------------------------------------------------

// Create calibrated cameras based on the stereo rig (assumed the stereo cameras are the first)
bool getRCBCameraFromRig(dwImageProperties& rcbImageProps, dwContextHandle_t ctx);
// renders lines that show that rectified images have pixels that lie on the same horizontal line
// void renderHorizontalLines();

//------------------------------------------------------------------------------
// Method implementations
//------------------------------------------------------------------------------
int main(int argc, const char **argv)
{
	// Program arguments
	std::string videosString;

	// DriveNet
	// videosString = DataPath::get() + "/samples/raw/rccb.raw";
	// videosString += "," + DataPath::get() + "/samples/raw/rccb.raw";

        // bicycle
	// videosString = DataPath::get() + "/samples/stereo/twizy/dw_20191002_132942_0.000000_0.000000/stereo_video_first.raw";
	// videosString += "," + DataPath::get() + "/samples/stereo/twizy/dw_20191002_132942_0.000000_0.000000/stereo_video_second.raw";

        // car
	// videosString = DataPath::get() + "/samples/stereo/twizy/dw_20191002_132610_0.000000_0.000000/stereo_video_first.raw";
	// videosString += "," + DataPath::get() + "/samples/stereo/twizy/dw_20191002_132610_0.000000_0.000000/stereo_video_second.raw";

        // person
	// videosString = DataPath::get() + "/samples/stereo/twizy/dw_20191002_132423_0.000000_0.000000/stereo_video_first.raw";
	// videosString += "," + DataPath::get() + "/samples/stereo/twizy/dw_20191002_132423_0.000000_0.000000/stereo_video_second.raw";

        // Traffic lights at TU/e entrance
	videosString = DataPath::get() + "/samples/rawTUe/stereoTwizy/dw_20190919_114231_0.000000_0.000000/stereo_video_first.raw";
	videosString += "," + DataPath::get() + "/samples/rawTUe/stereoTwizy/dw_20190919_114231_0.000000_0.000000/stereo_video_second.raw";

        // person with bicycle
	// videosString = DataPath::get() + "/samples/rawTUe/stereoTwizy/dw_20190403_163308_0.000000_0.000000/stereo_video_first.raw";
	// videosString += "," + DataPath::get() + "/samples/rawTUe/stereoTwizy/dw_20190403_163308_0.000000_0.000000/stereo_video_second.raw";


	ProgramArguments arguments({
#ifdef VIBRANTE
		ProgramArguments::Option_t("camera-type", "ar0231-rccb-ssc"),
				ProgramArguments::Option_t("slave", "0"),
				ProgramArguments::Option_t("input-type", "camera"),
				ProgramArguments::Option_t("fifo-size", "5"),
				ProgramArguments::Option_t("selector-mask", "000100000001"), // ab cd ef
#else
				ProgramArguments::Option_t("videos", videosString.c_str()),
#endif

				ProgramArguments::Option_t("stopFrame", "0"),
				ProgramArguments::Option_t("skipFrame", "0"), // 0 skip no frames, 1 skip 1 frame every 2, and so on
				ProgramArguments::Option_t("skipInference", "0"),

				// StereoNet
				// ProgramArguments::Option_t("rigconfig", (DataPath::get() + "/samples/stereo/full.xml").c_str(), "Rig configuration file."),
				// ProgramArguments::Option_t("video0", (DataPath::get() + std::string{"/samples/stereo/left_1.h264"}).c_str(), "Left input video."),
				// ProgramArguments::Option_t("video1", (DataPath::get() + std::string{"/samples/stereo/right_1.h264"}).c_str(), "Right input video."),

				// StereoPilotNet
				// ProgramArguments::Option_t("rigconfig", (DataPath::get() + "/samples/stereo/tue-stereo-twizy.xml").c_str(), "Rig configuration file."),
				ProgramArguments::Option_t("rigconfig", (DataPath::get() + "/samples/stereo/tue-stereo-twizy-1-2.xml").c_str(), "Rig configuration file."),
				// ProgramArguments::Option_t("video0", (DataPath::get() + std::string{"/samples/stereo/left-stereo-tue.h264"}).c_str(), "Left input video."),
				// ProgramArguments::Option_t("video1", (DataPath::get() + std::string{"/samples/stereo/right-stereo-tue.h264"}).c_str(), "Right input video."),

				// lane detection
				ProgramArguments::Option_t("threshold", "0.3"),
				ProgramArguments::Option_t("maxDistance", "50.0"),

				ProgramArguments::Option_t("fov", "60"),
				ProgramArguments::Option_t("fovX", "60.6"),
				ProgramArguments::Option_t("fovY", "36.1"),

				ProgramArguments::Option_t("level","0","Log level = 0, 1, 2, 3, pyramid level to display the disparity, depends on the number of levels (default 4)"),
				ProgramArguments::Option_t("single-input", "0", "If set to 0 both left and right inputs are provided. If set to 1 only one input is provided."),
				ProgramArguments::Option_t("single_side", "1", "If set to 1 it computes only the left image and approximates occlusions by thresholding the confidence map. If set to 0, it computes left and right stereo images and performs stereo pipeline."),
				ProgramArguments::Option_t("device", "0", "Selects whether to use iGPU (1) or dGPU(0)")
		        /*
		       	The stereo output is color coded for clarity and some pixels are masked if they
		        are occluded or invalid. It is possible to use keyboard input to change parameters:
                    0-6: changes the level of refinement (0 no refinement)
		            Q,W: changes the gain to the color code for visualizaion
		            O  : toggles occlusions
		            K  : infills occlusions (only if on)
		            +,-: changes invalidy threshold (appears as white pixels)
		            I  : toggle horizontal infill of invalidity pixels
		        */
	});

	// init framework
	initSampleApp(argc, argv, &arguments, NULL, 1280, 800);

	// gWindow->makeCurrent();
	gWindow->setOnKeypressCallback(keyPress);

	// set window resize callback
	gWindow->setOnResizeWindowCallback(resizeWindowCallbackGrid);

	// init driveworks
	initSdk(&gSdk, gWindow);

#ifdef VIBRANTE
	gInputType = gArguments.get("input-type");
#else
	gInputType = "video";
#endif

	// create HAL and camera
	dwImageProperties rawImageProps;
	dwImageProperties rcbImageProps;
	dwCameraProperties cameraProps;

	bool sensorsInitialized = initSensors(&gSal, &gCameras, &rawImageProps, &cameraProps, gSdk);

	if (sensorsInitialized) {

		gImageWidth = rawImageProps.width;
		gImageHeight = rawImageProps.height;

		// std::cout << "main()=>Before demosaic ==> rawImageProps image with " << rawImageProps.width << "x"
		//           << rawImageProps.height << std::endl;

		// Configure grid for N cameras rendering
		// configureGrid(&gGrid, gWindow->width(), gWindow->height(), gImageWidth, gImageHeight, gNumCameras);
		configureGrid(&gGrid, gWindow->width(), gWindow->height(), gImageWidth, gImageHeight, 4);

		// create 1 RCCB pipeline to handle all frames
		dwSoftISPHandle_t rawPipeline;
		dwSoftISPParams softISPParams;
		dwSoftISP_initParamsFromCamera(&softISPParams, cameraProps);
		dwSoftISP_initialize(&rawPipeline, softISPParams, gSdk);
		dwSoftISP_setCUDAStream(g_cudaStream, rawPipeline);
		dwSoftISP_setDemosaicMethod(DW_SOFT_ISP_DEMOSAIC_METHOD_INTERPOLATION, rawPipeline); // added
		dwSoftISP_getDemosaicImageProperties(&rcbImageProps, rawPipeline);
		// dwSoftISP_setDenoiseMethod(DW_SOFT_ISP_DENOISE_METHOD_NONE, rawPipeline); // added
		// std::cout << "main()=>After demosaic ==> rcbImageProps image with " << rcbImageProps.width << "x"
		//          << rcbImageProps.height << std::endl;

		// std::cout << "main()=>After demosaic ==> rawImageProps image with " << rawImageProps.width << "x"
		//           << rawImageProps.height << std::endl;

		// PilotNet
		if ((initDetector(rcbImageProps, gSdk, g_cudaStream)) && (initTracker(rcbImageProps, gSdk, g_cudaStream))) {
			// if (!initRenderer(rcbImageProps, gSdk)) return false;
			initRenderer(rcbImageProps, &gRenderer, gSdk, gWindow);
			runPipeline(rawPipeline, rawImageProps, rcbImageProps, cameraProps.framerate, gSdk, g_cudaStream);
		}

		dwSoftISP_release(&rawPipeline);
	}

	// release DW modules
	releaseGrid();
	release();

	// release framework
	releaseSampleApp();

	std::cout << "Happy autonomous driving!" <<std::endl;
	std::cout << "Mobile Perception Systems Research Lab, TU/e, Eindhoven, Netherlands!" <<std::endl;

	return 0;
}

//------------------------------------------------------------------------------
bool initSensors(dwSALHandle_t *sal, std::vector<Camera> *cameras, dwImageProperties *cameraImageProperties,
		dwCameraProperties* cameraProperties, dwContextHandle_t ctx)
{
	result = dwSAL_initialize(sal, ctx);
	if (result != DW_SUCCESS) {
		std::cerr << "Cannot initialize SAL: " << dwGetStatusName(result) << std::endl;
		return false;
	}

	// create GMSL Camera interface
	bool ret = false;
	if (gInputType.compare("camera") == 0) {
		ret = initNCameras(cameras, *sal);
	}
	else{
		ret = initNVideos(cameras, *sal);
	}
	if(!ret)
		return ret;

	for(uint32_t i = 0; i < cameras->size(); ++i){
		dwSensorCamera_getImageProperties(cameraImageProperties, DW_CAMERA_RAW_IMAGE, (*cameras)[i].sensor);
		dwSensorCamera_getSensorProperties(cameraProperties, (*cameras)[i].sensor);

		if(gInputType.compare("camera") == 0){
			if(cameraProperties->rawFormat == DW_CAMERA_RAW_FORMAT_RCCB ||
					cameraProperties->rawFormat == DW_CAMERA_RAW_FORMAT_BCCR ||
					cameraProperties->rawFormat == DW_CAMERA_RAW_FORMAT_CRBC ||
					cameraProperties->rawFormat == DW_CAMERA_RAW_FORMAT_CBRC){

				std::cout << "Camera image with " << cameraProperties->resolution.x << "x"
						<< cameraProperties->resolution.y << " at " << cameraProperties->framerate << " FPS" << std::endl;
			}
			else{
				std::cerr << "Camera is not supported" << std::endl;

				return false;
			}
		}
		else
			std::cout << "initSensors()=>Camera image with " << cameraProperties->resolution.x << "x"
			<< cameraProperties->resolution.y << " at " << cameraProperties->framerate << " FPS" << std::endl;
	}

	return true;
}

//------------------------------------------------------------------------------
bool initNCameras(std::vector<Camera> *cameras, dwSALHandle_t sal)
{
	std::string selector = gArguments.get("selector-mask");

	// identify active ports
	int idx             = 0;
	int cnt[3]          = {0, 0, 0};
	std::string port[3] = {"ab", "cd", "ef"};
	// std::string port[3] = {"cd", "ab", "ef"};
	for (size_t i = 0; i < selector.length() && i < 12; i++, idx++) {
		const char s = selector[i];
		if (s == '1') {
			cnt[idx / 4]++;
		}
	}

	// how many cameras selected in a port
	for (size_t p = 0; p < 3; p++) {
		if (cnt[p] > 0) {
			std::string params;

			params += std::string("csi-port=") + port[p];
			params += ",camera-type=" + gArguments.get("camera-type");
			// when using the mask, just ask for all cameras, mask will select properly
			params += ",serialize=false,output-format=raw,camera-count=4";

			if (selector.size() >= p*4) {
				params += ",camera-mask="+ selector.substr(p*4, std::min(selector.size() - p*4, size_t{4}));
			}

			params += ",slave="  + gArguments.get("slave");

			params += ",fifo-size="  + gArguments.get("fifo-size");

			dwSensorHandle_t salSensor = DW_NULL_HANDLE;
			dwSensorParams salParams;
			salParams.parameters = params.c_str();
			salParams.protocol = "camera.gmsl";
			result = dwSAL_createSensor(&salSensor, salParams, sal);
			if (result == DW_SUCCESS) {
				Camera cam;
				cam.sensor = salSensor;

				dwImageProperties cameraImageProperties;
				dwSensorCamera_getImageProperties(&cameraImageProperties,
						DW_CAMERA_PROCESSED_IMAGE,
						salSensor);

				dwCameraProperties cameraProperties;
				dwSensorCamera_getSensorProperties(&cameraProperties, salSensor);
				cam.numSiblings = cameraProperties.siblings;

				cameras->push_back(cam);

				gNumCameras += cam.numSiblings;
			}
			else
			{
				std::cerr << "Cannot create driver: " << salParams.protocol
						<< " with params: " << salParams.parameters << std::endl
						<< "Error: " << dwGetStatusName(result) << std::endl;
						if (result == DW_INVALID_ARGUMENT) {
							std::cerr << "It is possible the given camera is not supported. "
									<< "Please refer to the documentation for this sample."
									<< std::endl;
						}
						return false;
			}
		}
	}
	return true;
}

//------------------------------------------------------------------------------
bool initNVideos(std::vector<Camera> *cameras, dwSALHandle_t sal)
{
	std::string videos = gArguments.get("videos");
	int idx = 0;
	// static int count = 0;

	while(true){
		size_t found = videos.find(",", idx);

		Camera cam;
		dwSensorHandle_t salSensor = DW_NULL_HANDLE;
		dwSensorParams params;

		std::string parameterString = "video=" + videos.substr(idx, found - idx);
		params.parameters           = parameterString.c_str();
		params.protocol             = "camera.virtual";
		result                      = dwSAL_createSensor(&salSensor, params, sal);
		if (result != DW_SUCCESS) {
			std::cerr << "Cannot create driver: camera.virtual with params: " << params.parameters
					<< std::endl << "Error: " << dwGetStatusName(result) << std::endl;
			return false;
		}

		cam.sensor = salSensor;
		cam.numSiblings = 1;
		cameras->push_back(cam);
		++gNumCameras;
		// std::cout << "Initialize video " << count++ << std::endl;

		if(found == std::string::npos)
			break;
		idx = found + 1;
	}

	return true;
}

//------------------------------------------------------------------------------
void runPipeline(dwSoftISPHandle_t softISP, const dwImageProperties& rawImageProps,
		const dwImageProperties& rcbImageProps, float32_t framerate, dwContextHandle_t ctx, cudaStream_t m_cudaStream)
{
	typedef std::chrono::high_resolution_clock myclock_t;
	typedef std::chrono::time_point<myclock_t> timepoint_t;
	auto frameDuration         = std::chrono::milliseconds((int)(1000 / framerate));
	timepoint_t lastUpdateTime = myclock_t::now();

	// processing time of frameworks
	// auto beginDriveNetTime = std::chrono::high_resolution_clock::now();
	// auto beginLaneNetTime = std::chrono::high_resolution_clock::now();
	// auto beginOpenRoadNetTime = std::chrono::high_resolution_clock::now();
	auto beginPilotNetTime = std::chrono::high_resolution_clock::now();

	// crate RCB image for each camera, to be feeded as an array to drivenet
	std::vector<dwImageCUDA*> rcbArray(gNumCameras);
	{

		for(uint32_t camIdx = 0; camIdx < gNumCameras; ++camIdx) {
			rcbArray[camIdx] = new dwImageCUDA{};

			dwImageCUDA_create(rcbArray[camIdx], &rcbImageProps, DW_IMAGE_CUDA_PITCH);
		}
	}

	// RGBA image to display
	std::vector<dwImageCUDA*> rgbaImage(gNumCameras);
	{
		dwImageProperties rgbaImageProperties = rcbImageProps;
		rgbaImageProperties.type              = DW_IMAGE_CUDA;
		rgbaImageProperties.planeCount        = 1;
		rgbaImageProperties.pxlFormat         = DW_IMAGE_RGBA;
		rgbaImageProperties.pxlType           = DW_TYPE_UINT8;

		for(uint32_t camIdx = 0; camIdx < gNumCameras; ++camIdx){
			rgbaImage[camIdx] = new dwImageCUDA{};
			dwImageCUDA_create(rgbaImage[camIdx], &rgbaImageProperties, DW_IMAGE_CUDA_PITCH);
		}
	}

	// container for output of the stereo rectifier. The properties are set to RGBA interleaved uint8 because
	// the stereo rectifier supports it and also because we want to visualize the colored rectified images
	props = rcbImageProps;
	props.type = DW_IMAGE_CUDA;
	props.planeCount = 1;
	props.pxlFormat = DW_IMAGE_RGBA;
	props.pxlType = DW_TYPE_UINT8;
	props.width = rgbaImage[0]->prop.width;
	props.height = rgbaImage[0]->prop.height;

	// get calibrated cameras from the rig configuration
	if (!getRCBCameraFromRig(props, ctx)) {
		gRun = false;
	}

	// std::cout << "After getRGBACameraFromRig image size (width x height): " << props.width << "x" << props.height << std::endl;

	// get the extrinsics transformation matrices (NOTE that this sample assumes the stereo cameras are the
	// first two of the enumerated camera sensors and are ordered LEFT and RIGHT)
	dwTransformation left2Rig, right2Rig;
	CHECK_DW_ERROR(dwRigConfiguration_getSensorToRigTransformation(&left2Rig, 0, m_rigConfiguration));

	dwStatus res = dwRigConfiguration_getSensorToRigTransformation(&right2Rig, 1, m_rigConfiguration);
	/*
    // converting a 3D point on the lanegraph to pixel coordinates on an image 
    dwVector2f out; 
    out.x = -1.0; // invalid coordinates, out of bounds 
    out.y = -1.0; 

    // apply transformation on rigPt -> camPt 
    float camarr[3] = {0, 0, 0}; 
    float rigarr[3] = {rigPt.x, rigPt.y, rigPt.z}; 

    float rig2cam[16];
    Mat4_IsoInv(rig2cam, right2Rig.array); 
    Mat4_Axp(camarr, rig2cam, rigarr);

    // check whether the camarr will be inside the camera field of view 
    bool isInFov{false}; 
    dwCalibratedCamera_isRayInsideFOV(&isInFov, camarr[0], camarr[1], camarr[2], m_calibratedCamera); 
	 */
	// lane detection and free space detection
	// free-space detection
	if (res != DW_SUCCESS) //only compute free space boundary in image space
	{
		std::cerr << "Cannot parse rig configuration:  " << dwGetStatusName(res)
						<< std::endl;
		std::cerr << "Compute free space boundary in image space only."
				<< std::endl;
		res = dwFreeSpaceDetector_initializeFreeSpaceNet(&m_freeSpaceDetector, props.width, props.height, m_cudaStream, ctx);
	} else //compute free space boundary in image and vehicle coordinates
	{
		std::string maxDistanceStr = gArguments.get("maxDistance");
		if (maxDistanceStr != "50.0") {
			try {
				m_maxDistance = std::stof(maxDistanceStr);
				if (m_maxDistance < 0.0f) {
					std::cerr << "maxDistance cannot be negative." << std::endl;
					// return false;
				}
			} catch (...) {
				std::cerr << "Given maxDistance can't be parsed" << std::endl;
				// return false;
			}
		}
		res = dwFreeSpaceDetector_initializeCalibratedFreeSpaceNet(&m_freeSpaceDetector, props.width, props.height,
				m_cudaStream, left2Rig, m_maxDistance, m_cameraModel[0], ctx);
		if (res != DW_SUCCESS) {
			std::cerr << "Cannot initialize FreeSpaceNet: "
					<< dwGetStatusName(res) << std::endl;
			// return false;
		}
	}

	// lane detection
	res = dwLaneDetector_initializeLaneNet(&m_laneDetector, props.width, props.height, ctx);
	if (res != DW_SUCCESS) {
		std::cerr << "Cannot initialize LaneNet: " << dwGetStatusName(res)
						<< std::endl;
		// return false;
	}
	// std::cout << "props.width " << props.width << "props.height "
	//           << props.height << std::endl;

	std::string inputThreshold = gArguments.get("threshold");
	if (inputThreshold != "0.3") {
		try {
			m_threshold = std::stof(inputThreshold);
		} catch (...) {
			std::cerr << "Given threshold can't be parsed" << std::endl;
			// return false;
		}
	}

	res = dwLaneDetectorLaneNet_setDetectionThreshold(m_threshold, m_laneDetector);
	if (res != DW_SUCCESS) {
		std::cerr << "Cannot set PilotNet threshold: " << dwGetStatusName(res)
						<< std::endl;
		// return false;
	}
	// std::cout << "threshold: " << m_threshold << std::endl;

	res = dwLaneDetector_setCameraHandle(m_cameraModel[0], m_laneDetector);
	if (res != DW_SUCCESS) {
		std::cerr << "Cannot initialize lane detector calibrated camera: "
				<< dwGetStatusName(res) << std::endl;
		// return false;
	}

	res = dwLaneDetector_setCameraExtrinsics(left2Rig, m_laneDetector);
	if (res != DW_SUCCESS) {
		std::cerr << "Cannot initialize lane detector camera extrinsics: "
				<< dwGetStatusName(res) << std::endl;
		// return false;
	}

	res = dwLaneDetector_setMaxLaneDistance(m_maxDistance, m_laneDetector);
	if (res != DW_SUCCESS) {
		std::cerr
		<< "Cannot set lane detector maximum detection distance in meter: "
		<< dwGetStatusName(res) << std::endl;
		// return false;
	}
	// std::cout << "maxDistance: " << m_maxDistance << std::endl;

	float32_t fov;
	res = dwCalibratedCamera_getHorizontalFOV(&fov, m_cameraModel[0]);
	if (res != DW_SUCCESS) {
		std::cerr << "Cannot get camera horizontal FOV: "
				<< dwGetStatusName(res) << std::endl;
		// return false;
	}

	res = dwLaneDetectorLaneNet_setHorizontalFOV(RAD2DEG(fov), m_laneDetector);
	if (res != DW_SUCCESS) {
		std::cerr << "Cannot set camera horizontal FOV: "
				<< dwGetStatusName(res) << std::endl;
		// return false;
	}
	// std::cout << "camera horizontal FOV: " << fov << std::endl;

	// stereo, lane detection and free space detection

	m_levelStop = static_cast<uint8_t>(atoi(gArguments.get("level").c_str()));
	m_side = gArguments.get("single_side") == std::string("0") ? DW_STEREO_SIDE_BOTH : DW_STEREO_SIDE_LEFT;

	// initialize the stereo rectifier using the camera models (intrinsics) and transformations (extrinsics)
	CHECK_DW_ERROR(dwStereoRectifier_initialize(&m_stereoRectifier, m_cameraModel[0], m_cameraModel[1], left2Rig, right2Rig, ctx));

	// allocate memory for the rectified image
	for(uint32_t i=0; i<DW_STEREO_SIDE_COUNT; ++i) {
		dwImageCUDA_create(&m_outputRectified[i], &props, DW_IMAGE_CUDA_PITCH);
	}

	// initialize the image with same resolution
	// as the rectified image that will be used for display
	CHECK_DW_ERROR(dwStereoRectifier_getCropROI(&m_roi, m_stereoRectifier));
	// props.width = static_cast<udistanceint32_t>(m_roi.width);
	// props.height = static_cast<uint32_t>(m_roi.height);

	// std::cout << "ROI: " << m_roi.width << "x" << m_roi.height << std::endl;
	// std::cout << "After dwStereoRectifier_getCropROI image size (width x height): " << props.width << "x" << props.height << std::endl;

	// allocate memory for the anaglyph
	dwImageCUDA_create(&m_outputAnaglyph, &props, DW_IMAGE_CUDA_PITCH);

	// setup pyramids for the stereo algorithm. The algorithm accepts grayscale images at multiple resolutions
	// this is why we prepare a DW_IMAGE_R (single channel, gray scale)...
	dwImageProperties pyrProp = props;
	pyrProp.pxlFormat = DW_IMAGE_R;
	dwImageCUDA_create(&m_outputRectifiedR, &pyrProp, DW_IMAGE_CUDA_PITCH);

	// the stereo algorithm inputs two gaussian pyramids built from the rectified input. This is because the
	// algorithm requires a multi resolution representation.
	dwPyramidConfig pyramidConf;
	pyramidConf.dataType = DW_TYPE_UINT8;
	pyramidConf.height = props.height;
	pyramidConf.width = props.width;
	// the default input has a very high resolution so many levels guarantee a better coverage of the image
	pyramidConf.levelCount = m_levelCount;

	for (int32_t i = 0; i < DW_STEREO_SIDE_BOTH; ++i) {
		CHECK_DW_ERROR(dwPyramid_initialize(&m_pyramids[i], ctx, 0, pyramidConf));
	}

	// setup stereo algorithm
	// stereo parameters setup by the module. By default the level we stop at is 0, which is max resolution
	// in order to improve performance we stop at level specified by default as 1 in the input arguments
	dwStereoParams stereoParams;
	CHECK_DW_ERROR(dwStereo_initParams(&stereoParams));

	// set the same levels as the pyramid (it is not necessary that the stereo algorithm starts from the
	// lowest level of a pyramid, since this can be easily used by another module that requires more lower levels)
	// since the pyramid is built for the stereo purpose we set the levels the same as the pyramids. In other
	// use cases it can be possible that the pyramid has too many levels and not all are necessary for the
	// stereo algorithm, so we can decide to use less
	stereoParams.levelCount = pyramidConf.levelCount;

	// level at which to stop computing disparity map
	stereoParams.levelStop = m_levelStop;
	// m_levelStop = stereoParams.levelStop;

	// specifies which side to compute the disparity map from, if BOTH, LEFT or RIGHT only
	stereoParams.side = m_side;

	CHECK_DW_ERROR(dwStereo_initialize(&m_stereoAlgorithm, props.width, props.height, &stereoParams, ctx));

	// setup a format converter that converts from the RGBA output of the stereo rectifier to R as input
	// for the pyramid
	CHECK_DW_ERROR(dwImageFormatConverter_initialize(&m_RGBA2R, props.type, ctx));

	// finally, based on the resolution we build the disparity map at, we setup an image streamer
	// for display with that resolution
	// CHECK_DW_ERROR(dwStereo_getSize(&props.width, &props.height, m_levelStop, m_stereoAlgorithm));
	CHECK_DW_ERROR(dwStereo_getSize(&props.width, &props.height, stereoParams.levelStop, m_stereoAlgorithm));

	// dwImageCUDA_create(&m_colorDisparity, &pyrProp, DW_IMAGE_CUDA_PITCH);
	dwImageCUDA_create(&m_colorDisparity, &props, DW_IMAGE_CUDA_PITCH);

	// gain for the color coding, proportional to the level we stop at. Lower gain means flatter colors.
	// m_colorGain = COLOR_GAIN * (1 << m_levelStop);
	m_colorGain = COLOR_GAIN * (1 << (stereoParams.levelStop));

	// DriveNetNCameras
	{
		result = dwImageStreamer_initialize(&cuda2gl, &rgbaImage[0]->prop, DW_IMAGE_GL, ctx);
		if (result != DW_SUCCESS) {
			std::cerr << "Image streamer initialization failed: " << dwGetStatusName(result) << std::endl;
			gRun = false;
		}
	}

	// Initialize N streamers, where N is the number of ports used
	dwImageStreamerHandle_t input2cuda[gCameras.size()];
	for(uint32_t i = 0; i < gCameras.size(); ++i){
		dwImageStreamer_initialize(&input2cuda[i], &rawImageProps, DW_IMAGE_CUDA, ctx);
		// std::cout << "runPipeline ==> rawImageProps image with " << rawImageProps.width << "x"
		//           << rawImageProps.height << std::endl;
		// dwImageStreamer_initialize(&input2cuda[i], &props, DW_IMAGE_CUDA, ctx);
	}

	// uint32_t numClasses = gClassLabels.size();

	uint32_t frame = 0;
	uint32_t stopFrame = atoi(gArguments.get("stopFrame").c_str());
	uint32_t skipFrame = std::stoi(gArguments.get("skipFrame")) + 1;
	// uint32_t skipInference = std::stoi(gArguments.get("skipInference")) + 1;

	for(uint32_t camNum = 0; camNum < gCameras.size(); ++camNum){
		gRun = gRun && dwSensor_start(gCameras[camNum].sensor) == DW_SUCCESS;
	}

	while (gRun && !gWindow->shouldClose()) {
		std::this_thread::yield();

		// run with at most 30FPS when the input is a video
		if (gInputType.compare("video") == 0) {
			std::chrono::milliseconds timeSinceUpdate =
					std::chrono::duration_cast<std::chrono::milliseconds>(myclock_t::now() - lastUpdateTime);

			if (timeSinceUpdate < frameDuration)
				continue;
			lastUpdateTime = myclock_t::now();
		}

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		uint32_t countEndOfStream = 0;
		for(uint32_t camNum = 0, camIdx = 0; camNum < gCameras.size(); ++camNum){
			for(uint32_t sIdx = 0; sIdx < gCameras[camNum].numSiblings; ++sIdx, ++camIdx){

				// grab camera frame
				dwCameraFrameHandle_t frameHandle;
				{
					// result = dwSensorCamera_readFrame(&frameHandle, sIdx, 2000000, gCameras[camNum].sensor);
					result = dwSensorCamera_readFrame(&frameHandle, sIdx, 20000, gCameras[camNum].sensor);
					if (result == DW_END_OF_STREAM) {
						++countEndOfStream;
						continue;
					}
					else if (result != DW_SUCCESS) {
						std::cerr << "Cannot read frame: " << dwGetStatusName(result) << std::endl;
						continue;
					}
					else if(frame % skipFrame != 0){
						result = dwSensorCamera_returnFrame(&frameHandle);
						if (result != DW_SUCCESS) {
							std::cerr << "Cannot return frame: " << dwGetStatusName(result) << std::endl;
						}
						continue;
					}
				}

				// extract CUDA RAW image from the frame
				dwImageCUDA *rawImageCUDA;
				const dwImageDataLines* dataLines;
				{
					result = DW_SUCCESS;

					// depending if we are grabbing from camera, we need to use nvmedia->cuda streamer
					// or if we are grabbing from video, then we use cpu->cuda streamer

					dwImageCPU *rawImageCPU;

#ifdef VIBRANTE
					dwImageNvMedia *rawImageNvMedia;

					if (gInputType.compare("camera") == 0) {
						result = dwSensorCamera_getImageNvMedia(&rawImageNvMedia, DW_CAMERA_RAW_IMAGE,
								frameHandle);
					}else
#endif
					{
						result = dwSensorCamera_getImageCPU(&rawImageCPU, DW_CAMERA_RAW_IMAGE,
								frameHandle);
					}
					if (result != DW_SUCCESS) {
						std::cerr << "Cannot get raw image: " << dwGetStatusName(result) << std::endl;
						continue;
					}

					result = dwSensorCamera_getDataLines(&dataLines, frameHandle);
					if (result != DW_SUCCESS) {
						std::cerr << "Cannot get datalines: " << dwGetStatusName(result) << std::endl;
						continue;
					}

					// process
#ifdef VIBRANTE
					if (gInputType.compare("camera") == 0) {
						result = dwImageStreamer_postNvMedia(rawImageNvMedia, input2cuda[camNum]);
					}else
#endif
					{
						result = dwImageStreamer_postCPU(rawImageCPU, input2cuda[camNum]);
					}
					if (result != DW_SUCCESS) {
						std::cerr << "Cannot post CPU image: " << dwGetStatusName(result) << std::endl;
						continue;
					}

					result = dwImageStreamer_receiveCUDA(&rawImageCUDA, 10000, input2cuda[camNum]);
					if (result != DW_SUCCESS) {
						std::cerr << "Cannot get receive CUDA image: " << dwGetStatusName(result) << std::endl;
						continue;
					}
				}

				// Raw -> RCB & RGBA
				{
					dwSoftISP_bindRawInput(rawImageCUDA, softISP);
					dwSoftISP_bindDemosaicOutput(rcbArray[camIdx], softISP);
					dwSoftISP_bindTonemapOutput(rgbaImage[camIdx], softISP);
					result = dwSoftISP_processDeviceAsync(DW_SOFT_ISP_PROCESS_TYPE_DEMOSAIC | DW_SOFT_ISP_PROCESS_TYPE_TONEMAP,
							softISP);

					if (result != DW_SUCCESS) {
						std::cerr << "Cannot run rccb pipeline: " << dwGetStatusName(result) << std::endl;
						gRun = false;
						continue;
					}
				}

				// return RAW CUDA image back, we have now a copy through RawPipeline
				{
					result = dwImageStreamer_returnReceivedCUDA(rawImageCUDA, input2cuda[camNum]);
					if (result != DW_SUCCESS) {
						std::cerr << "Cannot return receive CUDA: " << dwGetStatusName(result) << std::endl;
						continue;
					}

#ifdef VIBRANTE
					if (gInputType.compare("camera") == 0) {
						dwImageNvMedia* imageNvmedia = nullptr;
						result = dwImageStreamer_waitPostedNvMedia(&imageNvmedia, 200000, input2cuda[camNum]);
					}else
#endif
					{
						dwImageCPU* cpuImage = nullptr;
						result = dwImageStreamer_waitPostedCPU(&cpuImage, 200000, input2cuda[camNum]);
					}
					if (result != DW_SUCCESS) {
						std::cerr << "Cannot wait posted: " << dwGetStatusName(result) << std::endl;
						continue;
					}
				}

				// camera frame is now free
				result = dwSensorCamera_returnFrame(&frameHandle);
				if (result != DW_SUCCESS) {
					std::cerr << "Cannot return frame: " << dwGetStatusName(result) << std::endl;
					continue;
				}
			}
		} // end of stereo camera frame

		// If cameras reached end of stream, reset sensors and trackers
		if(countEndOfStream == gNumCameras){
			for(uint32_t camIdx = 0; camIdx < gNumCameras; ++camIdx){
				std::cout << "Camera " << camIdx << " reached end of stream" << std::endl;
				dwSensor_reset(gCameras[camIdx].sensor);
			}
			resetDetector();
			resetTracker();

			frame = 0;
			gFrameInference = 0;

			continue;
		}

		// Skip frame
		if(frame % skipFrame != 0){
			++frame;
			continue;
		}

        // std::cout << "Input image frame size (left): " << rgbaImage[0]->prop.width << "x" << rgbaImage[0]->prop.height << std::endl;

        gridCellRect(&rect, gGrid, 0);
        dwRenderer_setRect(rect, gRenderer);

        dwImageStreamer_postCUDA(rgbaImage[DW_STEREO_SIDE_LEFT], cuda2gl);
        result = dwImageStreamer_receiveGL(&frameGL, 200000, cuda2gl);

        if( result == DW_SUCCESS ) {
            dwRenderer_renderTexture(frameGL->tex, frameGL->target, gRenderer);

            // overlay text
            dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, gRenderer);
            dwRenderer_renderText(150, 10, "Input Image Frame (Left)", gRenderer);

            dwImageStreamer_returnReceivedGL(frameGL, cuda2gl);
            dwImageCUDA *returnedFrame;
            dwImageStreamer_waitPostedCUDA(&returnedFrame, 200000, cuda2gl);
        } 

        // std::cout << "Input image frame size (right): " << rgbaImage[1]->prop.width << "x" << rgbaImage[1]->prop.height << std::endl;

        gridCellRect(&rect, gGrid, 1);
        dwRenderer_setRect(rect, gRenderer);

        dwImageStreamer_postCUDA(rgbaImage[DW_STEREO_SIDE_RIGHT], cuda2gl);
        result = dwImageStreamer_receiveGL(&frameGL, 200000, cuda2gl);

        if( result == DW_SUCCESS ) {
            dwRenderer_renderTexture(frameGL->tex, frameGL->target, gRenderer);

            // overlay text
            dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, gRenderer);
            dwRenderer_renderText(150, 10, "Input Image Frame (Right)", gRenderer);

            dwImageStreamer_returnReceivedGL(frameGL, cuda2gl);
            dwImageCUDA *returnedFrame;
            dwImageStreamer_waitPostedCUDA(&returnedFrame, 200000, cuda2gl);
        }

		// stereo
		CHECK_DW_ERROR(dwStereoRectifier_rectify(&m_outputRectified[DW_STEREO_SIDE_LEFT],
				&m_outputRectified[DW_STEREO_SIDE_RIGHT],
				rgbaImage[DW_STEREO_SIDE_LEFT], rgbaImage[DW_STEREO_SIDE_RIGHT], m_stereoRectifier));

		// std::cout << "Stereo Rectified image frame size (left): " << m_outputRectified[DW_STEREO_SIDE_LEFT].prop.width << "x" << m_outputRectified[DW_STEREO_SIDE_LEFT].prop.height << std::endl;
/* commented
		gridCellRect(&rect, gGrid, 0);
		dwRenderer_setRect(rect, gRenderer);

		dwImageStreamer_postCUDA(&m_outputRectified[DW_STEREO_SIDE_LEFT], cuda2gl);
		result = dwImageStreamer_receiveGL(&frameGL, 200000, cuda2gl);

		if( result == DW_SUCCESS ) {
			dwRenderer_renderTexture(frameGL->tex, frameGL->target, gRenderer);

			// overlay text
			dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, gRenderer);
			dwRenderer_renderText(150, 10, "Stereo Rectified Image (Left)", gRenderer);

			dwImageStreamer_returnReceivedGL(frameGL, cuda2gl);
			dwImageCUDA *returnedFrame;
			dwImageStreamer_waitPostedCUDA(&returnedFrame, 200000, cuda2gl);
		}

		// renderHorizontalLines();

		// std::cout << "Stereo Rectified image frame size (right): " << m_outputRectified[DW_STEREO_SIDE_RIGHT].prop.width << "x" << m_outputRectified[DW_STEREO_SIDE_RIGHT].prop.height << std::endl;

		gridCellRect(&rect, gGrid, 1);
		dwRenderer_setRect(rect, gRenderer);

		dwImageStreamer_postCUDA(&m_outputRectified[DW_STEREO_SIDE_RIGHT], cuda2gl);
		result = dwImageStreamer_receiveGL(&frameGL, 200000, cuda2gl);

		if( result == DW_SUCCESS ) {
			dwRenderer_renderTexture(frameGL->tex, frameGL->target, gRenderer);

			// overlay text
			dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, gRenderer);
			dwRenderer_renderText(150, 10, "Stereo Rectified Image (Right)", gRenderer);

			dwImageStreamer_returnReceivedGL(frameGL, cuda2gl);
			dwImageCUDA *returnedFrame;
			dwImageStreamer_waitPostedCUDA(&returnedFrame, 200000, cuda2gl);
		}
*/
		/*
        // remap the rectified images to their valid ROI
        dwImageCUDA croppedRectified[2];
        for (int32_t i = 0; i < DW_STEREO_SIDE_BOTH; ++i) {
            CHECK_DW_ERROR(dwImageCUDA_mapToROI(&croppedRectified[i], &m_outputRectified[i], m_roi));
        }

        // props.width = static_cast<uint32_t>(m_roi.width);
        // props.height = static_cast<uint32_t>(m_roi.height);

        std::cout << "Cropped Rectified image (croppedRectified): " << croppedRectified[DW_STEREO_SIDE_LEFT].prop.width << "x" << croppedRectified[DW_STEREO_SIDE_LEFT].prop.height << std::endl;
		 */

		// create anaglyph of the input (left red, right blue)
		// createAnaglyph(m_outputAnaglyph, croppedRectified[0], croppedRectified[1]);
		createAnaglyph(m_outputAnaglyph, m_outputRectified[0], m_outputRectified[1]);

		// build pyramids
		for (int32_t i = 0; i < DW_STEREO_SIDE_BOTH; ++i) {
			// convert RGBA to R
			CHECK_DW_ERROR(dwImageFormatConverter_copyConvertCUDA(&m_outputRectifiedR, &m_outputRectified[i],
					m_RGBA2R, 0));
			// CHECK_DW_ERROR(dwImageFormatConverter_copyConvertCUDA(&m_outputRectifiedR, &croppedRectified[i],
			//                                                       m_RGBA2R, 0));

			// feed image to build a pyramid
			CHECK_DW_ERROR(dwPyramid_build(&m_outputRectifiedR, m_pyramids[i]));
		}
		/*
        // build pyramids
        {
            // dw::common::ProfileCUDASection s(&m_profiler, "Pyramid build");
            dwImageCUDA planeImage{};
            for (int32_t i = 0; i < DW_STEREO_SIDE_BOTH; ++i) {
                // CHECK_DW_ERROR(dwImageCUDA_getPlaneAsImage(&planeImage, &croppedRectified[i], 0));
                CHECK_DW_ERROR(dwImageCUDA_getPlaneAsImage(&planeImage, &m_outputRectified[i], 0));
                CHECK_DW_ERROR(dwPyramid_build(&planeImage, m_pyramids[i]));
            }
        }
		 */

		// compute disparity
		CHECK_DW_ERROR(dwStereo_computeDisparity(m_pyramids[DW_STEREO_SIDE_LEFT],
				m_pyramids[DW_STEREO_SIDE_RIGHT],
				m_stereoAlgorithm));

		// the grayscale disparity is colorcoded for better results
		CHECK_DW_ERROR(dwStereo_getDisparity(&disparity, DW_STEREO_SIDE_LEFT, m_stereoAlgorithm));

		// dwTransformation *Q = {};
		// CHECK_DW_ERROR(dwStereoRectifier_getReprojectionMatrix(Q, m_stereoRectifier));

		dwImageCUDA image = *disparity;
		const size_t pitch = image.pitch[0];
		const uint32_t width = image.prop.width;

		// std::cout << "Disparity image frame size (left and right): " << image.prop.width << "x" << image.prop.height << std::endl;

		const uint32_t height = (image.prop.pxlFormat == DW_IMAGE_RCB ?  3 * image.prop.height : image.prop.height);

		// const uint32_t height = image.prop.height;

		// std::cout << "After Disparity ==> width = " << image.prop.width << "height: " << image.prop.height << std::endl;

		void* data = image.dptr[0];
		size_t bytesPerPixel = 0;

		switch(image.prop.pxlType) {
		case DW_TYPE_INT8:
		case DW_TYPE_UINT8:
			bytesPerPixel = 1;
			// std::cout << "bytesPerPixel = 1" << std::endl;
			break;
		case DW_TYPE_INT16:
		case DW_TYPE_UINT16:
		case DW_TYPE_FLOAT16:
			bytesPerPixel = 2;
			// std::cout << "bytesPerPixel = 2" << std::endl;
			break;
		case DW_TYPE_INT32:
		case DW_TYPE_UINT32:
		case DW_TYPE_FLOAT32:
			bytesPerPixel = 4;
			// std::cout << "bytesPerPixel = 4" << std::endl;
			break;
		case DW_TYPE_INT64:
		case DW_TYPE_UINT64:
		case DW_TYPE_FLOAT64:
			bytesPerPixel = 8;
			// std::cout << "bytesPerPixel = 8" << std::endl;
			break;
		case DW_TYPE_UNKNOWN:
		case DW_TYPE_BOOL:
		default:
			std::cerr << "Invalid or unsupported pixel type." << std::endl;
			return;
		}

		const size_t host_pitch = width * bytesPerPixel;
		char* hbuf = new char[host_pitch * height];
		if (hbuf)
		{
			cudaError_t error = cudaMemcpy2D(static_cast<void*>(hbuf), host_pitch, data, pitch, host_pitch, height, cudaMemcpyDeviceToHost);

			if (error != cudaSuccess) {
				std::cerr << "dumpOutput, memcopy failed." << std::endl;
				return;
			}
			/*
            minDisp = hbuf[0];
            maxDisp = hbuf[0];
	    for(std::size_t i = 0; i < (host_pitch * height); ++i) {
   		if ( hbuf[i] < minDisp )
		    minDisp = hbuf[i] ;
   		if ( hbuf[i] > maxDisp )
		    maxDisp = hbuf[i] ;
            }
			 */
			// depth = ((5.8*300)/minDisp); // (((5.8*300)/minDisp)*0.001)
			// std::cout << "Disparity in pixels (minimum & maximum): " << minDisp << " & " << maxDisp << std::endl; // b=30cm and f=5.8mm/1937 pixels
			delete[] hbuf;
		}

		// colorCode(&m_colorDisparity, *disparity, 4.0 * (1 << m_levelStop));
		colorCode(&m_colorDisparity, *disparity, 4.0 * (1 << (stereoParams.levelStop)));
		// colorCode(&m_colorDisparity, *disparity, m_colorGain);

		CHECK_DW_ERROR(dwStereo_getConfidence(&confidence, DW_STEREO_SIDE_LEFT, m_stereoAlgorithm));

		// confidence map is used to show occlusions as black pixels
		mixDispConf(&m_colorDisparity, *confidence, m_invalidThreshold >= 0.0f);

		// mix disparity and confidence, where confidence is occlusion, show black, where it is invalidity show white, leave
		// as is otherwise. See README for instructions on how to change the threshold of invalidity
		// if ((m_occlusionInfill == DW_FALSE) && (m_occlusion == DW_TRUE)) {
		//     mixDispConf(&m_colorDisparity, *confidence, m_invalidThreshold >= 0.0f);
		// }

		// std::cout << "Stereo disparity image frame size (left): " << m_outputRectified[DW_STEREO_SIDE_LEFT].prop.width << "x" << m_outputRectified[DW_STEREO_SIDE_LEFT].prop.height << std::endl;
/*
        // display left disparity
        gridCellRect(&rect, gGrid, 0);
        dwRenderer_setRect(rect, gRenderer);

        dwImageStreamer_postCUDA(&m_colorDisparity, cuda2gl);
        result = dwImageStreamer_receiveGL(&frameGL, 200000, cuda2gl);

        if( result == DW_SUCCESS ) {
            dwRenderer_renderTexture(frameGL->tex, frameGL->target, gRenderer);

            // overlay text
            dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, gRenderer);
            dwRenderer_renderText(150, 10, "Stereo Disparity Image (Left)", gRenderer);

            dwImageStreamer_returnReceivedGL(frameGL, cuda2gl);
            dwImageCUDA *returnedFrame;
            dwImageStreamer_waitPostedCUDA(&returnedFrame, 200000, cuda2gl);
        }    

      	// std::cout << "Stereo disparity image frame size (right): " << m_outputRectified[DW_STEREO_SIDE_RIGHT].prop.width << "x" << m_outputRectified[DW_STEREO_SIDE_RIGHT].prop.height << std::endl;

        // display right disparity
        gridCellRect(&rect, gGrid, 1);
        dwRenderer_setRect(rect, gRenderer);

        dwImageStreamer_postCUDA(&m_colorDisparity, cuda2gl);
        result = dwImageStreamer_receiveGL(&frameGL, 200000, cuda2gl);

        if( result == DW_SUCCESS ) {
            dwRenderer_renderTexture(frameGL->tex, frameGL->target, gRenderer);

            // overlay text
            dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, gRenderer);
            dwRenderer_renderText(150, 10, "Stereo Disparity Image (Right)", gRenderer);

            dwImageStreamer_returnReceivedGL(frameGL, cuda2gl);
            dwImageCUDA *returnedFrame;
            dwImageStreamer_waitPostedCUDA(&returnedFrame, 200000, cuda2gl);
        }
*/
		/*
      	// std::cout << "Stereo perception image frame size (left): " << m_outputRectified[DW_STEREO_SIDE_LEFT].prop.width << "x" << m_outputRectified[DW_STEREO_SIDE_LEFT].prop.height << std::endl;

        // display detections
        gridCellRect(&rect, gGrid, 6);
        dwRenderer_setRect(rect, gRenderer);

        dwImageStreamer_postCUDA(rgbaImage[DW_STEREO_SIDE_LEFT], cuda2gl);
        result = dwImageStreamer_receiveGL(&frameGL, 200000, cuda2gl);

        if( result == DW_SUCCESS ) {
            dwRenderer_renderTexture(frameGL->tex, frameGL->target, gRenderer);

            // overlay text
            dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, gRenderer);
            dwRenderer_renderText(150, 10, "DNN modules", gRenderer);

            dwImageStreamer_returnReceivedGL(frameGL, cuda2gl);
            dwImageCUDA *returnedFrame;
            dwImageStreamer_waitPostedCUDA(&returnedFrame, 200000, cuda2gl);
        }
		 */
		// std::cout << "Stereo disparity image frame size (left): " << m_outputRectified[DW_STEREO_SIDE_LEFT].prop.width << "x" << m_outputRectified[DW_STEREO_SIDE_LEFT].prop.height << std::endl;

		// display disparity
		gridCellRect(&rect, gGrid, 2);
		dwRenderer_setRect(rect, gRenderer);

		dwImageStreamer_postCUDA(&m_colorDisparity, cuda2gl);
		result = dwImageStreamer_receiveGL(&frameGL, 200000, cuda2gl);

		if( result == DW_SUCCESS ) {
			dwRenderer_renderTexture(frameGL->tex, frameGL->target, gRenderer);

			// overlay text
			dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, gRenderer);
			dwRenderer_renderText(150, 10, "Stereo Disparity Image (Reference)", gRenderer);

			dwImageStreamer_returnReceivedGL(frameGL, cuda2gl);
			dwImageCUDA *returnedFrame;
			dwImageStreamer_waitPostedCUDA(&returnedFrame, 200000, cuda2gl);
		}

		// display stereo perception
		gridCellRect(&rect, gGrid, 3);
		dwRenderer_setRect(rect, gRenderer);
		// dwImageStreamer_postCUDA(rgbaImage[DW_STEREO_SIDE_LEFT], cuda2gl);
		dwImageStreamer_postCUDA(&m_outputRectified[DW_STEREO_SIDE_LEFT], cuda2gl);
		// dwImageStreamer_postCUDA(&m_outputRectifiedR, cuda2gl);
		result = dwImageStreamer_receiveGL(&frameGL, 200000, cuda2gl);

		if( result == DW_SUCCESS ) {
			dwRenderer_renderTexture(frameGL->tex, frameGL->target, gRenderer);

			// overlay text
			dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, gRenderer);
			dwRenderer_renderText(150, 10, "Stereo Perception Image", gRenderer);

			dwImageStreamer_returnReceivedGL(frameGL, cuda2gl);
			dwImageCUDA *returnedFrame;
			dwImageStreamer_waitPostedCUDA(&returnedFrame, 200000, cuda2gl);
		}

		if (driveNetDetector)
		{
			// DriveNetNCameras
			// Detect, track and render objects
			// std::vector<std::pair<dwBox2D, std::string>> boxList[gNumCameras*numClasses];
			// detectTrack(rcbArray.data(), skipInference, boxList, ctx);
			// gWindow->makeCurrent();

			// Detect, track and render objects
			// inferDetectorAsync(&m_outputRectifiedR);
			inferDetectorAsync(&m_outputRectified[DW_STEREO_SIDE_LEFT]);
			// inferDetectorAsync(&m_colorDisparity);
			// inferTrackerAsync(rgbaImage[DW_STEREO_SIDE_LEFT]);
			// inferTrackerAsync(&m_outputRectified[DW_STEREO_SIDE_LEFT]);
			processResults(ctx);

			// render boxes with labels for each detected class
			for (size_t classIdx = 0; classIdx < getNumClasses(); classIdx++)
			{
				// render bounding box
				dwRenderer_setColor(boxColors[classIdx % MAX_BOX_COLORS], gRenderer);
				drawBoxesWithLabels(getResult(classIdx), static_cast<float32_t>(rcbImageProps.width), static_cast<float32_t>(rcbImageProps.height), gLineBuffer, gRenderer);
				drawBoxesWithLabels(getResult(classIdx), static_cast<float32_t>(m_outputRectified[DW_STEREO_SIDE_LEFT].prop.width), static_cast<float32_t>(m_outputRectified[DW_STEREO_SIDE_LEFT].prop.height), gLineBuffer, gRenderer);
				// drawBoxesWithLabels(getResult(classIdx), static_cast<float32_t>(m_colorDisparity.prop.width), static_cast<float32_t>(m_colorDisparity.prop.height), gLineBuffer, gRenderer);
			}

			// draw ROI of the first image
			drawROIs(drivenetParams.ROIs[0], DW_RENDERER_COLOR_LIGHTBLUE, gLineBuffer, gRenderer);

			// draw ROI of the second image
			drawROIs(drivenetParams.ROIs[1], DW_RENDERER_COLOR_YELLOW, gLineBuffer, gRenderer);

			// auto endDriveNetTime = std::chrono::high_resolution_clock::now();
        	// std::chrono::milliseconds timeProcessDriveNet = std::chrono::duration_cast<std::chrono::milliseconds>(endDriveNetTime - beginDriveNetTime);
        	// std::cout << "01. DriveNet Framework is (in ms): " << timeProcessDriveNet.count() << std::endl;
            //    beginDriveNetTime = std::chrono::high_resolution_clock::now();

		}

		if (laneDetector)
		{
			// lane detector
			// std::cout << "laneDetector" << std::endl;

			dwLaneDetection lanes {};
			dwStatus res = dwLaneDetector_processDeviceAsync(&m_outputRectified[DW_STEREO_SIDE_LEFT], m_laneDetector);
			res = res == DW_SUCCESS ? dwLaneDetector_interpretHost(m_laneDetector) : res;

			if (res != DW_SUCCESS) {
				std::cerr << "runDetector failed with: " << dwGetStatusName(res) << std::endl;
			}

			dwLaneDetector_getLaneDetections(&lanes, m_laneDetector);
			drawLaneMarkings(lanes, 6.0f, gLineBuffer, gRenderer);

			/* auto endLaneNetTime = std::chrono::high_resolution_clock::now();
	        std::chrono::milliseconds timeProcessLaneNet = std::chrono::duration_cast<std::chrono::milliseconds>(endLaneNetTime - beginLaneNetTime);
	        std::cout << "02. LaneNet Framework is (in ms): " <<  timeProcessLaneNet.count() << std::endl;
                beginLaneNetTime = std::chrono::high_resolution_clock::now();*/
		}

		if (freeSpaceDetector)
		{
			// free space detection
			// std::cout << "freeSpaceDetector" << std::endl;

			dwFreeSpaceDetection boundary {};

			res = dwFreeSpaceDetector_processDeviceAsync(&m_outputRectified[DW_STEREO_SIDE_LEFT], m_freeSpaceDetector);
			res = res == DW_SUCCESS ? dwFreeSpaceDetector_interpretHost(m_freeSpaceDetector) : res;

			if (res != DW_SUCCESS) {
				std::cerr << "runDetector failed with: " << dwGetStatusName(res) << std::endl;
			}
			else
				dwFreeSpaceDetector_getBoundaryDetection(&boundary, m_freeSpaceDetector);

			drawFreeSpaceBoundary(&boundary, gLineBuffer, gRenderer);

			/* auto endOpenRoadNetTime = std::chrono::high_resolution_clock::now();
        	std::chrono::milliseconds timeProcessOpenRoadNet = std::chrono::duration_cast<std::chrono::milliseconds>(endOpenRoadNetTime - beginOpenRoadNetTime);
        	std::cout << "03. OpenRoadNet Framework is (in ms): " <<  timeProcessOpenRoadNet.count() << std::endl;
                beginOpenRoadNetTime = std::chrono::high_resolution_clock::now(); */
		}
		// end of detections

		auto endPilotNetTime = std::chrono::high_resolution_clock::now();
		std::chrono::milliseconds timeProcessPilotNet = std::chrono::duration_cast<std::chrono::milliseconds>(endPilotNetTime - beginPilotNetTime);
		std::cout << "Stereo Perception is (in ms): " <<  timeProcessPilotNet.count() << std::endl;
		beginPilotNetTime = std::chrono::high_resolution_clock::now();

		gWindow->swapBuffers();

		++frame;

		if(stopFrame && frame == stopFrame)
			break;
	} // end of while loop

	for(uint32_t camNum = 0; camNum < gCameras.size(); ++camNum) {
		dwSensor_stop(gCameras[camNum].sensor);
	}

	for(uint32_t camIdx = 0; camIdx < gNumCameras; ++camIdx){
		dwImageCUDA_destroy(rgbaImage[camIdx]);
		dwImageCUDA_destroy(rcbArray[camIdx]);
		delete rcbArray[camIdx];
		delete rgbaImage[camIdx];
	}

	// Release streamers
	for(uint32_t i = 0; i < gCameras.size(); ++i){
		dwImageStreamer_release(&input2cuda[i]);
	}
	dwImageStreamer_release(&cuda2gl);
}

//------------------------------------------------------------------------------
void resizeWindowCallbackGrid(int width, int height) {
	configureGrid(&gGrid, width, height, gImageWidth, gImageHeight, 4);
	// configureGrid(&gGrid, width, height, gImageWidth, gImageHeight, gNumCameras);
}

//------------------------------------------------------------------------------
void releaseGrid()
{
	// stereo

	for(uint32_t i=0; i<DW_STEREO_SIDE_COUNT; ++i) {
		dwImageCUDA_destroy(&m_outputRectified[i]);
	}

	dwImageCUDA_destroy(&m_outputAnaglyph);
	dwImageCUDA_destroy(&m_colorDisparity);
	dwImageCUDA_destroy(&m_outputRectifiedR);

	dwImageFormatConverter_release(&m_RGBA2R);
	dwStereoRectifier_release(&m_stereoRectifier);

	for(int i=0; i<DW_STEREO_SIDE_COUNT; ++i)
		dwPyramid_release(&m_pyramids[i]);

	dwStereo_release(&m_stereoAlgorithm);
	dwObjectDetector_release(&m_driveNetDetector);
	dwObjectTracker_release(&m_objectTracker);

	if (gCameras.size()) {
		for(uint32_t i = 0; i < gCameras.size(); ++i)
			dwSAL_releaseSensor(&(gCameras[i].sensor));
	}

	if (gSal) {
		dwSAL_release(&gSal);
	}
}

//------------------------------------------------------------------------------
bool getRCBCameraFromRig(dwImageProperties& rcbImageProps, dwContextHandle_t ctx)
{
	// std::cout << "getRCBCameraFromRig() " << std::endl;

	if (gArguments.get("rigconfig").empty()) {
		throw std::runtime_error("Rig configuration file not specified, please provide a rig "
				"configuration file with the calibration of the stereo camera");
	}

	CHECK_DW_ERROR(dwRigConfiguration_initializeFromFile(&m_rigConfiguration, gSdk,
			gArguments.get("rigconfig").c_str()));

	// check if 2 cameras are in there (only the stereo cameras are present for demonstrative purposes)
	uint32_t cameraCount = 0;
	CHECK_DW_ERROR(dwRigConfiguration_getSensorCount(&cameraCount, m_rigConfiguration));
	if (cameraCount != 2) {
		throw std::runtime_error("Wrong number of cameras in rig file.");
	}

	// get the camera models stored in the rig (contain the intrinsics)
	dwCameraRigHandle_t rig;
	CHECK_DW_ERROR(dwCameraRig_initializeFromConfig(&rig, &cameraCount, m_cameraModel, 2,
			gSdk, m_rigConfiguration));
	CHECK_DW_ERROR(dwCameraRig_release(&rig));

	// uint32_t totalWidth = 0;
	for (int32_t i = 0; i < DW_STEREO_SIDE_BOTH; ++i) {
		dwCameraModel cameraModel{};
		CHECK_DW_ERROR(dwRigConfiguration_getCameraModel(&cameraModel, i, m_rigConfiguration));

		switch (cameraModel) {
		case DW_CAMERA_MODEL_PINHOLE: {
			// std::cout << "DW_CAMERA_MODEL_PINHOLE " << std::endl;
			dwPinholeCameraConfig config{};
			CHECK_DW_ERROR(dwRigConfiguration_getPinholeCameraConfig(&config, i, m_rigConfiguration));
			CHECK_DW_ERROR(dwCalibratedCamera_initializePinhole(&m_calibratedCamera, ctx, &config));
			rcbImageProps.width = config.width;
			rcbImageProps.height = config.height;

			if (config.height != rcbImageProps.height) {
				throw std::runtime_error("Camera height and height in rig.xml don't match.");
			}
		}
		break;
		case DW_CAMERA_MODEL_OCAM: {
			// std::cout << "DW_CAMERA_MODEL_OCAM " << std::endl;
			dwOCamCameraConfig config{};
			CHECK_DW_ERROR(dwRigConfiguration_getOCamCameraConfig(&config, i, m_rigConfiguration));
			CHECK_DW_ERROR(dwCalibratedCamera_initializeOCam(&m_calibratedCamera, ctx, &config);)
			rcbImageProps.width = config.width;
			rcbImageProps.height = config.height;

			if (config.height != rcbImageProps.height) {
				throw std::runtime_error("Camera height and height in rig.xml don't match.");
			}
		}
		break;
		case DW_CAMERA_MODEL_FTHETA: {
			// std::cout << "DW_CAMERA_MODEL_FTHETA " << std::endl;
			dwFThetaCameraConfig config{};
			CHECK_DW_ERROR(dwRigConfiguration_getFThetaCameraConfig(&config, i, m_rigConfiguration));
			CHECK_DW_ERROR(dwCalibratedCamera_initializeFTheta(&m_calibratedCamera, ctx, &config);)
			rcbImageProps.width = config.width;
			rcbImageProps.height = config.height;

			if (config.height != rcbImageProps.height) {
				throw std::runtime_error("Camera height and height in rig.xml don't match.");
			}
		}
		break;
		default:
			throw std::runtime_error("Unknown type of camera was provided by rig configuration");
		}
	}
	return true;
}

// PilotNet

//#######################################################################################
bool initDriveworks(dwContextHandle_t ctx)
{
	// std::cout << "initDriveworks! " << std::endl;
	// create a Logger to log to console
	// we keep the ownership of the logger at the application level
	dwLogger_initialize(getConsoleLoggerCallback(true));
	dwLogger_setLogLevel(DW_LOG_DEBUG);

	// instantiate Driveworks SDK context
	dwContextParameters sdkParams{};

	std::string path = DataPath::get();
	sdkParams.dataPath = path.c_str();

#ifdef VIBRANTE
	// std::cout << "initDriveworks on Drive PX 2! " << std::endl;
	sdkParams.eglDisplay = gWindow->getEGLDisplay();
#endif

	return dwInitialize(&ctx, DW_VERSION, &sdkParams) == DW_SUCCESS;
}

//------------------------------------------------------------------------------
bool initDetector(const dwImageProperties& rcbImageProps, dwContextHandle_t ctx, cudaStream_t m_stream)
{
	// std::cout << "initDetector! " << std::endl;

	// Initialize DriveNet network
	CHECK_DW_ERROR(dwDriveNet_initDefaultParams(&m_driveNetParams));

	// Set up max number of proposals and clusters
	m_driveNetParams.maxClustersPerClass = maxClustersPerClass;
	m_driveNetParams.maxProposalsPerClass = maxProposalsPerClass;

	// Set batch size to 2 for foveal.
	m_driveNetParams.networkBatchSize = DW_DRIVENET_BATCHSIZE_2;
	m_driveNetParams.networkPrecision = DW_DRIVENET_PRECISION_FP32;

	CHECK_DW_ERROR(dwDriveNet_initialize(&m_driveNet, &objectClusteringHandles, &driveNetClasses,
			&numDriveNetClasses, ctx, &m_driveNetParams));

	// Initialize Objec Detector from DriveNet
	dwObjectDetectorDNNParams tmpDNNParams;
	CHECK_DW_ERROR(dwObjectDetector_initDefaultParams(&tmpDNNParams, &detectorParams));

	// Enable fusing objects from different ROIs
	detectorParams.enableFuseObjects = DW_TRUE;

	// Two images will be given as input. Each image is a region on the image received from camera.
	detectorParams.maxNumImages = 2U;

	CHECK_DW_ERROR(dwObjectDetector_initializeFromDriveNet(&m_driveNetDetector, ctx, m_driveNet,
			&detectorParams));
	CHECK_DW_ERROR(dwObjectDetector_setCUDAStream(m_stream, m_driveNetDetector));

	// since our input images might have a different aspect ratio as the input to drivenet
	// we setup the ROI such that the crop happens from the top of the image
	float32_t aspectRatio = 1.0f;
	{
		dwBlobSize inputBlob;
		CHECK_DW_ERROR(dwDriveNet_getInputBlobsize(&inputBlob, m_driveNet));
		aspectRatio = static_cast<float32_t>(inputBlob.height) / static_cast<float32_t>(inputBlob.width);
	}

	// 1st image is a full resolution image as it comes out from the RawPipeline (cropped to DriveNet aspect ratio)
	dwRect fullROI;
	{
		fullROI = {0, 0, static_cast<int32_t>(rcbImageProps.width),
				static_cast<int32_t>(rcbImageProps.width * aspectRatio)};
		dwTransformation2D transformation = {{1.0f, 0.0f, 0.0f,
				0.0f, 1.0f, 0.0f,
				0.0f, 0.0f, 1.0f}};
		CHECK_DW_ERROR(dwObjectDetector_setROI(0, &fullROI, &transformation, m_driveNetDetector));
	}

	// 2nd image is a cropped out region within the 1/4-3/4 of the original image in the center
	{
		dwRect ROI = {fullROI.width/4, fullROI.height/4,
				fullROI.width/2, fullROI.height/2};
		// dwRect ROI = {fullROI.width/20, fullROI.height/20,
		//              fullROI.width/2, fullROI.height/2};
		dwTransformation2D transformation = {{1.0f, 0.0f, 0.0f,
				0.0f, 1.0f, 0.0f,
				0.0f, 0.0f, 1.0f}};
		CHECK_DW_ERROR(dwObjectDetector_setROI(1, &ROI, &transformation, m_driveNetDetector));
	}

	// fill out member structure according to the ROIs
	CHECK_DW_ERROR(dwObjectDetector_getROI(&detectorParams.ROIs[0],
			&detectorParams.transformations[0], 0, m_driveNetDetector));
	CHECK_DW_ERROR(dwObjectDetector_getROI(&detectorParams.ROIs[1],
			&detectorParams.transformations[1], 1, m_driveNetDetector));

	// Get which label name for each class id
	classLabels.resize(numDriveNetClasses);
	for (uint32_t classIdx = 0U; classIdx < numDriveNetClasses; ++classIdx)
	{
		const char *classLabel;
		CHECK_DW_ERROR(dwDriveNet_getClassLabel(&classLabel, classIdx, m_driveNet));
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
bool initTracker(const dwImageProperties& rcbImageProps, dwContextHandle_t ctx, cudaStream_t m_stream)
{
	// std::cout << "initTracker! " << std::endl;
	// initialize ObjectTracker - it will be required to track detected instances over multiple frames

	dwObjectFeatureTrackerParams featureTrackingParams;
	dwObjectTrackerParams objectTrackingParams[DW_OBJECT_MAX_CLASSES];
	CHECK_DW_ERROR(dwObjectTracker_initDefaultParams(&featureTrackingParams, objectTrackingParams,
			numDriveNetClasses));
	featureTrackingParams.maxFeatureCount = 8000;
	featureTrackingParams.detectorScoreThreshold = 0.0001f;
	featureTrackingParams.iterationsLK = 10;
	featureTrackingParams.windowSizeLK = 8;

	for (uint32_t classIdx = 0U; classIdx < numDriveNetClasses; ++classIdx)
	{
		objectTrackingParams[classIdx].confRateTrackMax = 0.05f;
		objectTrackingParams[classIdx].confRateTrackMin = 0.01f;
		objectTrackingParams[classIdx].confRateDetect = 0.5f;
		objectTrackingParams[classIdx].confThreshDiscard = 0.0f;
		objectTrackingParams[classIdx].maxFeatureCountPerBox = 200;
	}

	{
		CHECK_DW_ERROR(dwObjectTracker_initialize(&m_objectTracker, ctx,
				&rcbImageProps, &featureTrackingParams,
				objectTrackingParams, numDriveNetClasses));
	}

	CHECK_DW_ERROR(dwObjectTracker_setCUDAStream(m_stream, m_objectTracker));

	return true;
}

//#######################################################################################
bool initRenderer(const dwImageProperties& rcbImageProps, dwContextHandle_t ctx)
{
	// std::cout << "initRenderer! " << std::endl;
	// init renderer
	m_screenRectangle.height = gWindow->height();
	m_screenRectangle.width = gWindow->width();
	m_screenRectangle.x = 0;
	m_screenRectangle.y = 0;

	// setup line buffer for drawing horizontal lines. We draw horizontal lines to show that a rectified pair
	// has the same pixels lying on horizontal line
	unsigned int maxLines = 20000;
	setupRenderer(gRenderer, m_screenRectangle, ctx);
	setupLineBuffer(rcbImageProps, gLineBuffer, maxLines, ctx);

	return true;
}

//#######################################################################################
void setupRenderer(dwRendererHandle_t &renderer, const dwRect &screenRect, dwContextHandle_t ctx)
{
	CHECK_DW_ERROR( dwRenderer_initialize(&renderer, ctx) );

	float32_t boxColor[4] = {0.0f,1.0f,0.0f,1.0f};
	dwRenderer_setColor(boxColor, renderer);
	dwRenderer_setLineWidth(2.0f, renderer);
	dwRenderer_setRect(screenRect, renderer);
}

//#######################################################################################
void setupLineBuffer(const dwImageProperties& rcbImageProps, dwRenderBufferHandle_t &lineBuffer, unsigned int maxLines, dwContextHandle_t ctx)
{
	dwRenderBufferVertexLayout layout;
	layout.posFormat   = DW_RENDER_FORMAT_R32G32_FLOAT;
	layout.posSemantic = DW_RENDER_SEMANTIC_POS_XY;
	layout.colFormat   = DW_RENDER_FORMAT_NULL;
	layout.colSemantic = DW_RENDER_SEMANTIC_COL_NULL;
	layout.texFormat   = DW_RENDER_FORMAT_NULL;
	layout.texSemantic = DW_RENDER_SEMANTIC_TEX_NULL;
	dwRenderBuffer_initialize(&lineBuffer, layout, DW_RENDER_PRIM_LINELIST, maxLines, ctx);
	dwRenderBuffer_set2DCoordNormalizationFactors((float32_t)rcbImageProps.width,
			(float32_t)rcbImageProps.height, lineBuffer);

	/*
    for (float i = 70.0f; i < (rcbImageProps.height - 70.0f); i += 40.0f) {
        m_lines.push_back(dwLineSegment2Df{dwVector2f{0, i},
                                                           dwVector2f{
                                                               static_cast<float>(rcbImageProps.width)
                                                               , i}});
    }
	 */
}

//#######################################################################################
void renderCameraTexture(dwImageStreamerHandle_t streamer, dwRendererHandle_t renderer)
{
	dwImageGL *frameGL = nullptr;

	if (dwImageStreamer_receiveGL(&frameGL, 30000, streamer) != DW_SUCCESS) {
		std::cerr << "did not received GL frame within 30ms" << std::endl;
	} else {
		// render received texture
		dwRenderer_renderTexture(frameGL->tex, frameGL->target, renderer);

		// overlay text
		dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, renderer);
		dwRenderer_renderText(20, 20, "Rectified", renderer);

		dwImageStreamer_returnReceivedGL(frameGL, streamer);
	}
}

/*
//#######################################################################################
void renderHorizontalLines()
{
    // makes lines red and slightly transparent
    const float32_t colorRedTrans[4] = {1.0f, 0.0f, 0.0f, 0.5f};
    renderLineSegments(m_lines, 1.0f, colorRedTrans);
}
 */

//------------------------------------------------------------------------------
void resetDetector()
{
	for (uint32_t classIdx = 0U; classIdx < classLabels.size(); ++classIdx) {
		numClusters[classIdx] = 0U;
		numMergedObjects[classIdx] = 0U;
		numTrackedObjects[classIdx] = 0U;
	}
}

//------------------------------------------------------------------------------
void resetTracker()
{
	CHECK_DW_ERROR(dwObjectTracker_reset(m_objectTracker));
}

//------------------------------------------------------------------------------
void inferDetectorAsync(const dwImageCUDA* rcbImage)
{
	// we feed two images to the DriveNet module, the first one will have full ROI
	// the second one, is the same image, however with an ROI cropped in the center
	const dwImageCUDA* rcbImagePtr[2] = {rcbImage, rcbImage};
	CHECK_DW_ERROR(dwObjectDetector_inferDeviceAsync(rcbImagePtr, 2U, m_driveNetDetector));
}

//------------------------------------------------------------------------------
void inferTrackerAsync(const dwImageCUDA* rcbImage)
{
	// track feature points on the rcb image
	CHECK_DW_ERROR(dwObjectTracker_featureTrackDeviceAsync(rcbImage, m_objectTracker));
}

//------------------------------------------------------------------------------
void processResults(dwContextHandle_t ctx)
{

	CHECK_DW_ERROR(dwObjectDetector_interpretHost(2U, m_driveNetDetector));

	// for each detection class, we do
	for (uint32_t classIdx = 0U; classIdx < classLabels.size(); ++classIdx)
	{
		// track detection from last frame given new feature tracker responses
		CHECK_DW_ERROR(dwObjectTracker_boxTrackHost(objectsTracked[classIdx].data(), &numTrackedObjects[classIdx],
				objectsMerged[classIdx].data(), numMergedObjects[classIdx],
				classIdx, m_objectTracker));

		// extract new detections from DriveNet
		CHECK_DW_ERROR(dwObjectDetector_getDetectedObjects(objectProposals[classIdx].data(),
				&numProposals[classIdx],
				0U, classIdx, m_driveNetDetector));

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
				maxClustersPerClass, toBeMerged, sizes, 2U, 0.1f, 0.1f, ctx));

		// extract now the actual bounding box of merged response in pixel coordinates to render on screen
		dnnBoxList[classIdx].resize(numMergedObjects[classIdx]);

		// for each detection object, we do
		for (uint32_t objIdx = 0U; objIdx < numMergedObjects[classIdx]; ++objIdx)
		{
			const dwObject &obj = objectsMerged[classIdx][objIdx];
			dwBox2D &box = dnnBoxList[classIdx][objIdx].first;
			box.x = static_cast<int32_t>(std::round(obj.box.x));
			box.y = static_cast<int32_t>(std::round(obj.box.y));
			box.width = static_cast<int32_t>(std::round(obj.box.width));
			box.height = static_cast<int32_t>(std::round(obj.box.height));

			// box is a cropped out region within the 1/4-3/4 of the original box in the center for better disparity
			// dwRect m_roi = {box.width/4, box.height/4,
			//                 box.width/2, box.height/2};
			dwRect m_roi = {box.x + box.width/4, box.y + box.height/4, box.width/2, box.height/2};
			// dwRect m_roi = {box.x, box.y, box.width, box.height};

			CHECK_DW_ERROR(dwImageCUDA_mapToROI(&objDisparity, disparity, m_roi));

			dwImageCUDA image = objDisparity;
			size_t pitch = image.pitch[0];
			uint32_t width = image.prop.width;
			uint32_t height = (image.prop.pxlFormat == DW_IMAGE_RCB ? 3 * image.prop.height : image.prop.height);
			void* data = image.dptr[0];
			size_t bytesPerPixel = 0;

			switch (image.prop.pxlType)
			{
			case DW_TYPE_INT8:
			case DW_TYPE_UINT8:
				bytesPerPixel = 1;
				// std::cout << "bytesPerPixel = 1" << std::endl;
				break;
			case DW_TYPE_INT16:
			case DW_TYPE_UINT16:
			case DW_TYPE_FLOAT16:
				bytesPerPixel = 2;
				// std::cout << "bytesPerPixel = 2" << std::endl;
				break;
			case DW_TYPE_INT32:
			case DW_TYPE_UINT32:
			case DW_TYPE_FLOAT32:
				bytesPerPixel = 4;
				// std::cout << "bytesPerPixel = 4" << std::endl;
				break;
			case DW_TYPE_INT64:
			case DW_TYPE_UINT64:
			case DW_TYPE_FLOAT64:
				bytesPerPixel = 8;
				// std::cout << "bytesPerPixel = 8" << std::endl;
				break;
			case DW_TYPE_UNKNOWN:
			case DW_TYPE_BOOL:
			default:
				std::cerr << "Invalid or unsupported pixel type." << std::endl;
				break;
			}

			const size_t host_pitch = width * bytesPerPixel;
			char* hbuf = new char[host_pitch * height];
			int sum = 0;
			std::size_t i = 0;
			int max_d = 2;
			int min_d = 1024;

			if (hbuf)
			{
				cudaError_t error = cudaMemcpy2D(static_cast<void*>(hbuf), host_pitch, data, pitch, width, height, cudaMemcpyDeviceToHost);

				if (error != cudaSuccess) {
					std::cerr << "dumpOutput, memcopy failed." << std::endl;
					return;
				}

				int cc3 = 0;
				for(i = 0; i < (host_pitch * height); ++i)
				{
					if(hbuf[i] > 1)
					{
						sum += hbuf[i];
						cc3 = cc3 +1;
						if(min_d > hbuf[i])
							min_d = hbuf[i];
						if(max_d < hbuf[i])
							max_d = hbuf[i];
					}
				}

				if(cc3 != 0)
					sum = sum/cc3;
				else
					sum = 1;

				delete[] hbuf;

				std::stringstream ss1, ss2, alpha, beta, distance;

				ss1.precision(2);
				ss1.setf(std::ios::fixed, std::ios::floatfield);

				ss2.precision(2);
				ss2.setf(std::ios::fixed, std::ios::floatfield);

				ss2.precision(2);
				ss2.setf(std::ios::fixed, std::ios::floatfield);

				ss2 << (((1944.8f * 0.3f)/ (max_d))) << "m"; // 0.001f depth based on maximum disparity
				ss1 << (((1944.8f * 0.3f)/ (sum))) << "m"; // 0.001f  depth based on average disparity

				depth[objIdx] = (((1944.8f * 0.3f)/(sum))); // depth = ((focal length * baseline) / disparity)
/*
				if (m_roi.x > (1920.0f/2.0f))
					// angle[objIdx] = atan(((1920.0f / 2.0f) + m_roi.x) / depth[objIdx]); // view angle = ((horizontal pixels/2 + object position) / depth)
				    alphaAngle[objIdx] = (atan(((1920.0f / 2.0f) + m_roi.x) / 1944.8f) * (180.0/3.14)); // view angle = ((horizontal pixels/2 + object position) / depth)
				else
					// angle[objIdx] = atan(((1920.0f / 2.0f) - m_roi.x) / depth[objIdx]); // view angle = ((horizontal pixels/2 - object position) / depth)
					alphaAngle[objIdx] = (atan(((1920.0f / 2.0f) - m_roi.x) / 1944.8f) * (180.0/3.14)); // view angle = ((horizontal pixels/2 - object position) / depth)

				if (m_roi.y > (1208.0f/2.0f))
				    betaAngle[objIdx] = (atan(((1208.0f / 2.0f) + m_roi.y) / 1944.8f) * (180.0/3.14)); // view angle = ((horizontal pixels/2 + object position) / depth)
				else
					betaAngle[objIdx] = (atan(((1208.0f / 2.0f) - m_roi.y) / 1944.8f) * (180.0/3.14)); // view angle = ((horizontal pixels/2 - object position) / depth)

				alpha.precision(0);
				alpha.setf(std::ios::fixed, std::ios::floatfield);
				alpha << alphaAngle[objIdx];

				beta.precision(0);
				beta.setf(std::ios::fixed, std::ios::floatfield);
				beta << betaAngle[objIdx];

				distance.precision(2);
				distance.setf(std::ios::fixed, std::ios::floatfield);
				distance << depth[objIdx];
*/
				// dnnBoxList[classIdx][objIdx].second = classLabels[classIdx];
				// dnnBoxList[classIdx][objIdx].second = classLabels[classIdx] + std::to_string(objIdx)
				//                                                            + " H:" + alpha.str().c_str()
				//		                                                    + " V:" + beta.str().c_str()
				//		                                                    + " D:" + distance.str().c_str();
				// dnnBoxList[classIdx][objIdx].second = classLabels[classIdx] + std::to_string(objIdx) + " depth:" + ss1.str().c_str() + " " + ss2.str().c_str();
				dnnBoxList[classIdx][objIdx].second = classLabels[classIdx] + std::to_string(objIdx) + " depth:" + ss1.str().c_str();
			}
		}
	}
}

//------------------------------------------------------------------------------
void drawROIs(dwRect roi, const float32_t color[4],
		dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer) {
	float32_t x_start = static_cast<float32_t>(roi.x);
	float32_t x_end = static_cast<float32_t>(roi.x + roi.width);
	float32_t y_start = static_cast<float32_t>(roi.y);
	float32_t y_end = static_cast<float32_t>(roi.y + roi.height);

	float32_t *coords = nullptr;
	uint32_t maxVertices = 0;
	uint32_t vertexStride = 0;
	dwRenderBuffer_map(&coords, &maxVertices, &vertexStride, renderBuffer);
	coords[0] = x_start;
	coords[1] = y_start;
	coords += vertexStride;
	coords[0] = x_start;
	coords[1] = y_end;
	coords += vertexStride;
	coords[0] = x_start;
	coords[1] = y_end;
	coords += vertexStride;
	coords[0] = x_end;
	coords[1] = y_end;
	coords += vertexStride;
	coords[0] = x_end;
	coords[1] = y_end;
	coords += vertexStride;
	coords[0] = x_end;
	coords[1] = y_start;
	coords += vertexStride;
	coords[0] = x_end;
	coords[1] = y_start;
	coords += vertexStride;
	coords[0] = x_start;
	coords[1] = y_start;
	dwRenderBuffer_unmap(8, renderBuffer);
	dwRenderer_setColor(color, renderer);
	dwRenderer_setLineWidth(2, renderer);
	dwRenderer_renderBuffer(renderBuffer, renderer);
}

//------------------------------------------------------------------------------
const std::vector<std::pair<dwBox2D,std::string>>& getResult(uint32_t classIdx)
		{
	return dnnBoxList[classIdx];
		}

//#######################################################################################
void drawLaneMarkings(const dwLaneDetection &lanes,
		float32_t laneWidth, dwRenderBufferHandle_t renderBuffer,
		dwRendererHandle_t renderer) {

	// const float32_t DW_RENDERER_COLOR_DARKYELLOW[4] = { 180.0f / 255.0f, 180.0f
	// 		/ 255.0f, 10.0f / 255.0f, 1.0f };
	const float32_t DW_RENDERER_COLOR_CYAN[4] = { 10.0f / 255.0f, 230.0f
			/ 255.0f, 230.0f / 255.0f, 1.0f };

	float32_t colors[5][4];

	// memcpy(colors[0], DW_RENDERER_COLOR_DARKYELLOW, sizeof(colors[0]));
	memcpy(colors[0], DW_RENDERER_COLOR_YELLOW, sizeof(colors[0]));
	memcpy(colors[1], DW_RENDERER_COLOR_RED, sizeof(colors[1]));
	memcpy(colors[2], DW_RENDERER_COLOR_GREEN, sizeof(colors[2]));
	memcpy(colors[3], DW_RENDERER_COLOR_BLUE, sizeof(colors[3]));
	// memcpy(colors[4], DW_RENDERER_COLOR_CYAN, sizeof(colors[3]));
	memcpy(colors[4], DW_RENDERER_COLOR_CYAN, sizeof(colors[4]));

	drawLaneMarkingsCustomColor(colors, 5, lanes, laneWidth, renderBuffer, renderer);
}

//#######################################################################################
void drawLaneMarkingsCustomColor(float32_t laneColors[][4],
		uint32_t nColors, const dwLaneDetection &lanes, float32_t laneWidth,
		dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer) {

	drawLaneDetectionROI(renderBuffer, renderer);

	// draw lanes
	for (uint32_t laneIdx = 0; laneIdx < lanes.numLaneMarkings; ++laneIdx) {

		const dwLaneMarking& laneMarking = lanes.laneMarkings[laneIdx];

		dwLanePositionType category = laneMarking.positionType;

		if (category == DW_LANEMARK_POSITION_ADJACENT_LEFT)
			dwRenderer_setColor(laneColors[0 % nColors], renderer);
		else if (category == DW_LANEMARK_POSITION_EGO_LEFT)
			dwRenderer_setColor(laneColors[1 % nColors], renderer);
		else if (category == DW_LANEMARK_POSITION_EGO_RIGHT)
			dwRenderer_setColor(laneColors[2 % nColors], renderer);
		else if (category == DW_LANEMARK_POSITION_ADJACENT_RIGHT)
			dwRenderer_setColor(laneColors[3 % nColors], renderer);
		else
			dwRenderer_setColor(laneColors[4 % nColors], renderer);

		dwRenderer_setLineWidth(laneWidth, renderer);

		float32_t* coords = nullptr;
		uint32_t maxVertices = 0;
		uint32_t vertexStride = 0;
		dwRenderBuffer_map(&coords, &maxVertices, &vertexStride, renderBuffer);

		uint32_t n_verts = 0;
		dwVector2f previousP {};
		bool firstPoint = true;

		// get the points from lane markings
		// for each point we need to transform it into world space
		for (uint32_t pointIdx = 0; pointIdx < laneMarking.numPoints; ++pointIdx) {

			dwVector2f center;
			center.x = laneMarking.imagePoints[pointIdx].x;
			center.y = laneMarking.imagePoints[pointIdx].y;

			// print out the point in image space
			// printf("[Image Point] x: %f, y: %f\n", center.x, center.y);

			// u and v are coordinates in image space
			// to transform them, we need some 3D world space coordinates, let them be the following x, y and z

			float32_t u = center.x;
			float32_t v = center.y;

			// transform bottom center of the camera into 3D world space
			float32_t x, y, z;

			// this is the camera which acquired the original image
			// print its values in ray space
			CHECK_DW_ERROR(dwCalibratedCamera_pixel2Ray(&x, &y, &z, m_calibratedCamera, u, v));
			// printf("[Ray Point] x: %f, y: %f, z: %f\n", x, y, z);

			if (firstPoint) { // Special case for the first point
				previousP = center;
				firstPoint = false;
			} else {
				n_verts += 2;

				if (n_verts > maxVertices)
					break;

				coords[0] = static_cast<float32_t>(previousP.x);
				coords[1] = static_cast<float32_t>(previousP.y);
				coords += vertexStride;

				coords[0] = static_cast<float32_t>(center.x);
				coords[1] = static_cast<float32_t>(center.y);
				coords += vertexStride;

				previousP = center;
			}
		}

		dwRenderBuffer_unmap(n_verts, renderBuffer);
		dwRenderer_renderBuffer(renderBuffer, renderer);
	}
}

//#######################################################################################
void drawLaneDetectionROI(dwRenderBufferHandle_t renderBuffer,
		dwRendererHandle_t renderer) {

	dwRect roi{};
	dwLaneDetector_getDetectionROI(&roi, m_laneDetector);
	float32_t x_start = static_cast<float32_t>(roi.x);
	float32_t x_end = static_cast<float32_t>(roi.x + roi.width);
	float32_t y_start = static_cast<float32_t>(roi.y);
	float32_t y_end = static_cast<float32_t>(roi.y + roi.height);
	float32_t *coords = nullptr;
	uint32_t maxVertices = 0;
	uint32_t vertexStride = 0;
	dwRenderBuffer_map(&coords, &maxVertices, &vertexStride, renderBuffer);
	coords[0] = x_start;
	coords[1] = y_start;
	coords += vertexStride;
	coords[0] = x_start;
	coords[1] = y_end;
	coords += vertexStride;
	coords[0] = x_start;
	coords[1] = y_end;
	coords += vertexStride;
	coords[0] = x_end;
	coords[1] = y_end;
	coords += vertexStride;
	coords[0] = x_end;
	coords[1] = y_end;
	coords += vertexStride;
	coords[0] = x_end;
	coords[1] = y_start;
	coords += vertexStride;
	coords[0] = x_end;
	coords[1] = y_start;
	coords += vertexStride;
	coords[0] = x_start;
	coords[1] = y_start;
	dwRenderBuffer_unmap(8, renderBuffer);
	dwRenderer_setColor(DW_RENDERER_COLOR_YELLOW, renderer);
	dwRenderer_setLineWidth(2, renderer);
	dwRenderer_renderBuffer(renderBuffer, renderer);
}

//#######################################################################################
void drawFreeSpaceBoundary(dwFreeSpaceDetection* boundary,
		dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer) {
	drawFreeSpaceDetectionROI(renderBuffer, renderer);

	uint32_t n_verts = 0;
	float32_t* coords = nullptr;
	uint32_t maxVertices = 0;
	uint32_t vertexStride = 0;
	dwFreeSpaceBoundaryType category = boundary->boundaryType[0];
	float32_t maxWidth = 8.0; // 10 meters as a step, [0, 10) will have max line width
	float32_t witdhRatio = 0.8;
	float32_t dist2Width[20];
	dist2Width[0] = maxWidth;

	for (uint32_t i = 1; i < 20; i++)
		dist2Width[i] = dist2Width[i - 1] * witdhRatio;

	float32_t prevWidth, curWidth = maxWidth / 2;

	/*
        if(m_isRig) {
            prevWidth = dist2Width[static_cast<uint32_t>(boundary->boundaryWorldPoint[0].x/10)];
        } else {
            prevWidth = curWidth;
        }
	 */

	prevWidth = curWidth;

	dwRenderer_setLineWidth(prevWidth, renderer);

	if (category == DW_BOUNDARY_TYPE_OTHER)
		dwRenderer_setColor(DW_RENDERER_COLOR_YELLOW, renderer);
	else if (category == DW_BOUNDARY_TYPE_CURB)
		dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, renderer);
	else if (category == DW_BOUNDARY_TYPE_VEHICLE)
		dwRenderer_setColor(DW_RENDERER_COLOR_RED, renderer);
	else if (category == DW_BOUNDARY_TYPE_PERSON)
		dwRenderer_setColor(DW_RENDERER_COLOR_BLUE, renderer);

	dwRenderBuffer_map(&coords, &maxVertices, &vertexStride, renderBuffer);

	for (uint32_t i = 1; i < boundary->numberOfBoundaryPoints; ++i) {
		/*
                if(m_isRig) {
                    curWidth = dist2Width[static_cast<uint32_t>(boundary->boundaryWorldPoint[i].x/10)];
                }
		 */
		if (boundary->boundaryType[i] != boundary->boundaryType[i - 1]
		                                                        || curWidth != prevWidth) {
			dwRenderBuffer_unmap(n_verts, renderBuffer);
			dwRenderer_renderBuffer(renderBuffer, renderer);

			coords = nullptr;
			maxVertices = 0;
			vertexStride = 0;
			n_verts = 0;
			dwRenderer_setLineWidth(curWidth, renderer);

			category = boundary->boundaryType[i];
			if (category == DW_BOUNDARY_TYPE_OTHER)
				dwRenderer_setColor(DW_RENDERER_COLOR_YELLOW, renderer);
			else if (category == DW_BOUNDARY_TYPE_CURB)
				dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, renderer);
			else if (category == DW_BOUNDARY_TYPE_VEHICLE)
				dwRenderer_setColor(DW_RENDERER_COLOR_RED, renderer);
			else if (category == DW_BOUNDARY_TYPE_PERSON)
				dwRenderer_setColor(DW_RENDERER_COLOR_BLUE, renderer);

			dwRenderBuffer_map(&coords, &maxVertices, &vertexStride,
					renderBuffer);
		}

		n_verts += 2;

		if (n_verts > maxVertices)
			break;

		coords[0] = static_cast<float32_t>(boundary->boundaryImagePoint[i - 1].x);
		coords[1] =	static_cast<float32_t>(boundary->boundaryImagePoint[i - 1].y);
		coords += vertexStride;

		coords[0] = static_cast<float32_t>(boundary->boundaryImagePoint[i].x);
		coords[1] = static_cast<float32_t>(boundary->boundaryImagePoint[i].y);
		coords += vertexStride;
		prevWidth = curWidth;
	}

	dwRenderBuffer_unmap(n_verts, renderBuffer);
	dwRenderer_renderBuffer(renderBuffer, renderer);
}

//#######################################################################################
void drawFreeSpaceDetectionROI(dwRenderBufferHandle_t renderBuffer,
		dwRendererHandle_t renderer) {
	dwRect roi { };
	dwFreeSpaceDetector_getDetectionROI(&roi, m_freeSpaceDetector);
	float32_t x_start = static_cast<float32_t>(roi.x);
	float32_t x_end = static_cast<float32_t>(roi.x + roi.width);
	float32_t y_start = static_cast<float32_t>(roi.y);
	float32_t y_end = static_cast<float32_t>(roi.y + roi.height);
	float32_t *coords = nullptr;
	uint32_t maxVertices = 0;
	uint32_t vertexStride = 0;
	dwRenderBuffer_map(&coords, &maxVertices, &vertexStride, renderBuffer);
	coords[0] = x_start;
	coords[1] = y_start;
	coords += vertexStride;
	coords[0] = x_start;
	coords[1] = y_end;
	coords += vertexStride;
	coords[0] = x_start;
	coords[1] = y_end;
	coords += vertexStride;
	coords[0] = x_end;
	coords[1] = y_end;
	coords += vertexStride;
	coords[0] = x_end;
	coords[1] = y_end;
	coords += vertexStride;
	coords[0] = x_end;
	coords[1] = y_start;
	coords += vertexStride;
	coords[0] = x_end;
	coords[1] = y_start;
	coords += vertexStride;
	coords[0] = x_start;
	coords[1] = y_start;
	dwRenderBuffer_unmap(8, renderBuffer);
	dwRenderer_setColor(DW_RENDERER_COLOR_YELLOW, renderer);
	dwRenderer_setLineWidth(2, renderer);
	dwRenderer_renderBuffer(renderBuffer, renderer);
}

//------------------------------------------------------------------------------
void keyPress(int key)
{
	if (key == GLFW_KEY_O) {
		std::cout<<"Toggle occlusion"<<std::endl;
		m_occlusion = (m_occlusion == DW_TRUE) ? DW_FALSE : DW_TRUE;
		dwStereo_setOcclusionTest(m_occlusion, m_stereoAlgorithm);
	} else if (key == GLFW_KEY_K) {
		if (m_occlusion == DW_TRUE) {
			std::cout<<"Toggle occlusion infill"<<std::endl;
			m_occlusionInfill = (m_occlusionInfill == DW_TRUE) ? DW_FALSE : DW_TRUE;
			dwStereo_setOcclusionInfill(m_occlusionInfill, m_stereoAlgorithm);
		} else {
			std::cout<<"Cannot toggle occlusion infill, occlusion test is off"<<std::endl;
		}
	} else if (key == GLFW_KEY_I) {
		std::cout<<"Toggle invalidity infill"<<std::endl;
		m_infill = (m_infill == DW_TRUE) ? DW_FALSE : DW_TRUE;
		dwStereo_setInfill(m_infill, m_stereoAlgorithm);
	} else if (key == GLFW_KEY_0) {
		std::cout<<"Refinement 0"<<std::endl;
		dwStereo_setRefinementLevel(0, m_stereoAlgorithm);
	} else if (key == GLFW_KEY_1) {
		std::cout<<"Refinement 1"<<std::endl;
		dwStereo_setRefinementLevel(1, m_stereoAlgorithm);
	} else if (key == GLFW_KEY_2) {
		std::cout<<"Refinement 2"<<std::endl;
		dwStereo_setRefinementLevel(2, m_stereoAlgorithm);
	} else if (key == GLFW_KEY_3) {
		std::cout<<"Refinement 3"<<std::endl;
		dwStereo_setRefinementLevel(3, m_stereoAlgorithm);
	} else if (key == GLFW_KEY_4) {
		std::cout<<"Refinement 4"<<std::endl;
		dwStereo_setRefinementLevel(4, m_stereoAlgorithm);
	} else if (key == GLFW_KEY_5) {
		std::cout<<"Refinement 5"<<std::endl;
		dwStereo_setRefinementLevel(5, m_stereoAlgorithm);
	} else if (key == GLFW_KEY_6) {
		std::cout<<"Refinement 6"<<std::endl;
		dwStereo_setRefinementLevel(6, m_stereoAlgorithm);
	}else if (key == GLFW_KEY_W) {
		m_colorGain += 0.5f;
		if (m_colorGain > COLOR_GAIN_MAX) {
			m_colorGain = COLOR_GAIN_MAX;
		}
		std::cout<<"Color gain "<<m_colorGain<<std::endl;
	} else if (key == GLFW_KEY_Q) {
		m_colorGain -= 0.5f;
		if (m_colorGain < 0.0f) {
			m_colorGain = 0.0f;
		}
		std::cout<<"Color gain "<<m_colorGain<<std::endl;
	} else if (key == GLFW_KEY_KP_ADD) {
		m_invalidThreshold += 1.0f;
		if (m_invalidThreshold > INV_THR_MAX) {
			m_invalidThreshold = INV_THR_MAX;
		}
		dwStereo_setInvalidThreshold(m_invalidThreshold, m_stereoAlgorithm);
		std::cout<<"Invalidity thr " << m_invalidThreshold << std::endl;
	} else if (key == GLFW_KEY_KP_SUBTRACT) {
		m_invalidThreshold -= 1.0f;
		if (m_invalidThreshold < -1.0f) {
			m_invalidThreshold = -1.0f;
			dwStereo_setInvalidThreshold(0.0f, m_stereoAlgorithm);
		}
		if (m_invalidThreshold >= 0.0f) {
			dwStereo_setInvalidThreshold(m_invalidThreshold, m_stereoAlgorithm);
			std::cout<<"Invalidity thr "<< m_invalidThreshold << std::endl;
		} else {
			std::cout<<"Invalidity off "<<std::endl;
		}
	}

	// stop application
	if (key == GLFW_KEY_ESCAPE)
		gRun = false;

	// take screenshot
	// if (key == GLFW_KEY_S)
	//    gTakeScreenshot = true;
}
