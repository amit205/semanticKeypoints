#pragma once

#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "buffers.h"

namespace ORB_SLAM2
{
// Options for the network
struct Options {
    // Use 16 bit floating point type for inference
    bool FP16 = false;
    // Batch sizes to optimize for.
    std::vector<int32_t> optBatchSizes = {1};
    // Maximum allowable batch size
    int32_t maxBatchSize = 1;
    // Max allowable GPU memory to be used for model conversion, in bytes.
    // Applications should allow the engine builder as much workspace as they can afford;
    // at runtime, the SDK allocates no more than this and typically less.
    size_t maxWorkspaceSize = 4000000000;
    // GPU device index
    int deviceIndex = 0;
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};

// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

class Engine {
public:
    Engine(const Options& options);
    ~Engine();
    // Build the network
    bool build(std::string onnxModelPath, int level);
    // Load and prepare the network for inference
    bool loadNetwork();
    // Run inference.
    bool runInference(const std::vector<cv::Mat>& inputFaceChips, std::vector<cv::Mat>& featureVectors_descriptor_raw, std::vector<cv::Mat>& featureVectors_detector_logits, std::vector<cv::Mat>& featureVectors_detector);//, std::vector<cv::Mat>& featureVectors_segmentation);
private:
    // Converts the engine options into a string
    std::string serializeEngineOptions(const Options& options);

    void getGPUUUIDs(std::vector<std::string>& gpuUUIDs);

    bool doesFileExist(const std::string& filepath);

    TRTUniquePtr<nvinfer1::ICudaEngine> m_engine = nullptr;
    TRTUniquePtr<nvinfer1::IExecutionContext> m_context = nullptr;
    const Options& m_options;
    Logger m_logger;
    samplesCommon::ManagedBuffer m_inputBuff;
    samplesCommon::ManagedBuffer m_outputBuff_descriptor_raw;
    //samplesCommon::ManagedBuffer m_outputBuff_descriptor;
    samplesCommon::ManagedBuffer m_outputBuff_detector_logits;
    samplesCommon::ManagedBuffer m_outputBuff_detector;
   // samplesCommon::ManagedBuffer m_outputBuff_segmentation;
    size_t m_prevBatchSize = 0;
    std::string m_engineName;
    cudaStream_t m_cudaStream = nullptr;
};
}
