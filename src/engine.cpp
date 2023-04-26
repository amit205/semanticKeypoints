#include <iostream>
#include <fstream>

#include "engine.h"
#include "NvOnnxParser.h"

namespace ORB_SLAM2
{

void Logger::log(Severity severity, const char *msg) noexcept {
    // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
    // For the sake of this tutorial, will just log to the console.

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

bool Engine::doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

Engine::Engine(const Options &options)
    : m_options(options) {}

bool Engine::build(std::string onnxModelPath, int level) {
    // Only regenerate the engine file if it has not already been generated for the specified options
    // m_engineName = serializeEngineOptions(m_options);
    m_engineName = onnxModelPath.substr(onnxModelPath.find_last_of("/\\") + 1) + serializeEngineOptions(m_options)+std::to_string(level);
    std::cout << "Searching for engine file with name: " << m_engineName << std::endl;

    if (doesFileExist(m_engineName)) {
        std::cout << "Engine found, not regenerating..." << std::endl;
        return true;
    }

    // Was not able to find the engine file, generate...
    std::cout << "Engine not found, generating..." << std::endl;

    // Create our engine builder.
    auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Set the max supported batch size
    builder->setMaxBatchSize(m_options.maxBatchSize);

    // Define an explicit batch size and then create the network.
    // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = TRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer to the parser.
    // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.

    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Save the input height, width, and channels.
    // Require this info for inference.
    const auto input = network->getInput(0);
    const auto output = network->getOutput(0);
    const auto inputName = input->getName();
    const auto inputDims = input->getDimensions();
    std::cout << inputName << " " << inputDims << std::endl;
    int inputC, inputH, inputW;
    inputC = inputDims.d[3];
    switch (level) {
        case 0:
          //  inputH = 370;
          //  inputW = 1226;
            inputH = 368;
            inputW = 1224;
            break;
        case 1:
          //  inputH = 308;
          //  inputW = 1022;
            inputH = 304;
            inputW = 1016;
            break;
        case 2:
          //  inputH = 257;
          //  inputW = 851;
            inputH = 256;
            inputW = 848;
            break;
        case 3:
          //  inputH = 214;
          //  inputW = 709;
            inputH = 208;
            inputW = 704;
            break;
    }

    auto config = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }
    
    // Specify the optimization profiles and the
    IOptimizationProfile* defaultProfile = builder->createOptimizationProfile();
    defaultProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputH, inputW, inputC));
    defaultProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(1, inputH, inputW, inputC));
    defaultProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputH, inputW, inputC));
    config->addOptimizationProfile(defaultProfile);

    // Specify all the optimization profiles.
    for (const auto& optBatchSize: m_options.optBatchSizes) {
        if (optBatchSize == 1) {
            continue;
        }

        if (optBatchSize > m_options.maxBatchSize) {
            throw std::runtime_error("optBatchSize cannot be greater than maxBatchSize!");
        }
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputH, inputW, inputC));
        profile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(optBatchSize, inputH, inputW, inputC));
        profile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputH, inputW, inputC));
        config->addOptimizationProfile(profile);

    }

    config->setMaxWorkspaceSize(m_options.maxWorkspaceSize);

    if (m_options.FP16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream) {
        return false;
    }
    config->setProfileStream(*profileStream);

    // Build the engine
    TRTUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(m_engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << m_engineName << std::endl;

    return true;
}

Engine::~Engine() {
    if (m_cudaStream) {
        cudaStreamDestroy(m_cudaStream);
    }
}

bool Engine::loadNetwork() {
    // Read the serialized model from disk
    std::ifstream file(m_engineName, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    TRTUniquePtr<IRuntime> runtime{createInferRuntime(m_logger)};
    if (!runtime) {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    m_engine = TRTUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    m_context = TRTUniquePtr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    auto cudaRet = cudaStreamCreate(&m_cudaStream);
    if (cudaRet != 0) {
        throw std::runtime_error("Unable to create cuda stream");
    }

    return true;
}

bool Engine::runInference(const std::vector<cv::Mat> &inputFaceChips, std::vector<cv::Mat>& featureVectors_descriptor_raw, std::vector<cv::Mat>& featureVectors_detector_logits, std::vector<cv::Mat>& featureVectors_detector)//, std::vector<cv::Mat>& featureVectors_segmentation) 
{
    auto dims = m_engine->getBindingDimensions(0);
   
    auto img = inputFaceChips[0];
  
    Dims4 inputDims = {static_cast<int32_t>(inputFaceChips.size()), img.size[0], img.size[1], dims.d[3]};
  
    m_context->setBindingDimensions(0, inputDims);
    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all input dimensions specified.");
    }

    auto outputIndex_descriptor_raw = m_engine->getBindingIndex("descriptor_head");
    auto outputL_descriptor_raw = m_engine->getBindingDimensions(outputIndex_descriptor_raw);
    Dims4 outputDims_descriptor_raw = {static_cast<int32_t>(inputFaceChips.size()), img.size[0]/8, img.size[1]/8, outputL_descriptor_raw.d[3]};
  

    //auto outputIndex_descriptor = m_engine->getBindingIndex("descriptor_head_1");
    //auto outputL_descriptor = m_engine->getBindingDimensions(outputIndex_descriptor);
    
    auto outputIndex_detector_logits = m_engine->getBindingIndex("detector_head");
    auto outputL_detector_logits = m_engine->getBindingDimensions(outputIndex_detector_logits);
    Dims4 outputDims_detector_logits = {static_cast<int32_t>(inputFaceChips.size()), img.size[0]/8, img.size[1]/8, outputL_detector_logits.d[3]};
   

    auto outputIndex_detector = m_engine->getBindingIndex("detector_head_1");
    auto outputL_detector = m_engine->getBindingDimensions(outputIndex_detector);    
    Dims3 outputDims_detector = {static_cast<int32_t>(inputFaceChips.size()), img.size[0], img.size[1]};
  
    // comment out following if loading SuperPoint model
    //auto outputIndex_segmentation = m_engine->getBindingIndex("segmentation_head");
    //auto outputL_segmentation = m_engine->getBindingDimensions(outputIndex_segmentation);
    //Dims4 outputDims_segmentation = {static_cast<int32_t>(inputFaceChips.size()), img.size[0], img.size[1], outputL_segmentation.d[3]};
    
    auto batchSize = static_cast<int32_t>(inputFaceChips.size());
    //Only reallocate buffers if the batch size has changed
    if (m_prevBatchSize != inputFaceChips.size()) {

        m_inputBuff.hostBuffer.resize(inputDims);
        m_inputBuff.deviceBuffer.resize(inputDims);

        m_outputBuff_descriptor_raw.hostBuffer.resize(outputDims_descriptor_raw);
        m_outputBuff_descriptor_raw.deviceBuffer.resize(outputDims_descriptor_raw);

        //m_outputBuff_descriptor.hostBuffer.resize(outputDims_descriptor);
        //m_outputBuff_descriptor.deviceBuffer.resize(outputDims_descriptor);

        m_outputBuff_detector_logits.hostBuffer.resize(outputDims_detector_logits);
        m_outputBuff_detector_logits.deviceBuffer.resize(outputDims_detector_logits);

        m_outputBuff_detector.hostBuffer.resize(outputDims_detector);
        m_outputBuff_detector.deviceBuffer.resize(outputDims_detector);
        // comment out following if loading SuperPoint model
        //m_outputBuff_segmentation.hostBuffer.resize(outputDims_segmentation);
        //m_outputBuff_segmentation.deviceBuffer.resize(outputDims_segmentation);

        m_prevBatchSize = batchSize;
    }

    auto* hostDataBuffer = static_cast<float*>(m_inputBuff.hostBuffer.data());

    for (size_t batch = 0; batch < inputFaceChips.size(); ++batch) {
        auto image = inputFaceChips[batch];
        // Preprocess code
        //image.convertTo(image, CV_32FC1, 1.f / 255.f);
        //cv::subtract(image, cv::Scalar(0.5f, 0.5f, 0.5f), image, cv::noArray(), -1);
        //cv::divide(image, cv::Scalar(0.5f, 0.5f, 0.5f), image, 1, -1);

        // NHWC to NCHW conversion
        // NHWC: For each pixel, its 3 colors are stored together in RGB order.
        // For a 3 channel image, say RGB, pixels of the R channel are stored first, then the G channel and finally the B channel.
        // https://user-images.githubusercontent.com/20233731/85104458-3928a100-b23b-11ea-9e7e-95da726fef92.png
        int offset = dims.d[1] * dims.d[2] * dims.d[3] * batch;
        //int r = 0, g = 0, b = 0;
        for (int i = 0; i < dims.d[1] * dims.d[2] * dims.d[3]; ++i) {
            hostDataBuffer[offset + i] = *(reinterpret_cast<float*>(image.data) + i);  
            }
    }

    // Copy from CPU to GPU
    auto ret = cudaMemcpyAsync(m_inputBuff.deviceBuffer.data(), m_inputBuff.hostBuffer.data(), m_inputBuff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice, m_cudaStream);
    if (ret != 0) {
        return false;
    }

    std::vector<void*> predictionBindings = {m_inputBuff.deviceBuffer.data(), m_outputBuff_detector_logits.deviceBuffer.data(),m_outputBuff_detector.deviceBuffer.data(),m_outputBuff_descriptor_raw.deviceBuffer.data()};//, m_outputBuff_segmentation.deviceBuffer.data()}; //,m_outputBuff_descriptor.deviceBuffer.data() };
    // Run inference.
    bool status = m_context->enqueueV2(predictionBindings.data(), m_cudaStream, nullptr);
    if (!status) {
        return false;
    }
    // Copy the results back to CPU memory
    /*
    ret = cudaMemcpyAsync(m_outputBuff_descriptor.hostBuffer.data(), m_outputBuff_descriptor.deviceBuffer.data(), m_outputBuff_descriptor.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost, m_cudaStream);
    if (ret != 0) {
        std::cout << "Unable to copy descriptor buffer from GPU back to CPU" << std::endl;
        return false;
    }*/
    ret = cudaMemcpyAsync(m_outputBuff_descriptor_raw.hostBuffer.data(), m_outputBuff_descriptor_raw.deviceBuffer.data(), m_outputBuff_descriptor_raw.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost, m_cudaStream);
    if (ret != 0) {
        std::cout << "Unable to copy descriptor_raw buffer from GPU back to CPU" << std::endl;
        return false;
    }
    ret = cudaMemcpyAsync(m_outputBuff_detector_logits.hostBuffer.data(), m_outputBuff_detector_logits.deviceBuffer.data(), m_outputBuff_detector_logits.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost, m_cudaStream);
    if (ret != 0) {
        std::cout << "Unable to copy detector_logits buffer from GPU back to CPU" << std::endl;
        return false;
    }
    ret = cudaMemcpyAsync(m_outputBuff_detector.hostBuffer.data(), m_outputBuff_detector.deviceBuffer.data(), m_outputBuff_detector.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost, m_cudaStream);
    if (ret != 0) {
        std::cout << "Unable to copy detector buffer from GPU back to CPU" << std::endl;
        return false;
    }
    // comment out following if loading SuperPoint model
    /*
    ret = cudaMemcpyAsync(m_outputBuff_segmentation.hostBuffer.data(), m_outputBuff_segmentation.deviceBuffer.data(), m_outputBuff_segmentation.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost, m_cudaStream);
    if (ret != 0) {
        std::cout << "Unable to copy segmentation buffer from GPU back to CPU" << std::endl;
        return false;
    }*/
    ret = cudaStreamSynchronize(m_cudaStream);
    if (ret != 0) {
        std::cout << "Unable to synchronize cuda stream" << std::endl;
        return false;
    }
    // Copy to output 
    for (int batch = 0; batch < batchSize; ++batch) {
        int dims_descriptor_raw[] = { outputDims_descriptor_raw.d[1], outputDims_descriptor_raw.d[2], outputDims_descriptor_raw.d[3] };
        //int dims_descriptor[] = { outputL_descriptor.d[1], outputL_descriptor.d[2], outputL_descriptor.d[3] };
        int dims_detector_logits[] = { outputDims_detector_logits.d[1], outputDims_detector_logits.d[2], outputDims_detector_logits.d[3] };
        int dims_detector[] = { outputDims_detector.d[1], outputDims_detector.d[2] };
        //int dims_segmentation[] = { outputDims_segmentation.d[1], outputDims_segmentation.d[2], outputDims_segmentation.d[3] };

        cv::Mat featureVector_descriptor_raw(3, dims_descriptor_raw, CV_32F);
        //cv::Mat featureVector_descriptor(3, dims_descriptor, CV_32F);
        cv::Mat featureVector_detector_logits(3, dims_detector_logits, CV_32F);
        cv::Mat featureVector_detector(2, dims_detector, CV_32F);
        //cv::Mat featureVector_segmentation(3, dims_segmentation, CV_32F);
        
        memcpy(featureVector_descriptor_raw.data, reinterpret_cast<float*>(m_outputBuff_descriptor_raw.hostBuffer.data()) +
            batch * outputL_descriptor_raw.d[1] * outputL_descriptor_raw.d[2] * outputL_descriptor_raw.d[3] * sizeof(float), outputL_descriptor_raw.d[1] * outputL_descriptor_raw.d[2] * outputL_descriptor_raw.d[3] * sizeof(float));
        featureVectors_descriptor_raw.emplace_back(std::move(featureVector_descriptor_raw));
        /*
        memcpy(featureVector_descriptor.data, reinterpret_cast<float*>(m_outputBuff_descriptor.hostBuffer.data()) +
        batch * outputL_descriptor.d[1] * outputL_descriptor.d[2] * outputL_descriptor.d[3] * sizeof(float), outputL_descriptor.d[1] * outputL_descriptor.d[2] * outputL_descriptor.d[3] * sizeof(float ));
        featureVectors_descriptor.emplace_back(std::move(featureVector_descriptor));
        */
        memcpy(featureVector_detector_logits.data, reinterpret_cast<float*>(m_outputBuff_detector_logits.hostBuffer.data()) +
            batch * outputL_detector_logits.d[1] * outputL_detector_logits.d[2] * outputL_detector_logits.d[3] * sizeof(float), outputL_detector_logits.d[1] * outputL_detector_logits.d[2] * outputL_detector_logits.d[3] * sizeof(float));
        featureVectors_detector_logits.emplace_back(std::move(featureVector_detector_logits));

        memcpy(featureVector_detector.data, reinterpret_cast<float*>(m_outputBuff_detector.hostBuffer.data()) +
            batch * outputL_detector.d[1] * outputL_detector.d[2] * sizeof(float), outputL_detector.d[1] * outputL_detector.d[2] * sizeof(float));
        featureVectors_detector.emplace_back(std::move(featureVector_detector));

        /*memcpy(featureVector_segmentation.data, reinterpret_cast<float*>(m_outputBuff_segmentation.hostBuffer.data()) +
            batch * outputL_segmentation.d[1] * outputL_segmentation.d[2] * outputL_segmentation.d[3] * sizeof(float), outputL_segmentation.d[1] * outputL_segmentation.d[2] * outputL_segmentation.d[3] * sizeof(float));
        featureVectors_segmentation.emplace_back(std::move(featureVector_segmentation));*/
    }
    return true;
}

std::string Engine::serializeEngineOptions(const Options &options) {
    std::string engineName = "trt.engine";

    std::vector<std::string> gpuUUIDs;
    getGPUUUIDs(gpuUUIDs);

    if (static_cast<size_t>(options.deviceIndex) >= gpuUUIDs.size()) {
        throw std::runtime_error("Error, provided device index is out of range!");
    }

    engineName+= "." + gpuUUIDs[options.deviceIndex];

    // Serialize the specified options into the filename
    if (options.FP16) {
        engineName += ".fp16";
    } else {
        engineName += ".fp32";
    }

    engineName += "." + std::to_string(options.maxBatchSize) + ".";
    for (size_t i = 0; i < m_options.optBatchSizes.size(); ++i) {
        engineName += std::to_string(m_options.optBatchSizes[i]);
        if (i != m_options.optBatchSizes.size() - 1) {
            engineName += "_";
        }
    }

    engineName += "." + std::to_string(options.maxWorkspaceSize);

    return engineName;
}

void Engine::getGPUUUIDs(std::vector<std::string>& gpuUUIDs) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device=0; device<numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        char uuid[33];
        for(int b=0; b<16; b++) {
            sprintf(&uuid[b*2], "%02x", (unsigned char)prop.uuid.bytes[b]);
        }

        gpuUUIDs.push_back(std::string(uuid));
        // by comparing uuid against a preset list of valid uuids given by the client (using: nvidia-smi -L) we decide which gpus can be used.
    }
}

}