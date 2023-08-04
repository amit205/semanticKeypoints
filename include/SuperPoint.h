#ifndef SUPERPOINT_H
#define SUPERPOINT_H


#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "engine.h"
#include <vector>

#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif


namespace ORB_SLAM2
{
    struct SuperPoint : Engine {
        SuperPoint(const Options& options):Engine(options){}
    };

    //cv::Mat SPdetect(std::shared_ptr<SuperPoint> model, cv::Mat img, std::vector<cv::KeyPoint>& keypoints, double threshold, bool nms, bool cuda);
    //cv::Mat SPdetect(std::shared_ptr<SuperPoint> model_engine, cv::Mat img, std::vector<cv::KeyPoint>& keypoints, double threshold, bool nms, bool cuda);
    //torch::Tensor NMS(torch::Tensor kpts);
    void channelsLastToChannelsFirst(cv::Mat& input, cv::Mat& output);

    class SPDetector {
    public:
        SPDetector(std::shared_ptr<SuperPoint> _model_engine);
        void detect(cv::Mat& image, bool cuda, int level);
        void getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint>& keypoints, bool nms);
        void computeDescriptors(const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

    private:
        //std::shared_ptr<SuperPoint> model;
        
        std::shared_ptr<SuperPoint> model_engine1;
        std::shared_ptr<SuperPoint> model_engine2;
        std::shared_ptr<SuperPoint> model_engine3;
        std::shared_ptr<SuperPoint> model_engine4;
        std::shared_ptr<SuperPoint> model_engine5;
        std::shared_ptr<SuperPoint> model_engine6;
        std::shared_ptr<SuperPoint> model_engine7;
        std::shared_ptr<SuperPoint> model_engine8;
        torch::Tensor mProb;
        torch::Tensor mDesc;
        float image_scale_width;
        float image_scale_height;
        //cv::Mat mProb;
        //std::vector<cv::Mat> mDesc;
    };
}  // ORB_SLAM

#endif