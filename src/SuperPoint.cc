#include "SuperPoint.h"

namespace ORB_SLAM2
{
    const Options options;
    // options.optBatchSizes = { 2, 4, 6 };
    SuperPoint model_engine1(options);
    SuperPoint model_engine2(options);
    SuperPoint model_engine3(options);
    SuperPoint model_engine4(options);
    SuperPoint model_engine5(options);
    SuperPoint model_engine6(options);
    SuperPoint model_engine7(options);
    SuperPoint model_engine8(options);

    void NMS(cv::Mat det, cv::Mat conf, cv::Mat desc, std::vector<cv::KeyPoint> &pts, cv::Mat &descriptors,
             int border, int dist_thresh, int img_width, int img_height);
    void NMS2(std::vector<cv::KeyPoint> det, cv::Mat conf, std::vector<cv::KeyPoint> &pts,
              int border, int dist_thresh, int img_width, int img_height);

    SPDetector::SPDetector(std::shared_ptr<SuperPoint> _model_engine)
        : model_engine1(_model_engine),
          model_engine2(_model_engine),
          model_engine3(_model_engine),
          model_engine4(_model_engine),
          model_engine5(_model_engine),
          model_engine6(_model_engine),
          model_engine7(_model_engine),
          model_engine8(_model_engine)
    {
    }

    void SPDetector::detect(cv::Mat &img, bool cuda, int level)

    {

        std::vector<cv::Mat> images;
        std::vector<cv::Mat> featureVectors_descriptor_raw;
        std::vector<cv::Mat> featureVectors_detector_logits;
        std::vector<cv::Mat> featureVectors_detector;
        // comment out following if loading SuperPoint model
        // std::vector<cv::Mat> featureVectors_segmentation;
        featureVectors_descriptor_raw.clear();
        featureVectors_detector_logits.clear();
        featureVectors_detector.clear();
        // comment out following if loading SuperPoint model
        // featureVectors_segmentation.clear();
        images.clear();
        // std::cout << "Image size: " << img.size() << std::endl;
        cv::Mat imgConvert;
        img.convertTo(imgConvert, CV_32F, 1.f / 255.f);
        int width = img.cols;
        int height = img.rows;
        int width_c = width / 8;
        int height_c = height / 8;
        cv::Mat imgResized;
        cv::resize(imgConvert, imgResized, cv::Size(width_c * 8, height_c * 8), cv::INTER_LINEAR);
        // std::cout << "Resized Image size: " << imgResized.size() << std::endl;
        //  cv::imshow("Input",imgResized);
        //  cv::waitKey(1);
        images.push_back(imgResized);
        bool succ;

        switch (level)
        {
        case 0:
        {
            succ = model_engine1->runInference(images, featureVectors_descriptor_raw, featureVectors_detector_logits, featureVectors_detector); //, featureVectors_segmentation);
            if (!succ)
            {
                throw std::runtime_error("Unable to run semanticKeypoints inference.");
            }
            break;
        }
        case 1:
        {
            succ = model_engine2->runInference(images, featureVectors_descriptor_raw, featureVectors_detector_logits, featureVectors_detector); //, featureVectors_segmentation);
            if (!succ)
            {
                throw std::runtime_error("Unable to run semanticKeypoints inference.");
            }
            break;
        }
        case 2:
        {
            succ = model_engine3->runInference(images, featureVectors_descriptor_raw, featureVectors_detector_logits, featureVectors_detector); //, featureVectors_segmentation);
            if (!succ)
            {
                throw std::runtime_error("Unable to run semanticKeypoints inference.");
            }
            break;
        }
        case 3:
        {
            succ = model_engine4->runInference(images, featureVectors_descriptor_raw, featureVectors_detector_logits, featureVectors_detector); //, featureVectors_segmentation);
            if (!succ)
            {
                throw std::runtime_error("Unable to run semanticKeypoints inference.");
            }
            break;
        }
        case 4:
        {
            succ = model_engine5->runInference(images, featureVectors_descriptor_raw, featureVectors_detector_logits, featureVectors_detector); //, featureVectors_segmentation);
            if (!succ)
            {
                throw std::runtime_error("Unable to run semanticKeypoints inference.");
            }
            break;
        }
        case 5:
        {
            succ = model_engine6->runInference(images, featureVectors_descriptor_raw, featureVectors_detector_logits, featureVectors_detector); //, featureVectors_segmentation);
            if (!succ)
            {
                throw std::runtime_error("Unable to run semanticKeypoints inference.");
            }
            break;
        }
        case 6:
        {
            succ = model_engine7->runInference(images, featureVectors_descriptor_raw, featureVectors_detector_logits, featureVectors_detector); //, featureVectors_segmentation);
            if (!succ)
            {
                throw std::runtime_error("Unable to run semanticKeypoints inference.");
            }
            break;
        }
        case 7:
        {
            succ = model_engine8->runInference(images, featureVectors_descriptor_raw, featureVectors_detector_logits, featureVectors_detector); //, featureVectors_segmentation);
            if (!succ)
            {
                throw std::runtime_error("Unable to run semanticKeypoints inference.");
            }
            break;
        }
        }

        bool use_cuda = cuda && torch::cuda::is_available();
        torch::DeviceType device_type;
        if (use_cuda)
            device_type = torch::kCUDA;
        else
            device_type = torch::kCPU;
        torch::Device device(device_type);
        // comment out following if loading SuperPoint model
        // cv::Mat segImg(2, featureVectors_segmentation[0].size, CV_32FC1, featureVectors_segmentation[0].data);
        cv::Mat descriptorImg_raw(3, featureVectors_descriptor_raw[0].size, CV_32F, featureVectors_descriptor_raw[0].data);
        cv::Mat detectorLogits(3, featureVectors_detector_logits[0].size, CV_32F, featureVectors_detector_logits[0].data);
        cv::Mat detectorImg(2, featureVectors_detector[0].size, CV_32FC1, featureVectors_detector[0].data);
        // std::cout << "Detector output size: "<<detectorImg.size() << std::endl;
        cv::resize(detectorImg, detectorImg, cv::Size(width, height), cv::INTER_LINEAR);

        mProb = torch::zeros({detectorImg.size[0], detectorImg.size[1]}, torch::kF32).to(device);
        mDesc = torch::zeros({descriptorImg_raw.size[0], descriptorImg_raw.size[1], descriptorImg_raw.size[2]}, torch::kF32).to(device);

        mProb = mProb.set_requires_grad(false);
        mDesc = mDesc.set_requires_grad(false);
        std::memcpy(mProb.data_ptr(), detectorImg.data, sizeof(float) * mProb.numel());
        std::memcpy(mDesc.data_ptr(), descriptorImg_raw.data, sizeof(float) * mDesc.numel());
        mDesc = mDesc.unsqueeze(0);
        mDesc = mDesc.permute({0, 3, 1, 2});

        auto dn = torch::norm(mDesc, 2, 1);
        mDesc = mDesc.div(torch::unsqueeze(dn, 1));
    }

    void SPDetector::getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms)
    {
        auto prob = mProb.slice(0, iniY, maxY).slice(1, iniX, maxX); // [h, w]
        auto kpts = (prob > threshold);
        kpts = torch::nonzero(kpts); // [n_keypoints, 2]  (y, x)
        std::vector<cv::KeyPoint> keypoints_no_nms;
        for (int i = 0; i < kpts.size(0); i++)
        {
            float response = prob[kpts[i][0]][kpts[i][1]].item<float>();
            keypoints_no_nms.push_back(cv::KeyPoint(kpts[i][1].item<float>(), kpts[i][0].item<float>(), 8, -1, response));
        }

        if (nms)
        {
            cv::Mat conf(keypoints_no_nms.size(), 1, CV_32F);
            for (size_t i = 0; i < keypoints_no_nms.size(); i++)
            {
                int x = keypoints_no_nms[i].pt.x;
                int y = keypoints_no_nms[i].pt.y;
                conf.at<float>(i, 0) = prob[y][x].item<float>();
            }

            // cv::Mat descriptors;

            int border = 0;
            int dist_thresh = 4;
            int height = maxY - iniY;
            int width = maxX - iniX;

            NMS2(keypoints_no_nms, conf, keypoints, border, dist_thresh, width, height);
        }
        else
        {
            keypoints = keypoints_no_nms;
        }
    }

    void SPDetector::computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
    {
        cv::Mat kpt_mat(keypoints.size(), 2, CV_32F); // [n_keypoints, 2]  (y, x)
        for (size_t i = 0; i < keypoints.size(); i++)
        {
            kpt_mat.at<float>(i, 0) = (float)keypoints[i].pt.y;
            kpt_mat.at<float>(i, 1) = (float)keypoints[i].pt.x;
        }
        auto fkpts = torch::from_blob(kpt_mat.data, {keypoints.size(), 2}, torch::kFloat);
        // std::cout << "Keypoints: " << keypoints.size() << std::endl;
        auto grid = torch::zeros({1, 1, fkpts.size(0), 2});                         // [1, 1, n_keypoints, 2]
        grid[0][0].slice(1, 0, 1) = 2.0 * fkpts.slice(1, 1, 2) / mProb.size(1) - 1; // x
        grid[0][0].slice(1, 1, 2) = 2.0 * fkpts.slice(1, 0, 1) / mProb.size(0) - 1; // y
        auto desc = torch::grid_sampler(mDesc, grid, 0, 0, false);                  // [1, 256, 1, n_keypoints]
        desc = desc.squeeze(0).squeeze(1);                                          // [256, n_keypoints]
        // normalize to 1
        auto dn = torch::norm(desc, 2, 1);
        desc = desc.div(torch::unsqueeze(dn, 1));
        desc = desc.transpose(0, 1).contiguous(); // [n_keypoints, 256]
        // desc = desc.to(torch::kCPU);

        cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data_ptr<float>());
        // std::cout << "desc_mat: " << desc_mat.size() << std::endl;
        descriptors = desc_mat.clone();
    }

    void NMS2(std::vector<cv::KeyPoint> det, cv::Mat conf, std::vector<cv::KeyPoint> &pts,
              int border, int dist_thresh, int img_width, int img_height)
    {

        std::vector<cv::Point2f> pts_raw;

        for (int i = 0; i < det.size(); i++)
        {

            int u = (int)det[i].pt.x;
            int v = (int)det[i].pt.y;

            pts_raw.push_back(cv::Point2f(u, v));
        }

        cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
        cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

        cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

        grid.setTo(0);
        inds.setTo(0);
        confidence.setTo(0);

        for (int i = 0; i < pts_raw.size(); i++)
        {
            int uu = (int)pts_raw[i].x;
            int vv = (int)pts_raw[i].y;

            grid.at<char>(vv, uu) = 1;
            inds.at<unsigned short>(vv, uu) = i;

            confidence.at<float>(vv, uu) = conf.at<float>(i, 0);
        }

        cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);

        for (int i = 0; i < pts_raw.size(); i++)
        {
            int uu = (int)pts_raw[i].x + dist_thresh;
            int vv = (int)pts_raw[i].y + dist_thresh;

            if (grid.at<char>(vv, uu) != 1)
                continue;

            for (int k = -dist_thresh; k < (dist_thresh + 1); k++)
                for (int j = -dist_thresh; j < (dist_thresh + 1); j++)
                {
                    if (j == 0 && k == 0)
                        continue;

                    if (confidence.at<float>(vv + k, uu + j) < confidence.at<float>(vv, uu))
                        grid.at<char>(vv + k, uu + j) = 0;
                }
            grid.at<char>(vv, uu) = 2;
        }

        size_t valid_cnt = 0;
        std::vector<int> select_indice;

        for (int v = 0; v < (img_height + dist_thresh); v++)
        {
            for (int u = 0; u < (img_width + dist_thresh); u++)
            {
                if (u - dist_thresh >= (img_width - border) || u - dist_thresh < border || v - dist_thresh >= (img_height - border) || v - dist_thresh < border)
                    continue;

                if (grid.at<char>(v, u) == 2)
                {
                    int select_ind = (int)inds.at<unsigned short>(v - dist_thresh, u - dist_thresh);
                    cv::Point2f p = pts_raw[select_ind];
                    float response = conf.at<float>(select_ind, 0);
                    pts.push_back(cv::KeyPoint(p, 8.0f, -1, response));

                    select_indice.push_back(select_ind);
                    valid_cnt++;
                }
            }
        }

        // descriptors.create(select_indice.size(), 256, CV_32F);

        // for (int i=0; i<select_indice.size(); i++)
        // {
        //     for (int j=0; j < 256; j++)
        //     {
        //         descriptors.at<float>(i, j) = desc.at<float>(select_indice[i], j);
        //     }
        // }
    }

    void NMS(cv::Mat det, cv::Mat conf, cv::Mat desc, std::vector<cv::KeyPoint> &pts, cv::Mat &descriptors,
             int border, int dist_thresh, int img_width, int img_height)
    {

        std::vector<cv::Point2f> pts_raw;

        for (int i = 0; i < det.rows; i++)
        {

            int u = (int)det.at<float>(i, 0);
            int v = (int)det.at<float>(i, 1);
            // float conf = det.at<float>(i, 2);

            pts_raw.push_back(cv::Point2f(u, v));
        }

        cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
        cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

        cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

        grid.setTo(0);
        inds.setTo(0);
        confidence.setTo(0);

        for (int i = 0; i < pts_raw.size(); i++)
        {
            int uu = (int)pts_raw[i].x;
            int vv = (int)pts_raw[i].y;

            grid.at<char>(vv, uu) = 1;
            inds.at<unsigned short>(vv, uu) = i;

            confidence.at<float>(vv, uu) = conf.at<float>(i, 0);
        }

        cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);

        for (int i = 0; i < pts_raw.size(); i++)
        {
            int uu = (int)pts_raw[i].x + dist_thresh;
            int vv = (int)pts_raw[i].y + dist_thresh;

            if (grid.at<char>(vv, uu) != 1)
                continue;

            for (int k = -dist_thresh; k < (dist_thresh + 1); k++)
                for (int j = -dist_thresh; j < (dist_thresh + 1); j++)
                {
                    if (j == 0 && k == 0)
                        continue;

                    if (conf.at<float>(vv + k, uu + j) < conf.at<float>(vv, uu))
                        grid.at<char>(vv + k, uu + j) = 0;
                }
            grid.at<char>(vv, uu) = 2;
        }

        size_t valid_cnt = 0;
        std::vector<int> select_indice;

        for (int v = 0; v < (img_height + dist_thresh); v++)
        {
            for (int u = 0; u < (img_width + dist_thresh); u++)
            {
                if (u - dist_thresh >= (img_width - border) || u - dist_thresh < border || v - dist_thresh >= (img_height - border) || v - dist_thresh < border)
                    continue;

                if (grid.at<char>(v, u) == 2)
                {
                    int select_ind = (int)inds.at<unsigned short>(v - dist_thresh, u - dist_thresh);
                    pts.push_back(cv::KeyPoint(pts_raw[select_ind], 1.0f));

                    select_indice.push_back(select_ind);
                    valid_cnt++;
                }
            }
        }

        descriptors.create(select_indice.size(), 256, CV_32F);

        for (int i = 0; i < select_indice.size(); i++)
        {
            for (int j = 0; j < 256; j++)
            {
                descriptors.at<float>(i, j) = desc.at<float>(select_indice[i], j);
            }
        }
    }

} // namespace ORB_SLAM
