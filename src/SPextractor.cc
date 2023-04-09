/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. G�mez Rodr�guez, Jos� M.M. Montiel and Juan D. Tard�s, University of Zaragoza.
 * Copyright (C) 2014-2016 Ra�l Mur-Artal, Jos� M.M. Montiel and Juan D. Tard�s, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

#include "SPextractor.h"
#include "SuperPoint.h"
#include "engine.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;
    const Options options;

    const float factorPI = (float)(CV_PI / 180.f);

    static float IC_Angle(const Mat &image, Point2f pt, const vector<int> &u_max)
    {
        int m_01 = 0, m_10 = 0;

        const uchar *center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

        // Treat the center line differently, v=0
        for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
            m_10 += u * center[u];

        // Go line by line in the circuI853lar patch
        int step = (int)image.step1();
        for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = u_max[v];
            for (int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v * step], val_minus = center[u - v * step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        return fastAtan2((float)m_01, (float)m_10);
    }

    SPextractor::SPextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                             float _iniThFAST, float _minThFAST) : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
                                                                   iniThFAST(_iniThFAST), minThFAST(_minThFAST)
    {
        // model = make_shared<SuperPoint>();
        // torch::load(model, "superpoint.pt");    //put code here for semanticKeypoints

        model_engine1 = make_shared<SuperPoint>(options);
        model_engine2 = make_shared<SuperPoint>(options);
        model_engine3 = make_shared<SuperPoint>(options);
        model_engine4 = make_shared<SuperPoint>(options);
     
        // Engine engine_semanticKeypoints(options);
        // TODO: Specify your model here.
        // Must specify a dynamic batch size when exporting the model from onnx.
        const std::string onnxModelpath_semanticKeypoints = "SuperPoint-010423.onnx";
        int level = 0;
        std::cout << "Building engine and loading network for level: "<<level<<std::endl;
        bool succ = model_engine1->build(onnxModelpath_semanticKeypoints, level);
        if (!succ)
        {
            throw std::runtime_error("Unable to build TRT semanticKeypoints engine.");
        }
        succ = model_engine1->loadNetwork();
        if (!succ)
        {
            throw std::runtime_error("Unable to load TRT semanticKeypoints engine.");
        }
        level = level + 1;
        std::cout << "Building engine and loading network for level: "<<level<<std::endl;
        succ = model_engine2->build(onnxModelpath_semanticKeypoints, level);
        if (!succ)
        {
            throw std::runtime_error("Unable to build TRT semanticKeypoints engine.");
        }
        succ = model_engine2->loadNetwork();
        if (!succ)
        {
            throw std::runtime_error("Unable to load TRT semanticKeypoints engine.");
        }
        level = level+1;
        std::cout << "Building engine and loading network for level: "<<level<<std::endl;
        succ = model_engine3->build(onnxModelpath_semanticKeypoints, level);
        if (!succ)
        {
            throw std::runtime_error("Unable to build TRT semanticKeypoints engine.");
        }
        succ = model_engine3->loadNetwork();
        if (!succ)
        {
            throw std::runtime_error("Unable to load TRT semanticKeypoints engine.");
        }
        level = level + 1;
        std::cout << "Building engine and loading network for level: "<<level<<std::endl;
        succ = model_engine4->build(onnxModelpath_semanticKeypoints, level);
        if (!succ)
        {
            throw std::runtime_error("Unable to build TRT semanticKeypoints engine.");
        }
        succ = model_engine4->loadNetwork();
        if (!succ)
        {
            throw std::runtime_error("Unable to load TRT semanticKeypoints engine.");
        }
        if (nlevels !=level+1){
            throw std::runtime_error("Failed to build engine and load all the networks");
        }
        mvScaleFactor.resize(nlevels);
        mvLevelSigma2.resize(nlevels);
        mvScaleFactor[0] = 1.0f;
        mvLevelSigma2[0] = 1.0f;
        for (int i = 1; i < nlevels; i++)
        {
            mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
            mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
        }

        mvInvScaleFactor.resize(nlevels);
        mvInvLevelSigma2.resize(nlevels);
        for (int i = 0; i < nlevels; i++)
        {
            mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
            mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
        }

        mvImagePyramid.resize(nlevels);

        mnFeaturesPerLevel.resize(nlevels);
        float factor = 1.0f / scaleFactor;
        float nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));

        int sumFeatures = 0;
        for (int level = 0; level < nlevels - 1; level++)
        {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

        /*
        const int npoints = 512;
        const Point* pattern0 = (const Point*)bit_pattern_31_;
        std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));*/

        // This is for orientation
        //  pre-compute the end of a row in a circular patch
        umax.resize(HALF_PATCH_SIZE + 1);

        int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
        const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
        for (v = 0; v <= vmax; ++v)
            umax[v] = cvRound(sqrt(hp2 - v * v));

        // Make sure we are symmetric
        for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
        {
            while (umax[v0] == umax[v0 + 1])
                ++v0;
            umax[v] = v0;
            ++v0;
        }
    }

    static void computeOrientation(const Mat &image, vector<KeyPoint> &keypoints, const vector<int> &umax)
    {
        for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                                        keypointEnd = keypoints.end();
             keypoint != keypointEnd; ++keypoint)
        {
            keypoint->angle = IC_Angle(image, keypoint->pt, umax);
        }
    }

    void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
    {
        const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
        const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

        // Define boundaries of childs
        n1.UL = UL;
        n1.UR = cv::Point2i(UL.x + halfX, UL.y);
        n1.BL = cv::Point2i(UL.x, UL.y + halfY);
        n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
        n1.vKeys.reserve(vKeys.size());

        n2.UL = n1.UR;
        n2.UR = UR;
        n2.BL = n1.BR;
        n2.BR = cv::Point2i(UR.x, UL.y + halfY);
        n2.vKeys.reserve(vKeys.size());

        n3.UL = n1.BL;
        n3.UR = n1.BR;
        n3.BL = BL;
        n3.BR = cv::Point2i(n1.BR.x, BL.y);
        n3.vKeys.reserve(vKeys.size());

        n4.UL = n3.UR;
        n4.UR = n2.BR;
        n4.BL = n3.BR;
        n4.BR = BR;
        n4.vKeys.reserve(vKeys.size());

        // Associate points to childs
        for (size_t i = 0; i < vKeys.size(); i++)
        {
            const cv::KeyPoint &kp = vKeys[i];
            if (kp.pt.x < n1.UR.x)
            {
                if (kp.pt.y < n1.BR.y)
                    n1.vKeys.push_back(kp);
                else
                    n3.vKeys.push_back(kp);
            }
            else if (kp.pt.y < n1.BR.y)
                n2.vKeys.push_back(kp);
            else
                n4.vKeys.push_back(kp);
        }

        if (n1.vKeys.size() == 1)
            n1.bNoMore = true;
        if (n2.vKeys.size() == 1)
            n2.bNoMore = true;
        if (n3.vKeys.size() == 1)
            n3.bNoMore = true;
        if (n4.vKeys.size() == 1)
            n4.bNoMore = true;
    }

    static bool compareNodes(pair<int, ExtractorNode *> &e1, pair<int, ExtractorNode *> &e2)
    {
        if (e1.first < e2.first)
        {
            return true;
        }
        else if (e1.first > e2.first)
        {
            return false;
        }
        else
        {
            if (e1.second->UL.x < e2.second->UL.x)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }

    vector<cv::KeyPoint> SPextractor::DistributeOctTree(const vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
                                                        const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
    {
        // Compute how many initial nodes
        const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));

        const float hX = static_cast<float>(maxX - minX) / nIni;

        list<ExtractorNode> lNodes;

        vector<ExtractorNode *> vpIniNodes;
        vpIniNodes.resize(nIni);

        for (int i = 0; i < nIni; i++)
        {
            ExtractorNode ni;
            ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
            ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
            ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
            ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
            ni.vKeys.reserve(vToDistributeKeys.size());

            lNodes.push_back(ni);
            vpIniNodes[i] = &lNodes.back();
        }

        // Associate points to childs
        for (size_t i = 0; i < vToDistributeKeys.size(); i++)
        {
            const cv::KeyPoint &kp = vToDistributeKeys[i];
            vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
        }

        list<ExtractorNode>::iterator lit = lNodes.begin();

        while (lit != lNodes.end())
        {
            if (lit->vKeys.size() == 1)
            {
                lit->bNoMore = true;
                lit++;
            }
            else if (lit->vKeys.empty())
                lit = lNodes.erase(lit);
            else
                lit++;
        }

        bool bFinish = false;

        int iteration = 0;

        vector<pair<int, ExtractorNode *>> vSizeAndPointerToNode;
        vSizeAndPointerToNode.reserve(lNodes.size() * 4);

        while (!bFinish)
        {
            iteration++;

            int prevSize = lNodes.size();

            lit = lNodes.begin();

            int nToExpand = 0;

            vSizeAndPointerToNode.clear();

            while (lit != lNodes.end())
            {
                if (lit->bNoMore)
                {
                    // If node only contains one point do not subdivide and continue
                    lit++;
                    continue;
                }
                else
                {
                    // If more than one point, subdivide
                    ExtractorNode n1, n2, n3, n4;
                    lit->DivideNode(n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.vKeys.size() > 0)
                    {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0)
                    {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0)
                    {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0)
                    {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lit = lNodes.erase(lit);
                    continue;
                }
            }

            // Finish if there are more nodes than required features
            // or all nodes contain just one point
            if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
            {
                bFinish = true;
            }
            else if (((int)lNodes.size() + nToExpand * 3) > N)
            {

                while (!bFinish)
                {

                    prevSize = lNodes.size();

                    vector<pair<int, ExtractorNode *>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                    vSizeAndPointerToNode.clear();

                    sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end(), compareNodes);
                    for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
                    {
                        ExtractorNode n1, n2, n3, n4;
                        vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

                        // Add childs if they contain points
                        if (n1.vKeys.size() > 0)
                        {
                            lNodes.push_front(n1);
                            if (n1.vKeys.size() > 1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if (n2.vKeys.size() > 0)
                        {
                            lNodes.push_front(n2);
                            if (n2.vKeys.size() > 1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if (n3.vKeys.size() > 0)
                        {
                            lNodes.push_front(n3);
                            if (n3.vKeys.size() > 1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if (n4.vKeys.size() > 0)
                        {
                            lNodes.push_front(n4);
                            if (n4.vKeys.size() > 1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }

                        lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                        if ((int)lNodes.size() >= N)
                            break;
                    }

                    if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
                        bFinish = true;
                }
            }
        }

        // Retain the best point in each node
        vector<cv::KeyPoint> vResultKeys;
        vResultKeys.reserve(nfeatures);
        for (list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++)
        {
            vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
            cv::KeyPoint *pKP = &vNodeKeys[0];
            float maxResponse = pKP->response;

            for (size_t k = 1; k < vNodeKeys.size(); k++)
            {
                if (vNodeKeys[k].response > maxResponse)
                {
                    pKP = &vNodeKeys[k];
                    maxResponse = vNodeKeys[k].response;
                }
            }

            vResultKeys.push_back(*pKP);
        }

        return vResultKeys;
    }

    void SPextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint>> &allKeypoints, cv::Mat &_desc)
    {
        allKeypoints.resize(nlevels);

        vector<cv::Mat> vDesc;

        const float W = 35;
        SPDetector detector1(model_engine1);
        SPDetector detector2(model_engine2);
        SPDetector detector3(model_engine3);
        SPDetector detector4(model_engine4);
        for (int level = 0; level < nlevels; ++level)
        {
            // SPDetector detector(model);  // semanticKeypoints here
            // detector.detect(mvImagePyramid[level], false);   // semanticKeypoints here
            switch (level) {
                case 0:
                {
                    detector1.detect(mvImagePyramid[level], false, level);
                    break;
                }
                case 1:
                {
                    detector2.detect(mvImagePyramid[level], false, level);
                    break;
                }
                case 2:
                {
                    detector3.detect(mvImagePyramid[level], false, level);
                    break;
                }
                case 3:
                {
                    detector4.detect(mvImagePyramid[level], false, level);
                    break;
                }
                    }
                    
            const int minBorderX = EDGE_THRESHOLD - 3;
            const int minBorderY = minBorderX;
            const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
            const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

            vector<cv::KeyPoint> vToDistributeKeys;
            vToDistributeKeys.reserve(nfeatures * 10);
            const float width = (maxBorderX - minBorderX);
            const float height = (maxBorderY - minBorderY);

            const int nCols = width / W;
            const int nRows = height / W;
            const int wCell = ceil(width / nCols);
            const int hCell = ceil(height / nRows);

            for (int i = 0; i < nRows; i++)
            {
                const float iniY = minBorderY + i * hCell;
                float maxY = iniY + hCell + 6;

                if (iniY >= maxBorderY - 3)
                    continue;
                if (maxY > maxBorderY)
                    maxY = maxBorderY;

                for (int j = 0; j < nCols; j++)
                {
                    const float iniX = minBorderX + j * wCell;
                    float maxX = iniX + wCell + 6;
                    if (iniX >= maxBorderX - 6)
                        continue;
                    if (maxX > maxBorderX)
                        maxX = maxBorderX;

                    vector<cv::KeyPoint> vKeysCell;

                    // FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                    //     vKeysCell, iniThFAST, true);


                    //detector.getKeyPoints(iniThFAST, iniX, maxX, iniY, maxY, vKeysCell, true);
                    switch (level) {
                        case 0:
                        {
                            detector1.getKeyPoints(iniThFAST, iniX, maxX, iniY, maxY, vKeysCell, true);
                            break;
                            }
                        case 1:
                        {
                            detector2.getKeyPoints(iniThFAST, iniX, maxX, iniY, maxY, vKeysCell, true);
                            break;
                            }
                        case 2:
                        {
                            detector3.getKeyPoints(iniThFAST, iniX, maxX, iniY, maxY, vKeysCell, true);
                            break;
                        }
                        case 3:
                        {
                            detector4.getKeyPoints(iniThFAST, iniX, maxX, iniY, maxY, vKeysCell, true);
                            break;
                        }
                    }

                    // semantic keypoints above

                    /*if(bRight && j <= 13){
                        FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                             vKeysCell,10,true);
                    }
                    else if(!bRight && j >= 16){
                        FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                             vKeysCell,10,true);
                    }
                    else{
                        FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                             vKeysCell,iniThFAST,true);
                    }*/

                    if (vKeysCell.empty())
                    {
                        // FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                        //     vKeysCell, minThFAST, true);

                        //detector.getKeyPoints(minThFAST, iniX, maxX, iniY, maxY, vKeysCell, true);
                        switch (level) {
                        case 0:
                        {
                            detector1.getKeyPoints(minThFAST, iniX, maxX, iniY, maxY, vKeysCell, true);
                            break;
                            }
                        case 1:
                        {
                            detector2.getKeyPoints(minThFAST, iniX, maxX, iniY, maxY, vKeysCell, true);
                            break;
                            }
                        case 2:
                        {
                            detector3.getKeyPoints(minThFAST, iniX, maxX, iniY, maxY, vKeysCell, true);
                            break;
                        }
                        case 3:
                        {
                            detector4.getKeyPoints(minThFAST, iniX, maxX, iniY, maxY, vKeysCell, true);
                            break;
                        }
                    }

                        // semantic keypoints above

                        /*if(bRight && j <= 13){
                            FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                                 vKeysCell,5,true);
                        }
                        else if(!bRight && j >= 16){
                            FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                                 vKeysCell,5,true);
                        }
                        else{
                            FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                                 vKeysCell,minThFAST,true);
                        }*/
                    }

                    if (!vKeysCell.empty())
                    {
                        for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++)
                        {
                            (*vit).pt.x += j * wCell;
                            (*vit).pt.y += i * hCell;
                            vToDistributeKeys.push_back(*vit);
                        }
                    }
                }
            }

            vector<KeyPoint> &keypoints = allKeypoints[level];
            keypoints.reserve(nfeatures);

            keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                          minBorderY, maxBorderY, mnFeaturesPerLevel[level], level);

            const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];

            // Add border to coordinates and scale information
            const int nkps = keypoints.size();
            for (int i = 0; i < nkps; i++)
            {
                keypoints[i].pt.x += minBorderX;
                keypoints[i].pt.y += minBorderY;
                keypoints[i].octave = level;
                keypoints[i].size = scaledPatchSize;
            }
            // semanticKeypoints here
            cv::Mat desc;

            //detector.computeDescriptors(keypoints, desc);
            switch (level) {
                        case 0:
                        {
                            detector1.computeDescriptors(keypoints, desc);
                            break;
                            }
                        case 1:
                        {
                            detector2.computeDescriptors(keypoints, desc);
                            break;
                            }
                        case 2:
                        {
                            detector3.computeDescriptors(keypoints, desc);
                            break;
                        }
                        case 3:
                        {
                            detector4.computeDescriptors(keypoints, desc);
                            break;
                        }
                    }
            //std::cout <<"desc: "<< desc.size()<<std::endl;
            vDesc.push_back(desc);
        }
        // semanticKeypoints here
        
        cv::vconcat(vDesc, _desc);

        // compute orientations
        for (int level = 0; level < nlevels; ++level)
            computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
    }

    void SPextractor::operator()(InputArray _image, InputArray _mask, vector<KeyPoint> &_keypoints,
                                 OutputArray _descriptors)
    {
        if (_image.empty())
            return;

        Mat image = _image.getMat();
        assert(image.type() == CV_8UC1);

        Mat descriptors;

        // Pre-compute the scale pyramid
        ComputePyramid(image);
        
           
        vector<vector<KeyPoint>> allKeypoints;
        ComputeKeyPointsOctTree(allKeypoints, descriptors);
        cout << descriptors.rows << endl;

        int nkeypoints = 0;
        for (int level = 0; level < nlevels; ++level)
            nkeypoints += (int)allKeypoints[level].size();
        if (nkeypoints == 0)
            _descriptors.release();
        else
        {
            _descriptors.create(nkeypoints, 256, CV_32F);
            descriptors.copyTo(_descriptors.getMat());
        }

        _keypoints.clear();
        _keypoints.reserve(nkeypoints);

        int offset = 0;
        for (int level = 0; level < nlevels; ++level)
        {
            vector<KeyPoint> &keypoints = allKeypoints[level];
            int nkeypointsLevel = (int)keypoints.size();

            if (nkeypointsLevel == 0)
                continue;

            // // preprocess the resized image
            // Mat workingMat = mvImagePyramid[level].clone();
            // GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

            // // Compute the descriptors
            // Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
            // computeDescriptors(workingMat, keypoints, desc, pattern);

            // offset += nkeypointsLevel;

            // Scale keypoint coordinates
            if (level != 0)
            {
                float scale = mvScaleFactor[level]; // getScale(level, firstLevel, scaleFactor);
                for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                                                keypointEnd = keypoints.end();
                     keypoint != keypointEnd; ++keypoint)
                    keypoint->pt *= scale;
            }
            // And add the keypoints to the output
            _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
        }
    }

    /*
        int SPextractor::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
            OutputArray _descriptors, std::vector<int>& vLappingArea)
        {
            //cout << "[ORBextractor]: Max Features: " << nfeatures << endl;
            if (_image.empty())
                return -1;

            Mat image = _image.getMat();
            assert(image.type() == CV_8UC1);

            cv::Mat descriptors;

            // Pre-compute the scale pyramid
            ComputePyramid(image);

            vector < vector<KeyPoint> > allKeypoints;
            ComputeKeyPointsOctTree(allKeypoints, descriptors);
            std::cout << descriptors.rows << std::endl;
            //ComputeKeyPointsOld(allKeypoints);

            int nkeypoints = 0;
            for (int level = 0; level < nlevels; ++level)
                nkeypoints += (int)allKeypoints[level].size();
            if (nkeypoints == 0)
                _descriptors.release();
            else
            {
                _descriptors.create(nkeypoints, 256, CV_32F);
                //descriptors = _descriptors.getMat();
                descriptors.copyTo(_descriptors.getMat());

            }

            //_keypoints.clear();
            //_keypoints.reserve(nkeypoints);
            _keypoints = vector<cv::KeyPoint>(nkeypoints);

            int offset = 0;
            //Modified for speeding up stereo fisheye matching
            int monoIndex = 0, stereoIndex = nkeypoints - 1;
            for (int level = 0; level < nlevels; ++level)
            {
                vector<KeyPoint>& keypoints = allKeypoints[level];
                int nkeypointsLevel = (int)keypoints.size();

                if (nkeypointsLevel == 0)
                    continue;

                // preprocess the resized image
                Mat workingMat = mvImagePyramid[level].clone();
                GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

                // Compute the descriptors
                //Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
                Mat desc = cv::Mat(nkeypointsLevel, 256, CV_32F);

                offset += nkeypointsLevel;


                float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
                int i = 0;
                for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                    keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint) {

                    // Scale keypoint coordinates
                    if (level != 0) {
                        keypoint->pt *= scale;
                    }

                    if (keypoint->pt.x >= vLappingArea[0] && keypoint->pt.x <= vLappingArea[1]) {
                        _keypoints.at(stereoIndex) = (*keypoint);
                        //desc.row(i).copyTo(descriptors.row(stereoIndex));
                        stereoIndex--;
                    }
                    else {
                        _keypoints.at(monoIndex) = (*keypoint);
                        //desc.row(i).copyTo(descriptors.row(monoIndex));
                        monoIndex++;
                    }
                    i++;
                }
                // And add the keypoints to the output
                //_keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
            }
            //cout << "[ORBextractor]: extracted " << _keypoints.size() << " KeyPoints" << endl;
            return monoIndex;
        }*/

    void SPextractor::ComputePyramid(cv::Mat image)
    {
        for (int level = 0; level < nlevels; ++level)
        {
            float scale = mvInvScaleFactor[level];
            Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
            Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
            Mat temp(wholeSize, image.type()), masktemp;
            mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

            // Compute the resized image
            if (level != 0)
            {
                resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

                copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101 + BORDER_ISOLATED);
            }
            else
            {
                copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101);
            }
        }
    }

} // namespace ORB_SLAM
