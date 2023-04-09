/**
 * File: FORB.h
 * Date: June 2012
 * Author: Dorian Galvez-Lopez
 * Description: functions for ORB descriptors
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_F_ORB__
#define __D_T_F_ORB__

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>

#include "FClass.h"

namespace DBoW2 {

/// Functions to manipulate ORB descriptors
class FORB: protected FClass
{
public:

  /// Descriptor type
  typedef cv::Mat TDescriptor; // CV_8U
  /// Pointer to a single descriptor
  typedef const TDescriptor *pDescriptor;
  /// Descriptor length (in bytes)
  static const int L;

  /**
   * Calculates the mean value of a set of descriptors
   * @param descriptors
   * @param mean mean descriptor
   */
  static void meanValue(const std::vector<pDescriptor> &descriptors,
    TDescriptor &mean);

  /**
   * Calculates the distance between two descriptors
   * @param a
   * @param b
   * @return distance
   */
  static double distance(const TDescriptor &a, const TDescriptor &b);

  /**
   * Returns a string version of the descriptor
   * @param a descriptor
   * @return string version
   */
  static std::string toString(const TDescriptor &a);

  /**
   * Returns a descriptor from a string
   * @param a descriptor
   * @param s string version
   */
  static void fromString(TDescriptor &a, const std::string &s);

  /**
   * Returns a mat with the descriptors in float format
   * @param descriptors
   * @param mat (out) NxL 32F matrix
   */
  static void toMat32F(const std::vector<TDescriptor> &descriptors,
    cv::Mat &mat);

  static void toMat8U(const std::vector<TDescriptor> &descriptors,
    cv::Mat &mat);
  void toStream(const cv::Mat &m,std::ostream &str);
  void fromStream(cv::Mat &m,std::istream &str);
public:
  /**Returns the number of bytes of the descriptor
   * used for binary descriptors only*/
  static size_t getDescSizeBytes(const cv::Mat & d){return d.cols* d.elemSize();}

};

} // namespace DBoW2

#endif

