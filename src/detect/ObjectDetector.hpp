#ifndef OBJECT_DETECTOR_HPP
#define OBJECT_DETECTOR_HPP

#include <opencv2/core/types_c.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

namespace detect
{
/**
 * @brief The ObjectDetector class is used to detect objects using
 * the YOLO implementation from OpenCV
 */
class ObjectDetector
{
private:
    ObjectDetector();

    // Get the names of the output layers
    vector<String> getOutputsNames(const Net& net);

public:
    static ObjectDetector& getInstance();

    std::vector<Mat> getOutputLayers(cv::Mat& frame);

    // The function getPerfProfile returns the overall time for inference(t)
    double getPerfProfile();

    // Remove the bounding boxes with low confidence using non-maxima suppression
    void processFrame(Mat& frame);

    // Draw the predicted bounding box
    void drawPred(int classId, float conf, int left, int top, int right,
      int bottom, Mat& frame);


private:
    // Initialize the parameters
    int inpWidth = 416;  // Width of network's input image
    int inpHeight = 416; // Height of network's input image
    float confThreshold = 0.1; // Confidence threshold
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold

    // Give the configuration and weight files for the model
    String modelConfiguration = "detector.cfg";
    String modelWeights = "detector.weights";

    vector<string> classes;
    Net net;
};
} // namespace detect

#endif // OBJECT_DETECTOR_HPP
