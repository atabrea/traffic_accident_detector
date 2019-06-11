#include <fstream>
#include <sstream>
#include <iostream>

#include "detect/ObjectDetector.hpp"
#include "detect/ObjectTracker.hpp"

const char* keys =
"{help h usage ? | | Usage example }"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
;
using namespace cv;
using namespace dnn;
using namespace std;


int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    detect::ObjectDetector& objDetector = detect::ObjectDetector::getInstance();
    
    
    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    VideoCapture cap;
//    VideoWriter video;
    Mat frame, blob;
    
    try {
        
        outputFile = "yolo_out_cpp.avi";
        if (parser.has("image"))
        {
            // Open the image file
            str = parser.get<String>("image");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.jpg");
            outputFile = str;
        }
        else if (parser.has("video"))
        {
            // Open the video file
            str = parser.get<String>("video");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.avi");
            outputFile = str;
        }
        // Open the webcaom
        else cap.open(0);
        
    }
    catch(...) {
        cout << "Could not open the input image/video stream" << endl;
        return 0;
    }
    
//    // Get the video writer initialized to save the output video
//    if (!parser.has("image")) {
//        video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
//    }
    
    // Create a window
    static const string kWinName = "Traffic Accident Detector";
    namedWindow(kWinName, WINDOW_FULLSCREEN);
    moveWindow(kWinName, 0, 0);

    detect::ObjectTracker<cv::TrackerMedianFlow> objTracker;

    int frames = 0;
    auto start = chrono::steady_clock::now();

    // Process frames.
    while (waitKey(1) < 0)
    {
        // get frame from the video
        cap >> frame;

        // Stop the program if reached end of video
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
//            cout << "Output file is stored as " << outputFile << endl;
            //waitKey(3000);
            break;
        }
        
        if(frames % 5 == 0)
        {
            std::vector<cv::Rect> boxes = objDetector.processFrame(frame);

            objTracker.addTrackers(frame, boxes);
        }

        objTracker.updateTrackers(frame);

        auto& boxes = objTracker.getBoxes();

        for(auto& box : boxes)
        {
            rectangle(frame, box, Scalar(255, 0, 0), 2, 1);
        }
        
        // Put efficiency information.
        double t = objDetector.getPerfProfile();
        string label = format("Inference time for a frame : %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
        
//        // Write the frame with the detection boxes
//        Mat detectedFrame;
//        frame.convertTo(detectedFrame, CV_8U);
//        if (parser.has("image")) imwrite(outputFile, detectedFrame);
//        else video.write(detectedFrame);
        
        imshow(kWinName, frame);
        frames++;
    }

    auto end = chrono::steady_clock::now();
    auto elapsedTime = chrono::duration_cast<chrono::seconds>(end - start).count();
    std::cout << frames << " frames processed in "
              << elapsedTime << " sec" << std::endl
              << "Avg fps/s: " << frames / elapsedTime << std::endl;
    
    cap.release();
//    if (!parser.has("image")) video.release();

    return 0;
}


 
