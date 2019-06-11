#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

int main()
{
    cv::VideoCapture cap(0);
    cv::Ptr<cv::Tracker> tracker = cv::TrackerGOTURN::create();

    cv::Mat frame;
    if(!cap.read(frame))
    {
        std::cout << "Capture failed" << std::endl;
    }


    cv::Rect2d bbox;
    bbox = selectROI(frame, false);
    tracker->init(frame, bbox);

    while(cap.read(frame))
    {
        // Start timer
        double timer = (double)cv::getTickCount();

        // Update the tracking result
        bool ok = tracker->update(frame, bbox);

        // Calculate Frames per second (FPS)
        float fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);

        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, cv::Scalar( 255, 0, 0 ), 2, 1 );
        }
        else
        {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", cv::Point(100,80),
               cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,255),2);
        }

        // Display tracker type on frame
        putText(frame, "GOTURN Tracker", cv::Point(100,20), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(50,170,50),2);

        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)), cv::Point(100,50), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(50,170,50), 2);

        // Display frame.
        imshow("Tracking", frame);

        // Exit if ESC pressed.
        if(cv::waitKey(1) == 27) break;
    }

    cap.release();

}
