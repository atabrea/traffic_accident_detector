#include <opencv2/opencv.hpp>

int main()
{
//    cv::VideoCapture cap = cv::VideoCapture("https://play.webcamromania.ro/b3p4l5g5v534o2z2z2");
    cv::VideoCapture cap(0);

    cv::Mat frame, gray;

    while(cap.isOpened())
    {
        cap >> frame;
        if(frame.empty())
        {
            break;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::imshow("url", gray);
    }

    cap.release();

}
