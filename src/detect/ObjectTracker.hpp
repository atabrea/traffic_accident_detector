#ifndef OBJECT_TRACKER_HPP
#define OBJECT_TRACKER_HPP

#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;
using namespace std;


namespace detect
{
template<typename T = TrackerCSRT>
class ObjectTracker
{
public:
    ObjectTracker()
    {
    }

    void addTracker(cv::Mat& frame, cv::Rect& box)
    {
        Ptr<T> tracker = T::create();
        tracker->init(frame, box);
        multiTracker.push_back(tracker);
        boxes.push_back(box);
    }

    void addTrackers(cv::Mat& frame, std::vector<Rect>& boxes)
    {
        if(this->boxes.size() == boxes.size())
        {
            return;
        }
        for(Rect& box : boxes)
        {
            if(checkDistance(box))
            {
                addTracker(frame, box);
            }
        }
    }

    bool checkDistance(Rect& box)
    {
        for(auto& trackerBox : boxes)
        {
            if(sqrt(pow(box.tl().x - trackerBox.tl().x, 2) + pow(box.tl().y - trackerBox.tl().y, 2)) < 80)
            {
                return false;
            }
        }
        return true;
    }

    void updateTrackers(cv::Mat& frame)
    {
        for(int i=0; i < multiTracker.size(); i++)
        {
            auto tracker = multiTracker.begin();
            std::advance(tracker, i);

            std::list<cv::Rect2d>::iterator box = boxes.begin();
            std::advance(box, i);

            if(!(*tracker)->update(frame, *box))
            {
                std::cout << "Removing box" << std::endl;
                multiTracker.erase(tracker);
                boxes.erase(box);
                i--;
            }
        }
    }

    std::list<cv::Rect2d>& getBoxes()
    {
        return boxes;
    }

private:
    std::list<Ptr<T>> multiTracker;
    std::list<cv::Rect2d> boxes;
};
} // namespace detect

#endif // OBJECT_TRACKER_HPP
