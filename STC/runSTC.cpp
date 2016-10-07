#include <iostream>

#include "libSTC.hpp"
#include "runSTC.h"

void onMouse(int event, int x, int y, int, void*)
{
	if (selectCtrl)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);

		//selection &= cv::Rect(0, 0, frame.cols, frame.rows);
	}

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		selection.x = x;
		selection.y = y;
		origin.x = x;
		origin.y = y;
		selectCtrl = begSelect;
		break;
	case CV_EVENT_LBUTTONUP:
		selection.width = x - selection.x;
		selection.height = y - selection.y;
		selectCtrl = endSelect;
		break;
	}
}

int main()
{
	STC tracker;
	
	cv::VideoCapture video("bike.avi");
	cv::namedWindow("video", cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback("video", onMouse, 0);
	
	cv::Mat frame;
	video >> frame;

	const int thickness = 3;
	const int lineType = 8;

	cv::Mat tframe;
	while (true)
	{
		frame.copyTo(tframe);

		if (selectCtrl >= begSelect)
		{
			cv::rectangle(tframe, selection, cv::Scalar(0, 0, 255), thickness, lineType, 0);
			if (selectCtrl == endSelect)
				break;
		}
		cv::imshow("video", tframe);
		cv::waitKey(10);
	}

	tracker.init(selection, tframe);
	while (true)
	{
		video >> frame;
		if (frame.empty())
			break;

		double t = (double)cv::getTickCount();
		cv::Rect result = tracker.update(frame);
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << "Time cost: " << t << " ms." << std::endl;

		cv::rectangle(frame, result, cv::Scalar(0, 0, 255), thickness, lineType, 0);
		cv::imshow("video", frame);

		cv::waitKey(10);
	}

	cv::waitKey();
	return 0;
}