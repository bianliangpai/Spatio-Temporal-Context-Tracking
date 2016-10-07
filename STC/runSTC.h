enum
{
	nonSelect = 0,
	begSelect = 1,
	endSelect = 2
}selectCtrl;
struct
{
	int x;
	int y;
}origin;

cv::Rect selection;