#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class STCTracker
{
public:

	STCTracker()
	    : numFrame(0),
		  padding(1),
		  rho(0.075),
		  scale(1),
		  lambda(0.25),
		  alpha(2.25),
		  eps(0.00001) 
	{}

	~STCTracker()
	{
		delete[] maxconf;
	}

	void init(
		cv::Rect selection,
		cv::Mat frame
		)
	{
		if (frame.channels() == 3)
			cv::cvtColor(frame, frame, CV_RGB2GRAY);

		numFrame++;
		
		// initialization
		rect = selection;
		nPos = cv::Point2i(
			selection.x+selection.width/2,
			selection.y+selection.height/2
			);

		// store pre-computed weight window
		targetsz = selection.size();
		sz = cv::Size(
			selection.width*(1 + padding),
			selection.height*(1 + padding)
			);
		cv::createHanningWindow(hamming, sz, CV_32F);
		
		// store pre-computed confidence map
		dist = cv::Mat(sz, CV_32F);
		for (int i = 0; i < sz.height; i++)
		{
			float* dist_ptr = dist.ptr<float>(i);
			for (int j = 0; j < sz.width; j++)
			{
				float ti = (float)(i*2 - sz.height) / sz.height;
				float tj = (float)(j*2 - sz.width) / sz.width;
				dist_ptr[j] = ti*ti + tj*tj;
			}
		}
		cv::sqrt(dist, dist);
		cv::exp(-0.5 / alpha * dist, conf);
		cv::dft(conf, conff);

		// %update weight function w_{ sigma } in Eq.(11)
		sigma = (selection.width + selection.height) / 2;
		window = ReduceFrequencyEffect(sigma);

		// update the spatial context model h^{sc} in Eq.(9)
		cv::Mat contextprior;
		cv::Mat hscf;
		
		contextprior = getContext(frame, nPos, sz, window);
		cv::dft(contextprior, contextprior);
		cv::divide(conff, contextprior+eps, hscf);

		// update the spatio - temporal context model by Eq.(12)
		Hstcf = hscf;
	}

	cv::Rect update(cv::Mat frame)
	{
		if (frame.channels() == 3)
			cv::cvtColor(frame, frame, CV_RGB2GRAY);

		numFrame++;
		
		// update scale in Eq.(15)
		sigma *= scale;
		window = ReduceFrequencyEffect(sigma);

		cv::Mat contextprior(
			getContext(frame, nPos, sz, window)
			);

		// calculate response of the confidence map at all locations
		cv::Mat confmap;
		// Eq.(11)
		cv::dft(contextprior, contextprior);
		contextprior = contextprior.mul(Hstcf);
		cv::dft(
			contextprior,
			confmap,
			cv::DFT_REAL_OUTPUT + cv::DFT_INVERSE
			);
		// target location is at the maximum response
		double* fMaxValue = 0;
		cv::minMaxLoc(confmap, NULL, fMaxValue, NULL, &nPos);

		// About update
		contextprior = getContext(frame, nPos, sz, window);

		cv::Mat conftmp;
		cv::dft(contextprior, contextprior);
		contextprior = contextprior.mul(Hstcf);
		cv::dft(
			contextprior,
			conftmp,
			cv::DFT_REAL_OUTPUT + cv::DFT_INVERSE
			);

		double maxValue;
		cv::minMaxLoc(conftmp, NULL, &maxValue, NULL, NULL);
		maxconf[numFrame-1] = (float)maxValue;
		// update scale by Eq.(15)
		if (numFrame % num == 0)
		{
			float curScale = 0;
			for (int k = 0; k < num; k++)
			{
				int tNumFrame = numFrame - 1;
				int tIdx_1 = (tNumFrame - k < 0 ? tNumFrame + num - k : tNumFrame - k);
				int tIdx_2 = (tNumFrame - k - 1 < 0 ? tNumFrame + num - k - 1 : tNumFrame - k - 1);
				curScale += std::sqrt(maxconf[tIdx_1] / maxconf[tIdx_2]);
			}
			// update
			scale = (1 - lambda) * scale + lambda * (curScale / num);

			numFrame = 0;
		}

		// generate result rectange
		targetsz = cv::Size(targetsz.width*scale, targetsz.height*scale);
		int resX = nPos.x - targetsz.width / 2;
		int resY = nPos.y - targetsz.height / 2;

		return cv::Rect(resX, resY, targetsz.width, targetsz.height);
	}

private:

	cv::Mat getContext(
		const cv::Mat & frame,
		const cv::Point2i & nPos,
		const cv::Size & sz,
		const cv::Mat & window
		)
	{
		#define MAX(a, b) (a) > (b) ? (a) : (b)
		#define MIN(a, b) (a) < (b) ? (a) : (b)
		
		cv::Mat out(sz, CV_8U);
		for (int i = 0; i < sz.height; i++)
		{
			int numFrameCurRow = MAX(0, MIN(frame.rows-1, nPos.y - sz.height / 2 + i));
			
			uchar* out_ptr = out.ptr<uchar>(i);
			const uchar* frame_ptr = frame.ptr<uchar>(numFrameCurRow);

			for (int j = 0; j < sz.width; j++)
			{
				int numFrameCurCol = MAX(0, MIN(frame.cols-1, nPos.x - sz.width / 2 + j));

				memcpy(&out_ptr[j], &frame_ptr[numFrameCurCol], sizeof(uchar));
			}
		}
		out.convertTo(out, CV_32F);
		out -= cv::mean(out);
		out.mul(window);

		return out;
	}

	cv::Mat ReduceFrequencyEffect(float f)
	{
		cv::Mat weight;
		cv::exp(-0.5 / (f*f) * dist, weight);
		// use Hamming window to reduce frequency effect of image boundary
		window = hamming.mul(weight);
		window /= cv::sum(window)[0];

		return window;
	}

	// variable for STC
    const static int num = 5; // number of average frames
	float maxconf[num];

	int numFrame;             // number of current frame
	float padding;            // extra area surrounding the target
	float rho;                // the learning parameter rho in Eq.(12)
	float scale;              // initial scale ratio
	float lambda;             // lambda in Eq.(15)
	float alpha;              // parameter alpha in Eq.(6)
	float sigma;
	float eps;

	cv::Rect rect;            // rectangle [x,y,width,height]
	cv::Point2i nPos;         // center of the target
	cv::Size sz;              // size of context region
	cv::Size targetsz;        // size of tracking target

	cv::Mat dist;
	cv::Mat conf;             // confidence map function Eq.(6)
	cv::Mat conff;            // variable conf in frequency domain
	cv::Mat hamming;          // the hamming window
	cv::Mat window;
	cv::Mat frame;            // current frame image
	cv::Mat Hstcf;            // spatio-temporal context model

};