#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

#define COLOR true

int Error(const Mat & hist, int m, int from, int to)
{


    return 0;
}

void ScanImageAndReduceRound(const Mat& src, Mat& dst, int n)
{
    // accept only char type matrices
    src.copyTo(dst);

    MatIterator_<uchar> it, end;
    for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; ++it)
        *it = int( int(*it/(256.0/n)) * (256.0/n) );
}

void ScanImageAndReduceDyn(const Mat& src, Mat& dst, const uchar* const table)
{
    src.copyTo(dst);

    MatIterator_<uchar> it, end;
    for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; ++it)
        *it = table[*it];
}

int main(int argc, const char** argv)
{
    Mat image, imageRound, imageDyn;
    int n = 4;
    int histSize = 256;
    float range[] = { 0, 256 } ;
    const float* histRange[] = { range };

    if (argc != 2)
    {
        cout << " Usage: reducecolor image" << endl;
        return EXIT_FAILURE;
    }


    if (COLOR)
    {
        image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file
    }
    else
    {
        image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    }

    if (!image.data)                              // Check for invalid input
    {
        cout << "Could not open or find" << argv[1] << endl;
        return -1;
    }

    /// Display
    namedWindow("calcHist");
    namedWindow("Origin");
    namedWindow("Round");

    if (COLOR)
    {
        vector<Mat> bgr_planes;
        vector<Mat> bgr_round(3);
        split( image, bgr_planes );

        ScanImageAndReduceRound (bgr_planes[0], bgr_round[0], n);
        ScanImageAndReduceRound (bgr_planes[1], bgr_round[1], n);
        ScanImageAndReduceRound (bgr_planes[2], bgr_round[2], n);

        merge(bgr_round, imageRound);

        Mat b_hist, g_hist, r_hist;

        /// Compute the histograms:
        calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange);
        calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange);
        calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange);

        int hist_w = 512;
        int hist_h = 400;
        int bin_w = cvRound((double) hist_w / histSize);

        Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

        /// Normalize the result to [ 0, histImage.rows ]
        normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX);
        normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX);
        normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX);

        /// Draw for each channel
        for( int i = 1; i < histSize; i++ )
        {
          line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                           Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                           Scalar( 255, 0, 0), 2, 8, 0  );
          line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                           Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                           Scalar( 0, 255, 0), 2, 8, 0  );
          line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                           Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                           Scalar( 0, 0, 255), 2, 8, 0  );
        }

        imshow("calcHist", histImage);
    }
    else
    {
        Mat hist;

        calcHist(&image, 1, 0, Mat(), hist, 1, &histSize, histRange);

        ScanImageAndReduceRound (image, imageRound, n);

        int hist_w = 512;
        int hist_h = 400;
        int bin_w = cvRound((double) hist_w / histSize);

        Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

        /// Normalize the result to [ 0, histImage.rows ]
        normalize(hist, hist, 0, histImage.rows, NORM_MINMAX);

        /// Draw for each channel
        for( int i = 1; i < histSize; i++ )
        {
          line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                           Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                           Scalar( 255, 255, 255), 2, 8, 0  );
        }

        imshow("calcHist", histImage);

    }

    imshow("Origin", image);              // Show our image inside it.
    imshow("Round", imageRound);

    waitKey(0);                            // Wait for a keystroke in the window
    return EXIT_SUCCESS;
}
