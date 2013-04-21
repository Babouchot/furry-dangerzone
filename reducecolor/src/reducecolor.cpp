#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

#define COLOR false

unsigned Error(const Mat & hist, const vector<int> & g)
{
    unsigned e = 0;
    for(unsigned i = 0; i < g.size(); i++)
        e += cvRound(hist.at<float>(i)) * abs(g[i] - i);
    return e;
}

void ScanImageAndReduceRound(const Mat& src, Mat& dst, int n)
{
    // accept only char type matrices
    src.copyTo(dst);

    MatIterator_<uchar> it, end;
    for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; ++it)
        *it = int( int(*it/(256.0/n)) * (256.0/n) );
}

void ScanImageAndReduce3BruteForce(const Mat& src, Mat& dst, const int * histSize, const float ** ranges)
{
    Mat hist;
    calcHist(&src, 1, 0, Mat(), hist, 1, histSize, ranges);

    src.copyTo(dst);

    MatIterator_<uchar> it, end;
    for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; ++it)
        *it = int( int(*it/(256.0/3)) * (256.0/3) );
}

void ScanImageAndReduceDyn(const Mat& src, Mat& dst, unsigned n, const int * histSize, const float ** ranges)
{
    Mat hist;
    calcHist(&src, 1, 0, Mat(), hist, 1, histSize, ranges);

    vector< vector<int> > E(256, vector<int>(n)); // Error
    vector< vector< vector<int> > > G(256, vector< vector<int> >(n)); // vector generating this error

    // première colonne
    for (unsigned j = 0; j < n; j++)
    {
        E[0][j] = 0;
        G[0][j].push_back(0);
    }

    // première ligne
    for (unsigned i = 1; i < 256; i++)
    {
        vector<int> gprec = G[i-1][0];
        gprec.push_back(gprec[i-1]);
        vector<int> gsuiv (i+1, i);

        unsigned eprec = Error(hist, gprec);
        unsigned esuiv = Error(hist, gsuiv);

        if (eprec < esuiv)
        {
            E[i][0] = eprec;
            G[i][0] = gprec;
        }
        else if (eprec > esuiv)
        {
            E[i][0] = esuiv;
            G[i][0] = gsuiv;
        }
        else
        {
            cerr << "EGALITE !!!! [" << i << "][" << 0 << "]" << endl;
            E[i][0] = eprec;
            G[i][0] = gprec;
        }
    }

    for (unsigned i = 1; i < 256; i++)
        for (unsigned j = 1; j < n; j++)
        {
            vector<int> ggauche = G[i-1][j];
            ggauche.push_back(ggauche.at(i-1));
            vector<int> ghaut = G[i][j-1];
            for (unsigned m = (ghaut.at(i-1) + i) / 2; m < ghaut.size(); m++)
                ghaut[m] = i;
            ghaut.push_back(i);

            unsigned egauche = Error(hist, ggauche);
            unsigned ehaut = Error(hist, ghaut);

            if (egauche < ehaut)
            {
                E[i][j] = egauche;
                G[i][j] = ggauche;
            }
            else if (egauche > ehaut)
            {
                E[i][j] = ehaut;
                G[i][j] = ghaut;
                cerr << "Descente !!!! [" << i << "][" << j << "]" << endl;
            }
            else
            {
                // cerr << "EGALITE !!!! [" << i << "][" << j << "]" << endl;
                E[i][j] = egauche;
                G[i][j] = ggauche;
            }
        }

    src.copyTo(dst);
    MatIterator_<uchar> it, end;
    for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; ++it)
        *it = G[255][n-1].at(*it);
    cerr << "Erreur finale " << E[255][n-1] << endl; // 3268512
}

void ScanImageAndReduceDyn2(const Mat& src, Mat& dst, unsigned n, const int * histSize, const float ** ranges)
{

}

int main(int argc, const char** argv)
{
    Mat image, imageRound, imageDyn;
    int n = 3;
    int histSize = 256;
    float range[] = { 0, 256 } ;
    const float* histRange[] = { range };

    if (argc != 2)
    {
        cerr << " Usage: reducecolor image" << endl;
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
        cerr << "Could not open or find " << argv[1] << endl;
        return -1;
    }

    /// Display
    namedWindow("calcHist");
    namedWindow("Origin");
    namedWindow("Round");
    namedWindow("Dynamic");

    if (COLOR)
    {
        vector<Mat> bgr_planes;
        vector<Mat> bgr_round(3);
        vector<Mat> bgr_dyn(3);
        split( image, bgr_planes );

        ScanImageAndReduceRound (bgr_planes[0], bgr_round[0], n);
        ScanImageAndReduceRound (bgr_planes[1], bgr_round[1], n);
        ScanImageAndReduceRound (bgr_planes[2], bgr_round[2], n);

        ScanImageAndReduceDyn (bgr_planes[0], bgr_dyn[0], n, &histSize, histRange);
        ScanImageAndReduceDyn (bgr_planes[1], bgr_dyn[1], n, &histSize, histRange);
        ScanImageAndReduceDyn (bgr_planes[2], bgr_dyn[2], n, &histSize, histRange);

        merge(bgr_dyn, imageDyn);
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
        ScanImageAndReduceDyn (image, imageDyn, n, &histSize, histRange);

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
    imshow("Dynamic", imageDyn);

    waitKey(0);                            // Wait for a keystroke in the window
    return EXIT_SUCCESS;
}
