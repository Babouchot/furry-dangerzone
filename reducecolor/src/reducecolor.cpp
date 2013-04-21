#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

#define COLOR false

unsigned Error(const Mat & hist, const vector<unsigned> & g)
{
    unsigned e = 0;
    for(unsigned i = 0; i < g.size(); i++)
        e += cvRound(hist.at<float>(i)) * labs(g[i] - i);
    return e;
}

unsigned Error2(const Mat & image, const vector<unsigned> & g)
{
    unsigned e = 0;
    MatConstIterator_<uchar> it, end;
    for (it = image.begin<uchar>(), end = image.end<uchar>(); it != end; ++it)
    {
        e += labs(g[*it] - *it);
    }
    return e;
}

unsigned ScanImageAndReduceRound(const Mat& src, Mat& dst, int n) // renvoie l'erreur
{
    // accept only char type matrices
    src.copyTo(dst);

    unsigned e = 0;
    for( int i = 0; i < dst.rows; ++i)
        for( int j = 0; j < dst.cols; ++j )
        {
            unsigned HO = int( int(dst.at<uchar>(i,j)/(256.0/n)) * (256.0/n) );
            uchar he = int( int(dst.at<uchar>(i,j)/(256.0/n)) * (256.0/n) );
            cerr << endl;
            e += labs(int( int(dst.at<uchar>(i,j)/(256.0/n)) * (256.0/n) ) - dst.at<uchar>(i,j));
            dst.at<uchar>(i,j) = int( int(dst.at<uchar>(i,j)/(256.0/n)) * (256.0/n) );
        }
    return e;
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

unsigned ScanImageAndReduceDyn(const Mat& src, Mat& dst, unsigned n, const Mat & hist)
{
    vector< vector<unsigned> > E(256, vector<unsigned>(n)); // Error
    vector< vector< vector<unsigned> > > G(256, vector< vector<unsigned> >(n)); // vector generating this error

    // première colonne
    for (unsigned j = 0; j < n; j++)
    {
        E[0][j] = 0;
        G[0][j].push_back(0);
    }

    // première ligne
    for (unsigned i = 1; i < 256; i++)
    {
        vector<unsigned> gprec = G[i-1][0];
        gprec.push_back(gprec[i-1]);
        vector<unsigned> gsuiv (i+1, i);

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
            E[i][0] = eprec;
            G[i][0] = gprec;
        }
    }

    for (unsigned i = 1; i < 256; i++)
        for (unsigned j = 1; j < n; j++)
        {
            vector<unsigned> ggauche = G[i-1][j];
            ggauche.push_back(ggauche.at(i-1));

            unsigned egauche = Error(hist, ggauche);

            vector<unsigned> ghaut = G[i-1][j-1];
            unsigned ehaut;
            if((unsigned)(ghaut.at(i-1)) > i-1)
            {
                ehaut = egauche + 1;
            }
            else
            {
                ghaut.push_back(i + (i-1 - ghaut.at(i-1)));
                ehaut = Error(hist, ghaut);
            }

            if (egauche < ehaut)
            {
                E[i][j] = egauche;
                G[i][j] = ggauche;
            }
            else if (egauche > ehaut)
            {
                E[i][j] = ehaut;
                G[i][j] = ghaut;
            }
            else
            {
                E[i][j] = egauche;
                G[i][j] = ggauche;
            }
        }

        src.copyTo(dst);

        unsigned e = 0;
        for( int i = 0; i < dst.rows; ++i)
            for( int j = 0; j < dst.cols; ++j )
            {
                e += labs(G[255][n-1][dst.at<uchar>(i,j)] - dst.at<uchar>(i,j));
                dst.at<uchar>(i,j) = G[255][n-1][dst.at<uchar>(i,j)];
            }
        return e;
}

int main(int argc, const char** argv)
{
    Mat image, imageRound, imageDyn;
    int n = 8;
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

        Mat b_hist, g_hist, r_hist;

        /// Compute the histograms:
        calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange);
        calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange);
        calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange);

        ScanImageAndReduceRound (bgr_planes[0], bgr_round[0], n);
        ScanImageAndReduceRound (bgr_planes[1], bgr_round[1], n);
        ScanImageAndReduceRound (bgr_planes[2], bgr_round[2], n);

        ScanImageAndReduceDyn (bgr_planes[0], bgr_dyn[0], n, b_hist);
        ScanImageAndReduceDyn (bgr_planes[1], bgr_dyn[1], n, g_hist);
        ScanImageAndReduceDyn (bgr_planes[2], bgr_dyn[2], n, r_hist);

        merge(bgr_dyn, imageDyn);
        merge(bgr_round, imageRound);

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

        unsigned nb200 = 0;
        MatIterator_<uchar> it, end;
        for (it = image.begin<uchar>(), end = image.end<uchar>(); it != end; ++it)
        {
            if(*it == 200)
                nb200++;
        }
        cerr << nb200 << " / " << cvRound(hist.at<float>(200)) << endl;


        cerr << "Erreur Round : " << ScanImageAndReduceRound (image, imageRound, n) << endl;
        cerr << "Erreur Dynam : " << ScanImageAndReduceDyn (image, imageDyn, n, hist) << endl;

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
