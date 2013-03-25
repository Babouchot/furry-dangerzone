#include "opencv2/video/tracking.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

static void help()
{
    cout << "This reads from video camera (0 by default, or the camera number the user enters\n"
            "Usage: \n"
            "./camshiftdemo --cascade=<cascade_path> this is the primary trained classifier such as frontal face\n"
               "   [camera number]\n"
            "\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tc - reset the tracking\n"
            "\tb - switch to/from backprojection view\n"
            "\th - show/hide object histogram\n"
            "\tUsing OpenCV version " << CV_VERSION << endl << endl;
}

void rectToSquare (Rect &r, int xmax, int ymax)
{
    if (r.height > r.width)
    {
        r.x = r.x - (r.height - r.width)/2;
        if (r.x < 0) r.x = 0;
        r.width = r.height;
        if (r.x + r.width > xmax) r.x = xmax - r.width;
    }
    else
    {
        r.y = r.y - (r.width - r.height)/2;
        if (r.y < 0) r.y = 0;
        r.height = r.width;
        if (r.y + r.height > xmax) r.y = xmax - r.height;
    }
}

int main(int argc, const char** argv)
{
    const string cascadeOpt = "--cascade=";
    size_t cascadeOptLen = cascadeOpt.length();
    string inputName = "0", cascadeName = "";
    CascadeClassifier cascade;
    VideoCapture capture;
    Mat frame, img, hsv, mask, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
    MatND hist;
    bool backprojMode = false;
    bool trackObject = false;
    bool showHist = false;
    int vmin = 10, vmax = 256, smin = 30, smax = 256;
    Rect trackWindowHead, trackWindowHand;
    int channels[] = {0, 1};
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };

    string savein = "./lettres/";
    char chartosave;
    string saveformat = "bmp";
    bool saveMode = false;
    int nbsave = 200;
    int nbsaved = 0;
    int savesize = 128;


    help();

    for( int i = 1; i < argc; i++ )
    {
        cout << "Processing " << i << " " <<  argv[i] << endl;
        if( cascadeOpt.compare( 0, cascadeOptLen, argv[i], cascadeOptLen ) == 0 )
        {
            cascadeName.assign( argv[i] + cascadeOptLen );
            cout << "  from which we have cascadeName=" << cascadeName << endl;
        }
        else if( argv[i][0] == '-' )
        {
            cerr << "WARNING: Unknown option: " << argv[i] << endl;
        }
        else
        {
            inputName.assign(argv[i]);
        }
    }

    if (!cascade.load(cascadeName))
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

    if (isdigit(inputName[0]) && inputName.length() == 1)
    {
        capture.open(inputName[0] - '0');
    }
    else
    {
        capture.open(inputName);
    }

    if( !capture.isOpened() )
    {
        cout << "***Could not initialize capturing...***\n"
                "Current parameter's value: " << inputName << endl;
        return EXIT_FAILURE;
    }

    namedWindow("CamShift Face");

    cout << "In capture ..." << endl;
    for (;;)
    {
        capture >> frame;
        if (frame.empty())
            break;
        //frame.copyTo(img);
        flip( frame, img, 1 );

        cvtColor(img, hsv, CV_BGR2HSV);
        inRange(hsv, Scalar(0, smin, vmin),
                Scalar(180, smax, vmax), mask);

        if (trackObject && trackWindowHead.area() > 1)
        {
            calcBackProject(&hsv, 1, channels, hist, backproj, ranges);
            backproj &= mask;
            RotatedRect trackBoxHead = CamShift(backproj, trackWindowHead,
                    TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 1));

            rectangle(backproj, trackBoxHead.boundingRect(), Scalar(0, 0, 0), CV_FILLED);

            if (backprojMode)
                cvtColor(backproj, img, CV_GRAY2BGR);
            ellipse(img, trackBoxHead, Scalar(0, 0, 255), 3, CV_AA);

            if (trackWindowHand.area() <= trackWindowHead.area() / 8.0)
            {
                trackWindowHand.x = 0;
                trackWindowHand.y = 0;
                trackWindowHand.height = backproj.rows;
                trackWindowHand.width = backproj.cols;
            }

            RotatedRect trackBoxHand = CamShift(backproj, trackWindowHand,
                    TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 1));

            //ellipse(img, trackBoxHand, Scalar(255, 0, 0), 3, CV_AA);

            Rect handSquare = trackBoxHand.boundingRect();
            rectToSquare(handSquare, img.cols, img.rows);

            if (saveMode)
            {
                ostringstream oss117;
                oss117 << savein << chartosave << "/" << ++nbsaved << "." << saveformat;
                Mat imgToSave (backproj, handSquare);
                resize(imgToSave, imgToSave, Size(savesize, savesize));
                imwrite(oss117.str(), imgToSave);
                rectangle(img, Rect(0,0, img.cols, 20), Scalar(255, 100, 0), 3, CV_AA);
                cout << (img.cols*nbsave)/nbsaved << endl;
                rectangle(img, Rect(0,0, (img.cols*nbsaved)/nbsave, 20), Scalar(255, 100, 0), CV_FILLED);

                if(nbsaved >= nbsave)
                    saveMode = false;
            }

            rectangle(img, handSquare, Scalar(0, 255, 0), 3, CV_AA);
        }
        else
        {
            vector<Rect> faces;
            Mat gray, smallImg(img.rows, img.cols, CV_8UC1);
            cvtColor(img, gray, CV_BGR2GRAY);
            resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
            equalizeHist(smallImg, smallImg);
            cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0
                    | CV_HAAR_FIND_BIGGEST_OBJECT
                    //|CV_HAAR_DO_ROUGH_SEARCH
                    | CV_HAAR_SCALE_IMAGE
                    , Size(30, 30));
            vector<Rect>::const_iterator r;
            if ((r = faces.begin()) != faces.end())
            {
                trackWindowHead = *r;
                Mat roi(hsv, trackWindowHead), maskroi(mask, trackWindowHead);
                calcHist( &roi, 1, channels, maskroi,
                         hist, 2, histSize, ranges);
                normalize(hist, hist, 0, 255, CV_MINMAX);

                trackObject = true;

                int scale = 10;
                histimg = Mat::zeros(sbins*scale, hbins*10, CV_8UC3);

                for( int h = 0; h < hbins; h++ )
                    for( int s = 0; s < sbins; s++ )
                    {
                        float binVal = hist.at<float>(h, s);
                        int intensity = cvRound(binVal);
                        rectangle( histimg, Point(h*scale, s*scale),
                                    Point( (h+1)*scale - 1, (s+1)*scale - 1),
                                    Scalar::all(intensity),
                                    CV_FILLED );
                    }
            }
        }

        imshow("CamShift Face", img);
        if (showHist)
            imshow("Histogram", histimg);

        // detection des touches.
        char c = (char) waitKey(10);
        if (c == 27)
            break;
        switch (tolower(c))
        {
        case ',':
            backprojMode = !backprojMode;
            break;
        case ';':
            trackObject = 0;
            histimg = Scalar::all(0);
            break;
        case ':':
            showHist = !showHist;
            if (!showHist)
                destroyWindow("Histogram");
            else
                namedWindow("Histogram");
            break;
        default:
            if ('a' <= tolower(c) && tolower(c) <= 'z')
            {
                saveMode = true;
                nbsaved = 0;
                chartosave = toupper(c);
            }
            break;
        }
    }

    return EXIT_SUCCESS;
}
