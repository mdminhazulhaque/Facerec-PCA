/* OpenCV */
#include <opencv/cv.h>
#include <opencv/highgui.h>
/* Cpp Lib */
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <getopt.h>

/* Self-implemented Lib */
#include "FaceRecognizer.h"

using namespace std;
using namespace cv;

// Scale every value in a matrix so that it is in the 0-255 range. Thanks to
// the guys at stackoverflow for this one:
//
//     http://stackoverflow.com/questions/4974709/pca-svm-using-c-syntax-in-opencv-2-2
//
Mat toGrayscale(const Mat& mat) {
    Mat gMat(mat.rows, mat.cols, CV_8UC1);

    double min, max;
    minMaxLoc(mat, &min, &max);

    for(int row = 0; row < mat.rows; row++) {
        for(int col = 0; col < mat.cols; col++) {
            gMat.at<uchar>(row, col) = 255 * ((mat.at<float>(row, col) - min) / (max - min));
        }
    }

    return gMat;
}

// Test lists consist of the subject ID of the person of whom a face image is and the
// path to that image.
void readFile(const string& fileName, vector<string>& files, vector<int>& indexNumToSubjectIDMap)
{
    std::ifstream file(fileName.c_str(), ifstream::in);

    if(!file) {
        cerr << "Unable to open file: " << fileName << endl;
        exit(0);
    } else {
        cout << "open the file with path " << fileName << endl;
    }

    std::string line, path, trueSubjectID;
    while (std::getline(file, line)) {
        std::stringstream liness(line);
        std::getline(liness, trueSubjectID, ';');
        std::getline(liness, path);

        path.erase(std::remove(path.begin(), path.end(), '\r'), path.end());
        path.erase(std::remove(path.begin(), path.end(), '\n'), path.end());
        path.erase(std::remove(path.begin(), path.end(), ' '), path.end());

        cout << "Load image from " << path << " labeled as "
            << atoi( trueSubjectID.c_str()) << endl;
        files.push_back(path);
        indexNumToSubjectIDMap.push_back(atoi(trueSubjectID.c_str()));
    }
}

int main ()
{

    vector<string> testFaceFiles;
    vector<int> testImageIDToSubjectIDMap;

    vector<string> trainFaceFiles;
    vector<int> trainImageIDToSubjectIDMap;

    string trainingList = "train_data.txt";
    string testList = "test_data.txt";

    bool doShow = false;

    readFile(testList, testFaceFiles, testImageIDToSubjectIDMap);
    readFile(trainingList, trainFaceFiles, trainImageIDToSubjectIDMap);

    Mat testImg = imread(testFaceFiles[0], 0);
    Mat trainImg = imread(trainFaceFiles[0], 0);

    if(testImg.empty() || trainImg.empty())
    {
        return 1;
    }


    // Eigenfaces operates on a vector representation of the image so we calculate the
    // size of this vector.  Now we read the face images and reshape them into vectors
    // for the face recognizer to operate on.
    int imgVectorSize = trainImg.cols * trainImg.rows;

    // Create a matrix that has 'imgVectorSize' rows and as many columns as there are images.
    Mat trImgVectors(imgVectorSize, trainFaceFiles.size(), CV_32FC1);

    // Load the vector.
    for(unsigned int i = 0; i < trainFaceFiles.size(); i++)
    {
        Mat tmpTrImgVector = trImgVectors.col(i);
        Mat tmp_img;
        imread(trainFaceFiles[i], 0).convertTo(tmp_img, CV_32FC1);
        tmp_img.reshape(1, imgVectorSize).copyTo(tmpTrImgVector);
    }

    // On instantiating the myFaceRecognizer object automatically processes
    // the training images and projects them.
    clock_t pca_begin = clock();
    FaceRecognizer faceRecognizer(trImgVectors, trainImageIDToSubjectIDMap, 30);
    clock_t pca_end = clock();
    double pca_operation = double(pca_end - pca_begin) / CLOCKS_PER_SEC;
    cout << "init pca time: " << pca_operation * 1000 << endl;

    for(unsigned int i=0; i<testFaceFiles.size(); i++)
    {
        Mat rmat = imread(testFaceFiles[i],0);
        rmat.convertTo(testImg, CV_32FC1);

        // Look up the true subject ID of the current test subject.
        //int currentTestSubjectID = teImageIDToSubjectIDMap[i];

        // Compare a face from the list of test faces with the set of training faces stored in
        // the recognizer and try to find the closest match.
        clock_t time_begin = clock();
        int recAs = faceRecognizer.recognize(rmat.reshape(1, imgVectorSize));
        clock_t time_end = clock();

        double time_operation = double(time_end - time_begin) / CLOCKS_PER_SEC;
        cout << "file: " << testFaceFiles[i] << " id: " << recAs << " time: " << time_operation << endl;
    }

    // Let's show the average face and some of the test faces on screen.
    if(doShow)
    {
        Mat oAvgImage = toGrayscale(faceRecognizer.getAverage()).reshape(1, testImg.rows);
        Mat sAvgImage;
        // why should we resize as twice size of the origin?
        resize(oAvgImage, sAvgImage, Size(testImg.cols*2,testImg.rows*2), 0, 0, INTER_LINEAR);

        imshow("The average face", sAvgImage);

        // Show the test image and subtract it from the average face
        // <Note> the average face is not be enlarged
        Mat test_face = imread(testFaceFiles[0], 0);
        Mat sTest_face;
        resize(test_face, sTest_face, Size(testImg.cols*2, testImg.rows*2), 0, 0, INTER_LINEAR);

        imshow("Test face",sTest_face);
        Mat sFace_variance = sTest_face - sAvgImage;
        imshow("Phi of face", sFace_variance);

        // int eigenCount = trFaceFiles.size(); // Show all faces
        int eigenCount = 1;// Show all faces

        if(faceRecognizer.getEigenvectors().rows < eigenCount){
            eigenCount = faceRecognizer.getEigenvectors().rows;
        }

        Mat oEigenvalues = faceRecognizer.getEigenvalues();
        cout << "Corresponding " << to_string(eigenCount) << " eigenvalues\n"
             << " ------------------------\n ";
        for(int i = 0; i < eigenCount; i++) {
            stringstream windowTitle;
            windowTitle << "Eigenface No." << i << " Eigval: "<< oEigenvalues.row(i)  ;

            cout << "\tNo. "<< i << "\t" << oEigenvalues.row(i) << endl;

            Mat oEigenImage = toGrayscale(faceRecognizer.getEigenvectors().row(i)).reshape(1, testImg.rows);
            Mat sEigenImage;
            resize(oEigenImage, sEigenImage, Size(testImg.cols*2, testImg.rows*2), 0, 0, INTER_LINEAR);
            imshow(windowTitle.str(), sEigenImage);
        }

//        cout << "getEigenvectors :" << fr.getEigenvectors() << endl;
        //Mat ReconstructedFace = fr.reconstructFaces(0);
       // imshow("Reconstructed Faces",ReconstructedFace);

            Mat RectFace = toGrayscale(faceRecognizer.reconstructFaces(0)).reshape(1, testImg.rows);
            Mat sRectFace;
            resize(RectFace, sRectFace, Size(testImg.cols*2, testImg.rows*2), 0, 0, INTER_LINEAR);
            imshow("Reconstructed Face",sRectFace);

        waitKey(0);
    }
}
