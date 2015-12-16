#ifndef __FACERECOGNIZER_H__
#define __FACERECOGNIZER_H__

#include <iostream>
#include <vector>
#include <float.h>
#include <opencv/cv.h>

#include "PCA_Custom.h"

using namespace std;
using namespace cv;

class FaceRecognizer {
public:
    FaceRecognizer(const Mat& trFaces, const vector<int>& trImageIDToSubjectIDMap, int components);
    ~FaceRecognizer();

    /* Inner function, used for constructor. */
    void init(const Mat& trFaces, const vector<int>& trImageIDToSubjectIDMap, int components);

    /* Predict function */
    int recognize(const Mat& instance, int sim_measure = NORM_L2);

    /* Get functions : PCA wrappers */
    Mat getAverage();
    Mat getEigenvectors();
    Mat getEigenvalues();

    // Develop face reconstruction now
    Mat reconstructFaces(int ImageID);

private:
    /* major source of algorithm comes from OpenCV lib */
    PCA_Custom *pca;
    /* DB of faces from training data */
    vector<Mat> projTrFaces;
    /* Index of each traning faces */
    vector<int> trImageIDToSubjectIDMap;
};

#endif
