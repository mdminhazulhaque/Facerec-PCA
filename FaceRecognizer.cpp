#include "FaceRecognizer.h"

FaceRecognizer::FaceRecognizer(const Mat& trFaces, const vector<int>& trImageIDToSubjectIDMap, int components)
{
    init(trFaces, trImageIDToSubjectIDMap, components);
}

void FaceRecognizer::init(const Mat& trFaces, const vector<int>& trImageIDToSubjectIDMap, int components)
{
    int numSamples = trFaces.cols;
    this->pca = new PCA_Custom(trFaces, Mat(),CV_PCA_DATA_AS_COL, components);

    for(int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
        this->projTrFaces.push_back(pca->project(trFaces.col(sampleIdx)));
    }

    this->trImageIDToSubjectIDMap = trImageIDToSubjectIDMap;
}

FaceRecognizer::~FaceRecognizer()
{
    delete pca;
}

Mat FaceRecognizer::reconstructFaces(int ImageID)
{
    return pca->backProject(projTrFaces.at(ImageID));
}


/* The Major part of the implementation
 *
 * */
int FaceRecognizer::recognize(const Mat& unknown, int sim_measure)
{
    // Take the vector representation of unknown's face image and project it
    // into face space.
    Mat unkProjected = pca->project(unknown);

    // I now want to know which individual in my training data set has the shortest distance
    // to unknown.
    double closestFaceDist = DBL_MAX;
    // DBL_MAX is the MACRO from <float.h>, about 1e37

    int closestFaceID = -1;
    for(unsigned int i = 0; i < projTrFaces.size(); i++) {
        // Given two vectors and the NORM_L2 type ID the norm function computes the Euclidean distance:
        //     dist(SRC1,SRC2) = SQRT(SUM(SRC1(I) - SRC2(I))^2)
        // between the projections of the current training face and unknown.
        Mat src1 = projTrFaces[i];
        Mat src2 = unkProjected;

        // normType-> NORM_INF/NORM_L1/NORM_L2
        double dist = norm(src1, src2, sim_measure);

        // Every time I find somebody with a shorter distance I save the distance, map his
        // index number to his actual ID in the set of traing images and save the actual ID.
        if(dist < closestFaceDist) {
            closestFaceDist = dist;
            closestFaceID = this->trImageIDToSubjectIDMap[i];
        }
    }

    return closestFaceID;
}

// Returns the vector containing the average face.
Mat FaceRecognizer::getAverage()
{
    return pca->mean;
}


// Returns a matrix containing the eigenfaces.
Mat FaceRecognizer::getEigenvectors()
{
    return pca->eigenvectors;
}

// Returns a matrix containing the eigenvalues.
Mat FaceRecognizer::getEigenvalues()
{
    return pca->eigenvalues;
}
