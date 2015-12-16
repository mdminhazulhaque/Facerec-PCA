#ifndef __PCA_CUSTOM_H__
#define __PCA_CUSTOM_H__

#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/internal.hpp"

using namespace cv;

class PCA_Custom
{
public:
    //! default constructor
    PCA_Custom();
    //! the constructor that performs PCA_Custom
    PCA_Custom(InputArray data, InputArray mean, int flags, int maxComponents=0);
    PCA_Custom(InputArray data, InputArray mean, int flags, double retainedVariance);
    //! operator that performs PCA_Custom. The previously stored data, if any, is released
    PCA_Custom& operator()(InputArray data, InputArray mean, int flags, int maxComponents=0);
    PCA_Custom& computeVar(InputArray data, InputArray mean, int flags, double retainedVariance);
    //! projects vector from the original space to the principal components subspace
    Mat project(InputArray vec) const;
    //! projects vector from the original space to the principal components subspace
    void project(InputArray vec, OutputArray result) const;
    //! reconstructs the original vector from the projection
    Mat backProject(InputArray vec) const;
    //! reconstructs the original vector from the projection
    void backProject(InputArray vec, OutputArray result) const;

    Mat eigenvectors; //!< eigenvectors of the covariation matrix
    Mat eigenvalues; //!< eigenvalues of the covariation matrix
    Mat mean; //!< mean value subtracted before the projection and added after the back projection
};

#endif // __PCA_CUSTOM_H__
