QT += core gui widgets
HEADERS += \
    FaceRecognizer.h \
    PCA_Custom.h
SOURCES += \
    main.cpp \
    FaceRecognizer.cpp \
    PCA_Custom.cpp
CONFIG += c++11
OTHER_FILES += \
    test_data.txt \
    train_data.txt

unix {
    LIBS   += `pkg-config --libs --cflags opencv`
}

win32 {
    LIBS += -LC:\opencv2410\bin \
    -llibopencv_calib3d2410 \
    -llibopencv_contrib2410 \
    -llibopencv_core2410 \
    -llibopencv_features2d2410 \
    -llibopencv_flann2410 \
    -llibopencv_gpu2410 \
    -llibopencv_highgui2410 \
    -llibopencv_imgproc2410 \
    -llibopencv_legacy2410 \
    -llibopencv_ml2410 \
    -llibopencv_nonfree2410 \
    -llibopencv_objdetect2410 \
    -llibopencv_ocl2410 \
    -llibopencv_photo2410 \
    -llibopencv_stitching2410 \
    -llibopencv_superres2410 \
    -llibopencv_video2410 \
    -llibopencv_videostab2410
    INCLUDEPATH += C:\opencv2410\include
}
