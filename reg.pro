TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui

SOURCES += main.cpp
