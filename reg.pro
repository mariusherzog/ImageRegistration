TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CXXFLAGS += -std=c++11 -g -O3

LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui

SOURCES += main.cpp
