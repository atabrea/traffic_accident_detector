CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET(PROJECT_NAME accident_detector)
PROJECT(${PROJECT_NAME})

FIND_PACKAGE(OpenCV REQUIRED world)

LIST(APPEND SRCS
    detect/ObjectDetector.cpp
    detect/ObjectDetector.hpp
    detect/ObjectTracker.cpp
    detect/ObjectTracker.hpp)

foreach(SRC IN LISTS SRCS)
    LIST(APPEND SRCS_PATH src/${SRC})
endforeach()

INCLUDE_DIRECTORIES(src)

ADD_EXECUTABLE(${PROJECT_NAME} src/main.cpp ${SRCS_PATH})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})

ADD_EXECUTABLE(multi_tracker src/multiTracker.cpp)
TARGET_LINK_LIBRARIES(multi_tracker ${OpenCV_LIBS})

