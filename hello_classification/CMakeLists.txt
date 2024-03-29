# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "hello_classification")

find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "The Opecv lib is found!")
endif()

if(NOT OpenCV_FOUND)
    message(STATUS "The Opecv lib is not found!")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
file (GLOB SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        )
message("**********************" ${SRC})
# Find OpenCV components if exist
find_package(OpenCV COMPONENTS imgcodecs QUIET)
if(NOT(OpenCV_FOUND))
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
    return()
endif()

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SRC})

link_directories(${LIB_FOLDER})

# Create library file from sources.

message("**********************" ${SRC})
add_executable(${TARGET_NAME} ${SRC} ${CMAKE_CURRENT_SOURCE_DIR}/reid.cpp)

if(WIN32)
	# This target supports UNICODE on Windows
	set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "/D_UNICODE /DUNICODE")
endif()

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
COMPILE_PDB_NAME ${TARGET_NAME})

target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${OpenCV_LIBRARIES})


if(UNIX)
    target_link_libraries(${TARGET_NAME} ${LIB_DL} pthread)
endif()
