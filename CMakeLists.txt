cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# Disable in-source builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "\nFATAL: In-source builds are not allowed.\n"
        "You should create a separate directory for build files."
        )
endif()

# Set variables
set(PROJECT_NAME YORO)
set(PROJECT_LANGUAGE CXX)

# Environment variables
file(RELATIVE_PATH REL_INSTALL_PREFIX
    "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/yoro"
    )
set(CMAKE_INSTALL_PREFIX "${REL_INSTALL_PREFIX}" CACHE PATH "Installation directory")
option(BUILD_TEST "Enable building test" OFF)
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(BUILD_PYTHON_MODULE "Build python module (PyTorch extension)" ON)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Build configure")
if(CMAKE_BUILD_TYPE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
else()
    set(CMAKE_BUILD_TYPE Release)
endif()

# Export compile commands (For some useful tools like YCM, Sourcetrail, ...)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Initialize RPATH settings for shared library loading
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_RPATH_PREFIX "$ORIGIN")

# Library paths
set(LIB_PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}/yoro/api"
    )

# Utility paths
set(UTIL_PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}/utils"
    )

if(${BUILD_TEST})
    set(UTIL_PATHS ${UTIL_PATHS} "${CMAKE_CURRENT_SOURCE_DIR}/test")
endif()

# Set project
project(${PROJECT_NAME} ${PROJECT_LANGUAGE})

# Include cmake files
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/lib.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/util.cmake")
