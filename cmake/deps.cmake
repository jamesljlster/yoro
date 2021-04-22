# Project Dependences Configuration

# Find OpenCV
set(OpenCV_STATIC OFF CACHE BOOL "Using OpenCV static linking library")
find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_LESS "4.0.0")
    message(FATAL_ERROR "Error: OpenCV 4.0.0+ is required")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

# Find PyTorch
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

get_filename_component(TORCH_LIB_PATH "${Torch_DIR}/../../../lib" ABSOLUTE)
link_directories(${TORCH_LIB_PATH})

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Include subdirectories
include_directories(${DEPS_PATHS})

# Backup and set build type to release
if(NOT MSVC)
    set(CMAKE_BUILD_TYPE_BAK ${CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add subdirectory
foreach(DEPS_PATH ${DEPS_PATHS})
    add_subdirectory(${DEPS_PATH})
endforeach()

# Restore origin build type
if(NOT MSVC)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_BAK})
endif()

