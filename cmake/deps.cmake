# Project Dependences Configuration

# Find OpenCV
set(OpenCV_STATIC OFF CACHE BOOL "Using OpenCV static linking library")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find PyTorch
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# Find Python
find_package(Python REQUIRED COMPONENTS Development)
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

