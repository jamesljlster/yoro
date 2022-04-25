cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# Set library name
set(PROJECT_NAME yoro_api)
set(MODULE_NAME ${PROJECT_NAME}_pym)

# Set library public headers
set(PROJECT_CONFIG_HEADER_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/yoro_api_config.h.in"
    )
set(PROJECT_CONFIG_HEADER_DESTINATION
    "${CMAKE_CURRENT_SOURCE_DIR}/yoro_api_config.h"
    )
set(PROJECT_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/yoro_api.hpp"
    "${PROJECT_CONFIG_HEADER_DESTINATION}"
    )

# Compile setting
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -fopenmp -Wall -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O")

# Set default build option
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

# Set project
project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES ${PROJECT_LANGUAGE})

# Include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Set file list
file(GLOB PROJECT_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB MODULE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx")

# Configure package install prefix
set(LIB_DIR "lib/${PROJECT_NAME}")
set(INC_DIR "include/${PROJECT_NAME}")
set(EXPORT_DIR "lib/cmake/${PROJECT_NAME}")

# Configure header file
configure_file(${PROJECT_CONFIG_HEADER_SOURCE} ${PROJECT_CONFIG_HEADER_DESTINATION})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Build library
add_library(${PROJECT_NAME} ${PROJECT_SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 14
    OUTPUT_NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    INSTALL_RPATH "${TORCH_LIB_PATH}"
    )

target_link_libraries(${PROJECT_NAME}
    PUBLIC ${OpenCV_LIBS}
    PRIVATE ${TORCH_LIBRARIES}
    )

# Build python module
if(${BUILD_PYTHON_MODULE})
    get_filename_component(LIB_PATH "${CMAKE_INSTALL_PREFIX}/${LIB_DIR}" ABSOLUTE)
    file(RELATIVE_PATH PYM_INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_NAME}" "${LIB_PATH}")

    add_library(${MODULE_NAME} SHARED ${MODULE_SRCS})
    set_target_properties(${MODULE_NAME} PROPERTIES
        CXX_STANDARD 14
        OUTPUT_NAME ${MODULE_NAME}
        PREFIX ""
        INSTALL_RPATH "${CMAKE_RPATH_PREFIX}/${PYM_INSTALL_RPATH};${TORCH_LIB_PATH}"
        )

    target_compile_definitions(${MODULE_NAME} PRIVATE
        -DTORCH_EXTENSION_NAME=${MODULE_NAME}
        -DTORCH_API_INCLUDE_EXTENSION_H
        )

    target_link_libraries(${MODULE_NAME} PRIVATE
        ${PROJECT_NAME}
        torch_python
        )
endif()

# Install library files
set(LIB_TARGET ${PROJECT_NAME}-targets)
install(TARGETS ${PROJECT_NAME} DESTINATION ${LIB_DIR} EXPORT ${LIB_TARGET})
install(EXPORT ${LIB_TARGET} DESTINATION ${EXPORT_DIR})

install(FILES ${PROJECT_HEADERS} DESTINATION ${INC_DIR})

# Install python module
if(${BUILD_PYTHON_MODULE})
    install(TARGETS ${MODULE_NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Export package configuration
include(CMakePackageConfigHelpers)

set(LIB_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake")
set(LIB_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-version.cmake")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake.in" ${LIB_CONFIG_FILE}
    INSTALL_DESTINATION ${EXPORT_DIR}
    PATH_VARS INC_DIR LIB_DIR
    )

write_basic_package_version_file(
    ${LIB_VERSION_FILE}
    VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    COMPATIBILITY SameMajorVersion
)

install(FILES ${LIB_CONFIG_FILE} ${LIB_VERSION_FILE} DESTINATION ${EXPORT_DIR})
