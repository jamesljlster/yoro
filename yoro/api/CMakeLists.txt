cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# Set variables
set(PROJECT_NAME yoro_api)
set(MODULE_NAME ${PROJECT_NAME}_pym)
set(PROJECT_LANGUAGE CXX)
set(PROJECT_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/yoro_api.hpp)

# Compile setting
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O")

# Set default build option
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS OFF)
endif()

# Set project
project(${PROJECT_NAME} ${PROJECT_LANGUAGE})

# Include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Set file list
file(GLOB PROJECT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB MODULE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx)

# Build library
add_library(${PROJECT_NAME} ${PROJECT_SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 14
    OUTPUT_NAME ${PROJECT_NAME}
    PREFIX "lib"
    )

target_link_libraries(${PROJECT_NAME}
    PUBLIC ${OpenCV_LIBS}
    PRIVATE ${TORCH_LIBRARIES}
    )

# Build PyTorch module
add_library(${MODULE_NAME} SHARED ${MODULE_SRCS})
set_target_properties(${MODULE_NAME} PROPERTIES
    OUTPUT_NAME ${MODULE_NAME}
    PREFIX ""
    )

target_compile_definitions(${MODULE_NAME} PRIVATE
    -DTORCH_EXTENSION_NAME=${MODULE_NAME}
    -DTORCH_API_INCLUDE_EXTENSION_H
    )

target_link_libraries(${MODULE_NAME} PRIVATE
    ${PROJECT_NAME}
    torch_python
    )

# Install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
    )
install(FILES ${PROJECT_HEADERS}
    DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
    )

install(TARGETS ${MODULE_NAME} DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
