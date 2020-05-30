cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# Set variables
set(PROJECT_NAME yoro_api)
set(MODULE_NAME ${PROJECT_NAME}_pym)
set(PROJECT_LANGUAGE CXX)
if(${WITH_CUDA})
    set(PROJECT_LANGUAGE ${PROJECT_LANGUAGE} CUDA)
endif()

set(PROJECT_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/yoro_api.hpp)

# Compile setting
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    --compiler-options \"${CMAKE_CXX_FLAGS}\" --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")

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
file(GLOB CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
file(GLOB MODULE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx)

if(${WITH_CUDA})
    set(PROJECT_SRCS ${PROJECT_SRCS} ${CUDA_SRCS})
endif()

# Build library
add_library(${PROJECT_NAME} ${PROJECT_SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 14
    OUTPUT_NAME ${PROJECT_NAME}
    PREFIX "lib"
    )

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

# Build PyTorch module
add_library(${MODULE_NAME} SHARED ${MODULE_SRCS})
set_target_properties(${MODULE_NAME} PROPERTIES
    OUTPUT_NAME ${MODULE_NAME}
    PREFIX ""
    )

target_compile_definitions(${MODULE_NAME} PRIVATE
    -DTORCH_EXTENSION_NAME=${PROJECT_NAME}
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
