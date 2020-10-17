cmake_minimum_required(VERSION 3.10)

# Set variables
set(PROJECT_NAME yoro_test)
set(PROJECT_LANGUAGE CXX)
set(PROJECT_DEPS
    yoro_api
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    )

# Compile setting
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Set project
project(${PROJECT_NAME} ${PROJECT_LANGUAGE})

# Set file list
file(GLOB PROJECT_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Build test
foreach(TEST_FILE_PATH ${PROJECT_SRCS})
    # Get each file name
    get_filename_component(TEST_FILE_NAME ${TEST_FILE_PATH} NAME_WE)

    # Build executable
    add_executable(${TEST_FILE_NAME} ${TEST_FILE_PATH})
    set_target_properties(${TEST_FILE_NAME} PROPERTIES
        CXX_STANDARD 14
        OUTPUT_NAME ${TEST_FILE_NAME}
        )
    target_link_libraries(${TEST_FILE_NAME} ${PROJECT_DEPS})

endforeach()

