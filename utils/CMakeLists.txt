cmake_minimum_required(VERSION 3.10)

# Set variables
set(PROJECT_NAME yoro_util)

# Set file list
file(GLOB PY_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.py"
    )

# Configure utility
foreach(PY_FILE ${PY_SRCS})

    # Get utility name
    get_filename_component(PY_NAME ${PY_FILE} NAME_WE)

    # Configure utility with python executable path
    set(PY_UTIL "${CMAKE_CURRENT_BINARY_DIR}/${PY_NAME}")
    configure_file(${PY_FILE} "${PY_UTIL}" @ONLY)

    # Install utility
    install(FILES ${PY_UTIL} DESTINATION "bin")

endforeach()
