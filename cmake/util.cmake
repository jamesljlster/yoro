# Project Utility Configuration

# Add subdirectiories
foreach(UTIL_PATH ${UTIL_PATHS})
    add_subdirectory(${UTIL_PATH})
endforeach()
