# Project Library Configuration

# Include subdirectories
include_directories(${LIB_PATHS})

# Add subdirectiories
foreach(LIB_PATH ${LIB_PATHS})
    add_subdirectory(${LIB_PATH})
endforeach()
