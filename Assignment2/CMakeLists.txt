# CMake entry point
cmake_minimum_required(VERSION 3.10)

include(CMakePrintHelpers)
project(VC_IntroOpenGL)

cmake_print_variables(CMAKE_PREFIX_PATH)
cmake_print_variables(CMAKE_SOURCE_DIR)

# --- Dependencies ---
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${GLM_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    "external"
    ${GLFW_INCLUDE_DIRS}
    .
)

# Use experimental glm features
add_definitions(-DGLM_ENABLE_EXPERIMENTAL)

set(ALL_LIBS
    ${OPENGL_LIBRARY}
    glfw
    ${OpenCV_LIBS}
)

add_definitions(
    -DTW_STATIC
    -DTW_NO_LIB_PRAGMA
    -DTW_NO_DIRECT3D
    -DGLEW_STATIC
    -D_CRT_SECURE_NO_WARNINGS
)



# --------------------------------------------------------------------------
# Assignment 2 - OpenCV camera feed on textured quad
# --------------------------------------------------------------------------
file(GLOB_RECURSE COMMON_SOURCES "common/*.cpp" "common/*.hpp")

add_executable(Assignment2
    ${COMMON_SOURCES}
    source/webcamQuad.cpp
)

target_link_libraries(Assignment2
    ${ALL_LIBS}
)

# --------------------------------------------------------------------------
# Automatically copy shaders from src/ to the executable folder
# --------------------------------------------------------------------------
file(GLOB SHADERS "source/*.vert" "source/*.frag")
foreach(SHADER ${SHADERS})
    add_custom_command(TARGET Assignment2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SHADER}
        $<TARGET_FILE_DIR:Assignment2>
    )
endforeach()

# --------------------------------------------------------------------------
# Source grouping for IDE organization
# --------------------------------------------------------------------------
SOURCE_GROUP(common REGULAR_EXPRESSION ".*/common/.*")
SOURCE_GROUP(shaders REGULAR_EXPRESSION ".*/.*shader$")
