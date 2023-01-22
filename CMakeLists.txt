cmake_minimum_required(VERSION 3.20.0)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

project(scho_complete)
find_package(OpenGL REQUIRED)

add_executable(${PROJECT_NAME} main.c)

target_include_directories(${PROJECT_NAME} PUBLIC ./headers_libs ./lib/glew_build/include ./OpenCL/include)
target_link_directories(${PROJECT_NAME} PUBLIC ./OpenCL/lib)

include_directories(${OPENGL_INCLUDE_DIRS})
add_subdirectory(./lib/glfw)


if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_subdirectory(./lib/glew_build/build/cmake)
    target_link_libraries(${PROJECT_NAME} PUBLIC glew)
    target_compile_options(${PROJECT_NAME} PUBLIC -O0)
    target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wextra -pedantic)
    
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    STRING (REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    target_link_directories(${PROJECT_NAME} PUBLIC ./lib/glew_bin/lib)
    target_include_directories(${PROJECT_NAME} PUBLIC ./lib/glew_bin/include)
    target_link_libraries(${PROJECT_NAME} PUBLIC glew32s)
    target_compile_definitions(${PROJECT_NAME} PUBLIC GLEW_STATIC)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC glfw OpenGL::GL OpenCL)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()


