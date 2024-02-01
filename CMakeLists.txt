cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(TRIPS VERSION 1.0.0 LANGUAGES C CXX CUDA)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/External/saiga/cmake/")

include(helper_macros)
include(ExternalProject)
DefaultBuildType(RelWithDebInfo)
message("Build Options")

OptionsHelper(HEADLESS "Skips the viewer and other apps that require a window." OFF)
OptionsHelper(ASAN "Adds sanitize=address compiler flag" OFF)
OptionsHelper(TINY_EIGEN "Use saiga's tiny eigen library." ON)
OptionsHelper(LIVE_TRAIN_VIEWER "Adds a GL window to training, shows live training progress (only in non-headless mode)." OFF)
OptionsHelper(USE_VET "Adds VET: Visual Error Tomography functionality." OFF)
OptionsHelper(MINIMAL_GUI "Minimizes the gui." ON)


if (HEADLESS)
    set(SAIGA_MODULE_OPENGL OFF)
    set(SAIGA_BUILD_GLFW OFF)

endif ()




############# Required LIBRARIES ###############

if (UNIX)
    add_subdirectory(External/tensorboard_logger)
    set_property(TARGET tensorboard_logger PROPERTY POSITION_INDEPENDENT_CODE ON)
    PackageHelperTarget(tensorboard_logger TBL_FOUND)
    add_compile_definitions(TBLOGGER)
endif ()

# Saiga
set(SAIGA_BUILD_SAMPLES OFF)
set(SAIGA_BUILD_TESTS OFF)
set(SAIGA_MODULE_VULKAN OFF)
set(SAIGA_MODULE_VISION OFF)
if (HEADLESS)
    set(SAIGA_MODULE_OPENGL OFF)
    set(SAIGA_BUILD_GLFW OFF)
    message("HEADLESS MODE")
endif ()
set(SAIGA_NO_INSTALL ON)
set(SAIGA_USE_SUBMODULES ON)
set(SAIGA_WITH_FFMPEG OFF)
set(SAIGA_WITH_FREETYPE OFF)
set(SAIGA_WITH_YAMLCPP OFF)
set(SAIGA_WITH_OPENAL OFF)
set(SAIGA_WITH_MKL ON)
set(SAIGA_WITH_OPENMESH OFF)
set(SAIGA_WITH_OPENVR OFF)
set(SAIGA_WITH_G2O OFF)
set(SAIGA_WITH_CHOLMOD OFF)
set(SAIGA_WITH_OPENNI OFF)
set(SAIGA_WITH_K4A OFF)
set(SAIGA_WITH_TINY_EIGEN ON)

if (MSVC)
    #multiprocessor compilation for visual studio
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif ()

add_subdirectory(External/saiga)

PackageHelperTarget(saiga_core SAIGA_FOUND)
if (NOT HEADLESS)
    PackageHelperTarget(saiga_opengl SAIGA_FOUND)
endif ()
PackageHelperTarget(saiga_cuda SAIGA_FOUND)

set(CAFFE2_USE_CUDNN ON)
# Torch
find_package(Torch REQUIRED)
PackageHelperTarget(torch TORCH_FOUND)
message("CMAKE_EXE_LINKER_FLAGS before: " ${CMAKE_EXE_LINKER_FLAGS})
message("TORCH_LIBRARIES: " "${TORCH_LIBRARIES}")
message("TORCH_CXX_FLAGS: " "${TORCH_CXX_FLAGS}")

#replace flags
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")
message("CMAKE_EXE_LINKER_FLAGS after: " ${CMAKE_EXE_LINKER_FLAGS})


##this is broken!!!!
#set(CMAKE_EXE_LINKER_FLAGS "-Wl,--no-as-needed")

message("CUDNN_LIBRARY_PATH: " "${CUDNN_LIBRARY_PATH}" "; CUDNN_INCLUDE_PATH: " "${CUDNN_INCLUDE_PATH}")

add_subdirectory(External/tiny-cuda-nn)
PackageHelperTarget(tiny-cuda-nn tiny-cuda-nn_FOUND)

#openmp
#find_package(OpenMP REQUIRED)
#PackageHelperTarget(OpenMP::OpenMP_CXX OPENMP_FOUND)

include_directories(.)
include_directories(${PACKAGE_INCLUDES})

############# COMPILER FLAGS ###############
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread")

if (MSVC)
    #multiprocessor compilation for visual studio
    #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=return-type")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-backtrace-limit=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-psabi")
endif ()


message(STATUS CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
message(STATUS CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})

#did torch forget to also export cupti?
set(LIBS ${LIBS} ${LIB_TARGETS} "${TORCH_LIBRARIES}" ${CUDA_cupti_LIBRARY} ${CUDNN_LIBRARY_PATH})

############# C++ Standard and Filesystem stuff ###############

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    #dll has all symbols, to avoid __declspecs everywhere
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    include(GenerateExportHeader)
endif ()
############# SOURCE ###############


if (HEADLESS)
    message("headless mode")
else ()
    if (LIVE_TRAIN_VIEWER)
        add_compile_definitions(LIVE_VIEWER)
    endif ()
endif ()

if (USE_VET)
    add_compile_definitions(COMPILE_WITH_VET)
endif ()

if (MINIMAL_GUI)
    add_compile_definitions(MINIMAL_GUI)
endif ()


add_subdirectory(src)

if (USE_VET)
    if (CMAKE_SYSTEM_NAME MATCHES "Linux|Darwin")
        set(NEAT_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
        string(REPLACE ";" "|" CMAKE_PREFIX_PATH_ALT_SEP "${NEAT_CMAKE_PREFIX_PATH}")
        if (CMAKE_GENERATOR MATCHES "Ninja")
            ExternalProject_Add(NeAT
                    SOURCE_DIR ../External/NeAT
                    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/NeAT
                    LIST_SEPARATOR |
                    CMAKE_ARGS -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH_ALT_SEP}
                    INSTALL_COMMAND echo "Skipping install step.")
        else ()
            ExternalProject_Add(NeAT
                    SOURCE_DIR ../External/NeAT
                    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/NeAT
                    LIST_SEPARATOR |
                    CMAKE_ARGS -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH_ALT_SEP}
                    BUILD_COMMAND make -j20
                    INSTALL_COMMAND echo "Skipping install step.")
        endif ()
    endif ()
endif ()