cmake_minimum_required(VERSION 3.20)

project(
    dbow3_bindings
    LANGUAGES CXX
)

set(
    CMAKE_CXX_STANDARD
    17
)

set(
    CMAKE_CXX_STANDARD_REQUIRED
    ON
)


# ----------------------------------------------------------------------------------------------------------------------
# User-configurable DBoW3 path
# ----------------------------------------------------------------------------------------------------------------------

set(
    DBOW3_ROOT
    ""
    CACHE PATH
    "Path to the DBoW3 source directory"
)

if(
    NOT DBOW3_ROOT
)

    message(
        FATAL_ERROR
        "DBOW3_ROOT is not set. "
        "Configure CMake with -DDBOW3_ROOT=<path-to-DBoW3>."
    )

endif()


set(
    DBOW3_INCLUDE_DIR
    "${DBOW3_ROOT}/src"
)

set(
    DBOW3_LIBRARY
    "${DBOW3_ROOT}/build/bin/Release/DBoW3001.lib"
)

set(
    DBOW3_DLL
    "${DBOW3_ROOT}/build/bin/Release/DBoW3001.dll"
)


if(
    NOT EXISTS "${DBOW3_INCLUDE_DIR}/DBoW3.h"
)

    message(
        FATAL_ERROR
        "DBoW3 headers were not found in: ${DBOW3_INCLUDE_DIR}"
    )

endif()


if(
    NOT EXISTS "${DBOW3_LIBRARY}"
)

    message(
        FATAL_ERROR
        "DBoW3 library was not found: ${DBOW3_LIBRARY}"
    )

endif()


if(
    NOT EXISTS "${DBOW3_DLL}"
)

    message(
        FATAL_ERROR
        "DBoW3 DLL was not found: ${DBOW3_DLL}"
    )

endif()


# ----------------------------------------------------------------------------------------------------------------------
# Python
# ----------------------------------------------------------------------------------------------------------------------

find_package(
    Python
    COMPONENTS Interpreter Development
    REQUIRED
)


# ----------------------------------------------------------------------------------------------------------------------
# pybind11
# ----------------------------------------------------------------------------------------------------------------------

execute_process(
    COMMAND
        "${Python_EXECUTABLE}"
        -m
        pybind11
        --cmakedir

    OUTPUT_VARIABLE
        pybind11_DIR

    OUTPUT_STRIP_TRAILING_WHITESPACE

    COMMAND_ERROR_IS_FATAL
        ANY
)


find_package(
    pybind11
    CONFIG
    REQUIRED
)


# ----------------------------------------------------------------------------------------------------------------------
# OpenCV
# ----------------------------------------------------------------------------------------------------------------------

find_package(
    OpenCV
    REQUIRED
)


# ----------------------------------------------------------------------------------------------------------------------
# Python extension module
# ----------------------------------------------------------------------------------------------------------------------

pybind11_add_module(
    dbow3_bindings
    dbow3_bindings.cpp
)


target_include_directories(
    dbow3_bindings
    PRIVATE

    "${DBOW3_INCLUDE_DIR}"
)


target_link_libraries(
    dbow3_bindings
    PRIVATE

    ${OpenCV_LIBS}
    "${DBOW3_LIBRARY}"
)


# ----------------------------------------------------------------------------------------------------------------------
# Runtime DBoW3 DLL
# ----------------------------------------------------------------------------------------------------------------------

add_custom_command(
    TARGET dbow3_bindings
    POST_BUILD

    COMMAND
        ${CMAKE_COMMAND}
        -E copy_if_different

        "${DBOW3_DLL}"

        $<TARGET_FILE_DIR:dbow3_bindings>
)