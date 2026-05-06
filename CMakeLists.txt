cmake_minimum_required(VERSION 3.10)

project(memx_test_app LANGUAGES C CXX)

get_filename_component(app_name main NAME)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O0 -g -fopenmp")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -g -fopenmp")
else()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -fopenmp")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fopenmp")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

set(MEMX_SYSROOT $ENV{SDKTARGETSYSROOT})

if(NOT MEMX_SYSROOT)
  message(FATAL_ERROR "SDKTARGETSYSROOT is not set. Please source the SDK environment setup script first.")
endif()

set(MEMX_UTILS_INCLUDE_DIR
    "${MEMX_SYSROOT}/usr/src/debug/memx-runtime/2.2.0/mx_accl/include/utils"
    )

set(MEMX_ACCL_INCLUDE_DIR
    "${MEMX_SYSROOT}/usr/src/debug/memx-runtime/2.2.0/mx_accl/include"
)

set(MEMX_CORE_INCLUDE_DIR
    "${MEMX_SYSROOT}/usr/src/debug/memx-bench/1.0/include"
)

set(MEMX_LIBRARY
    "${MEMX_SYSROOT}/usr/lib/libmx_accl.so.2"
)

if(NOT EXISTS "${MEMX_ACCL_INCLUDE_DIR}/memx/accl/MxAccl.h")
  message(FATAL_ERROR "MemryX accl header not found at ${MEMX_ACCL_INCLUDE_DIR}/memx/accl/MxAccl.h")
endif()

if(NOT EXISTS "${MEMX_CORE_INCLUDE_DIR}/memx/memx.h")
  message(FATAL_ERROR "MemryX core header not found at ${MEMX_CORE_INCLUDE_DIR}/memx/memx.h")
endif()

if(NOT EXISTS "${MEMX_LIBRARY}")
  message(FATAL_ERROR "MemryX library not found at ${MEMX_LIBRARY}")
endif()

file(GLOB local_src
    "*.c"
    "*.cpp"
    "*.h"
	)

add_executable(${app_name} ${local_src})

target_include_directories(${app_name} PRIVATE
    ${MEMX_UTILS_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIRS}
    ${MEMX_ACCL_INCLUDE_DIR}
    ${MEMX_CORE_INCLUDE_DIR}
)

target_link_directories(${app_name} PRIVATE
    ${OPENCV_LIBRARY_DIRS}
)

target_compile_options(${app_name} PRIVATE
    ${OPENCV_CFLAGS_OTHER}
)

target_link_libraries(${app_name} PRIVATE
    ${MEMX_LIBRARY}
    ${OPENCV_LIBRARIES}
)
