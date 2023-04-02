cmake_minimum_required( VERSION 3.15 )
project( LIN_ALG VERSION 0.1.0 )
include( CTest )

# Project Directory
set( PROJECT_SOURCE_DIR /Users/longhornbuckle/Documents/linalg )

# Select Compilers
set( CMAKE_C_COMPILER /usr/bin/g++ )
set( CMAKE_C++_COMPILER /usr/bin/g++ )

set( CMAKE_VERBOSE_MAKEFILE ON )

# Compiler specific settings
if ( CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" )
  add_compile_options( -fpermissive )
elseif ( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
  add_compile_options( -fpermissive )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options( -fpermissive )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
endif()

# Google Test stuff
include_directories( ${PROJECT_SOURCE_DIR} )
include_directories( ${PROJECT_SOURCE_DIR}/experimental ) #include experimental STL
include( FetchContent )
FetchContent_Declare( gtest GIT_REPOSITORY https://github.com/google/googletest.git  GIT_TAG v1.13.0 )
# For Windows: Prevent overriding the parent project's compiler/linker settings
set( gtest_force_shared_crt ON CACHE BOOL "" FORCE )
FetchContent_MakeAvailable( gtest )
enable_testing()

# Vector / Matrix / Tensor Unit Tests
add_executable( vector_test tests/vector_test.cpp ) # add this executable
add_executable( matrix_test tests/matrix_test.cpp ) # add this executable
add_executable( tensor_test tests/tensor_test.cpp ) # add this executable
set_property( TARGET vector_test PROPERTY CXX_STANDARD 23 ) # set c++ version
target_link_libraries( vector_test GTest::gtest_main )
set_property( TARGET matrix_test PROPERTY CXX_STANDARD 23 ) # set c++ version
target_link_libraries( matrix_test GTest::gtest_main )
set_property( TARGET tensor_test PROPERTY CXX_STANDARD 23 ) # set c++ version
target_link_libraries( tensor_test GTest::gtest_main )
include( GoogleTest )
gtest_discover_tests( vector_test )
gtest_discover_tests( matrix_test )
gtest_discover_tests( tensor_test )

set( CPACK_PROJECT_NAME ${PROJECT_NAME} )
set( CPACK_PROJECT_VERSION ${PROJECT_VERSION} )
include( CPack )
