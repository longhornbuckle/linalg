cmake_minimum_required( VERSION 3.14 )
project( Playground VERSION 0.1.0 )
include( CTest )
set( PROJECT_SOURCE_DIR CMAKE_CURRENT_LIST_DIR )
set( CMAKE_C_COMPILER /usr/local/bin/gcc-12 )
set( CMAKE_C++_COMPILER /usr/local/bin/g++-12 )
add_compile_options( -fconcepts-diagnostics-depth=2 )
include_directories( ${PROJECT_SOURCE_DIR} )
include_directories( ${PROJECT_SOURCE_DIR}/experimental ) #include experimental STL
include( FetchContent )
FetchContent_Declare( googletest URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip )
# For Windows: Prevent overriding the parent project's compiler/linker settings
set( gtest_force_shared_crt ON CACHE BOOL "" FORCE )
FetchContent_MakeAvailable( googletest )
enable_testing()
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
