macro( linalg_add_test name )
  if( LINALG_TEST_LANGUAGE )
    set_source_files_properties( ${name} PROPERTIES LANGUAGE ${LINALG_TEST_LANGUAGE} )
  endif()
  set_property( TARGET ${name} PROPERTY CXX_STANDARD 23 ) # set c++ version
  add_executable( ${name} ${name}.cpp )
  target_link_libraries( ${name} linalg gtest_main )
  add_test( ${name} ${name} )
endmacro()

if( LINALG_USE_SYSTEM_GTEST )
  find_package( GTest CONFIG REQUIRED )
  add_library( gtest_main ALIAS GTest::gtest_main )
else()
  # adapted from https://github.com/google/googletest/blob/master/googletest/README.md
  configure_file( ${PROJECT_SOURCE_DIR}/cmake/googletest/CMakeLists.txt.in googletest-download/CMakeLists.txt )
  execute_process( COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
  )
  if( result )
    message( FATAL_ERROR "CMake step for googletest failed: ${result}" )
  endif()
  execute_process( COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
  )
  if( result )
    message( FATAL_ERROR "Build step for googletest failed: ${result}" )
  endif()

  # Prevent overriding the parent project's compiler/linker
  # settings on Windows
  set( gtest_force_shared_crt ON CACHE BOOL "" FORCE )

  # Add googletest directly to our build. This defines
  # the gtest and gtest_main targets.
  add_subdirectory( ${CMAKE_CURRENT_BINARY_DIR}/googletest-src
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
    EXCLUDE_FROM_ALL
  )
endif()

# Add vector / matrix / tensor tests
linalg_add_test(test_vector)
linalg_add_test(test_matrix)
linalg_add_test(test_tensor)
