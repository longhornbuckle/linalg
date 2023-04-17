cmake_minimum_required( VERSION 3.15 )
project( LIN_ALG VERSION 0.1.0 LANGUAGES CXX )

include( GNUInstallDirs )

list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

add_library( linalg INTERFACE )

target_include_directories( linalg INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/externals/mdspan/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

################################################################################

install( TARGETS linalg EXPORT linalgTargets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install( EXPORT linalgTargets
    FILE linalgTargets.cmake
    NAMESPACE std::experimental::math::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/linalg
)

export( TARGETS linalg
    NAMESPACE std::experimental::math::
    FILE linalgTargets.cmake
)

install( DIRECTORY include/experimental DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )

include( CMakePackageConfigHelpers )
configure_package_config_file( cmake/linalgConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/linalgConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/linalg
)
write_basic_package_version_file( ${CMAKE_CURRENT_BINARY_DIR}/linalgConfigVersion.cmake
  COMPATIBILITY SameMajorVersion
  ARCH_INDEPENDENT
)

install( FILES ${CMAKE_CURRENT_BINARY_DIR}/linalgConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/linalgConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/linalg
)

################################################################################

enable_testing()
add_subdirectory(tests)
