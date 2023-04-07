//==================================================================================================
//  File:       config.hpp
//
//  Summary:    This header defines macros for defining behavior in the presence of or in the
//              absence of particular feature support
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_MACROS_HPP
#define LINEAR_ALGEBRA_MACROS_HPP

// Define if STL execution policies are supported.
#ifndef LINALG_EXECUTION_POLICY
#  if ( __cpp_lib_execution >= 201603L )
#    define LINALG_EXECTUION_POLICY
#  else
#    define LINALG_EXECUTION_POLICY 0
#  endif
#endif

// Define execution::unseq if available.
// If not, then just use execution::seq instead.
#ifndef LINALG_EXECUTION_UNSEQ
#  if ( __cpp_lib_execution >= 201902L ) && ( ( LINALG_COMPILER_GNU >= 9 ) || ( LINALG_COMPILER_MSVC >= 1928 ) )
#    define LINALG_EXECUTION_UNSEQ execution::unseq
#  elif ( LINALG_EXECUTION_POLICY )
#    define LINALG_EXECUTION_UNSEQ execution::seq
#  else
#    define LINALG_EXECUTION_UNSEQ 0
#  endif
#endif

#endif  //- LINEAR_ALGEBRA_MACROS_HPP
