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
#  if ( __cpp_lib_execution >= 201603L ) && ( ( LINALG_COMPILER_GNU >= 9 ) || ( LINALG_COMPILER_MSVC >= 1914 ) )
#    define LINALG_EXECTUION_POLICY 1
#  else
#    define LINALG_EXECUTION_POLICY 0
#  endif
#endif

// Define execution::seq if available.
#ifndef LINALG_EXECUTION_SEQ
#  if LINALG_EXECTUION_POLICY
#    define LINALG_EXECUTION_SEQ execution::seq
#  else
#    define LINALG_EXECUTION_SEQ 0
#  endif
#endif

// Define execution::unseq if available.
// If not, then just use execution::seq instead.
#ifndef LINALG_EXECUTION_UNSEQ
#  if ( __cpp_lib_execution >= 201902L ) && ( ( LINALG_COMPILER_GNU >= 9 ) || ( LINALG_COMPILER_MSVC >= 1928 ) )
#    define LINALG_EXECUTION_UNSEQ execution::unseq
#  else
#    define LINALG_EXECUTION_UNSEQ LINALG_EXECUTION_SEQ
#  endif
#endif

// Force compiler to inline function
#ifndef LINALG_FORCE_INLINE_FUNCTION
#  ifdef LINALG_COMPILER_MSVC
#    define LINALG_FORCE_INLINE_FUNCTION __forceinline
#  else
#    define LINALG_FORCE_INLINE_FUNCTION __attribute__((always_inline))
#  endif
#endif

// Support for concepts
#ifndef LINALG_ENABLE_CONCEPTS
#  if ( __cpp_lib_concepts >= 201907L ) && ( ( LINALG_COMPILER_GNU >= 10 ) || ( LINALG_COMPILER_CLANG >= 16 ) )
#    define LINALG_ENABLE_CONCEPTS
#  endif
#endif

#endif  //- LINEAR_ALGEBRA_MACROS_HPP
