//==================================================================================================
//  File:       config.hpp
//
//  Summary:    This header defines macros for determining various feature support of the build
//              system.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_CONFIG_HPP
#define LINEAR_ALGEBRA_CONFIG_HPP

// Macro for determining if header may be included
// If not already supported, just set to false
#ifndef __has_include
#  define __has_include( x ) 0
#endif

// Get c++ version information
#if __has_include( <version> )
#  include <version>
#else
#  include <type_traits>
#  include <utility>
#endif

// Use of __cplusplus is less reliable with MSVC is less reliable so use separate macro
#ifdef _MSVC_LANG
#  define LINALG_CPLUSPLUS _MSVC_LANG
#else
#  define LINALG_CPLUSPLUS __cplusplus
#endif

// Check C++ version
// Note: no standardized version number for C++23 yet.
// Use of C++23 features will have to be individually tested for
#define LINALG_CXX_STD_17 201703L
#define LINALG_CXX_STD_20 202002L

#define LINALG_HAS_CXX_17 ( LINALG_CPLUSPLUS >= LINALG_CXX_STD_17 )
#define LINALG_HAS_CXX_20 ( LINALG_CPLUSPLUS >= LINALG_CXX_STD_20 )

// Define if clang compiler is being used
#ifndef LINALG_COMPILER_CLANG
#  if defined( __clang__ )
#    define LINALG_COMPILER_CLANG __clang__
#  endif
#endif

// Define if gnu compiler is being used
#if !defined( LINALG_COMPILER_GNU ) && !defined( LINALG_COMPILER_CLANG )
#  if defined( __GNUC__ )
#    define LINALG_COMPILER_GNU __GNUC__
#  endif
#endif

// Define if Intel compiler is being used
#ifndef LINALG_COMPILER_INTEL
#  if defined( __INTEL_COMPILER )
#    define LINALG_COMPILER_INTEL __INTEL_COMPILER
#  endif
#endif

// Define if Microsoft Visual Studio compiler is being used
#if !defined( LINALG_COMPILER_MSVC ) && !defined( LINALG_COMPILER_INTEL )
#  if defined( _MSC_VER )
#    define _MDSPAN_COMPILER_MSVC _MSC_VER
#  endif
#endif

// Macro for determining if particular attributes are supported.
// If not already supported, just set to false
#ifndef __has_cpp_attribute
#  define __has_cpp_attribute(x) 0
#endif

#endif  //- LINEAR_ALGEBRA_CONFIG_HPP
