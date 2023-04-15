//==================================================================================================
//  File:       forward_declarations.hpp
//
//  Summary:    This header forward declares the primary linear algebra classes.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_FORWARD_DECLARATIONS_HPP
#define LINEAR_ALGEBRA_FORWARD_DECLARATIONS_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{

// Default layout
using default_layout = ::std::experimental::layout_right;

// Dynamic-size, dynamic-capacity tensor
template < class  T,
           size_t R,
           class  Alloc  = ::std::allocator<T>,
           class  L      = default_layout,
           class  Access = ::std::experimental::default_accessor<T> >
class dr_tensor;

// Dynamic-size, dynamic-capacity matrix
template < class T,
           class Alloc  = ::std::allocator<T>,
           class L      = default_layout,
           class Access = ::std::experimental::default_accessor<T> >
class dr_matrix;

// Dynamic-size, dynamic-capacity vector
template < class T,
           class Alloc  = ::std::allocator<T>,
           class L      = default_layout,
           class Access = ::std::experimental::default_accessor<T> >
class dr_vector;

// Fixed-size, fixed-capacity tensor
template < class             T,
           class             L,
           class             A,
           ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... ) // Each dimension must be >= 0
#endif
class fs_tensor;

// Fixed-size, fixed-capacity matrix
template < class         T,
           ::std::size_t R,
           ::std::size_t C,
           class         L = default_layout,
           class         A = ::std::experimental::default_accessor<T>
#ifdef LINALG_ENABLE_CONCEPTS
           > requires ( ( R >= 0 ) && ( C >= 0 ) ) // Row and column must be >= 0
#else
           , typename = ::std::enable_if_t< ( ( R >= 0 ) && ( C >= 0 ) ) > >
#endif
class fs_matrix;

// Fixed-size, fixed-capacity vector
template < class         T,
           ::std::size_t N,
           class         L = default_layout,
           class         A = ::std::experimental::default_accessor<T>
#ifdef LINALG_ENABLE_CONCEPTS
           > requires ( N >= 0 ) // Number of elements must be >= 0
#else
           , typename = ::std::enable_if_t< ( N >= 0 ) > >
#endif
class fs_vector;

// Non-owning tensor view
template < class MDS
#ifdef LINALG_ENABLE_CONCEPTS
  > requires ( detail::is_mdspan_v<MDS> &&
             MDS::is_always_unique() ) // Each element in the mdspan must have a unique mapping. (i.e. span_type and const_underlying_span_type should be the same.)
#else
  , typename = ::std::enable_if_t< ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() ) > >
#endif
class tensor_view;

// Non-owning matrix view
template < class MDS
#ifdef LINALG_ENABLE_CONCEPTS
  > requires ( detail::is_mdspan_v<MDS> &&
               ( MDS::extents_type::rank()== 2 ) &&
               MDS::is_always_unique() ) // Each element in the mdspan must have a unique mapping. (i.e. span_type and const_underlying_span_type should be the same.)
#else
  , typename = ::std::enable_if_t< ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank()== 2 ) && MDS::is_always_unique() ) > >
#endif
class matrix_view;

// Non-owning vector view
template < class MDS
#ifdef LINALG_ENABLE_CONCEPTS
  > requires ( detail::is_mdspan_v<MDS> &&
               ( MDS::extents_type::rank()== 1 ) &&
               MDS::is_always_unique() ) // Each element in the mdspan must have a unique mapping. (i.e. span_type and const_underlying_span_type should be the same.)
#else
  , typename = enable_if_t< ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank()== 1 ) && MDS::is_always_unique() ) > >
#endif
class vector_view;

}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_FORWARD_DECLARATIONS_HPP
