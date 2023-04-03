#ifndef LINEAR_ALGEBRA_MATRIX_CONCEPTS_HPP
#define LINEAR_ALGEBRA_MATRIX_CONCEPTS_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{
namespace concepts
{

//=================================================================================================
//  Matrix Concepts
//=================================================================================================

// All matrix data types must satisfy tensor requirements
// All matrix data types must be a rank 2 tensor
// All matrix data types must support size / capacity const access functions
template < class M >
concept matrix_data = tensor_data<M> &&
( M::extents_type::rank() == 2 ) &&
requires( M& m ) // Functions
{
  { m.columns() }         noexcept -> same_as< typename M::size_type >;
  { m.rows() }            noexcept -> same_as< typename M::size_type >;
  { m.column_capacity() } noexcept -> same_as< typename M::size_type >;
  { m.row_capacity() }    noexcept -> same_as< typename M::size_type >;
};

// Matrix must be a matrix data
// Matrix must be a tensor
// All matrix data types must support unary functions: trans and conj
// All matrix data types must support binary functions: multiplication
template < class M >
concept matrix = matrix_data<M> &&
tensor<M> &&
requires ( M m ) // Unary operations
{
  trans( m );
  conj( m );
} &&
requires ( M m ) // Binary operations
{
  m * trans( m );
};

// Readable matrix data must A) be a matrix data
// Readable matrix data must B) be a readable tensor data
// Readable matrix data must C) provide index access
// Readable matrix data must D) provide a two dimension const span representation
// Readable matrix data must D) provide an implementation defined view of the underlying memory.
// Readable matrix data must E) provide a const row view
// Readable matrix data must F) provide a const column view
// Readable matrix data must G) provide a const submatrix view
template < class M >
concept readable_matrix_data = matrix_data<M> &&                                     // A)
readable_tensor_data<M> &&                                                           // B)
requires( const M& m, typename M::index_type index1, typename M::index_type index2 ) // C)
{
  { m[index1,index2] }       noexcept -> same_as<typename M::value_type>;
  { m.at(index1,index2) }             -> same_as<typename M::value_type>;
} &&
( M::span_type::extents_type::rank() == 2 ) &&                                       // D)
requires( const M& m, typename M::index_type index )                                 // E)
{
  typename M::const_row_type;
  { m.row(index) }                    -> same_as<typename M::const_row_type>;
} &&
readable_vector_data<typename M::const_row_type> &&
requires( const M& m, typename M::index_type index )                                 // F)
{
  typename M::const_column_type;
  { m.column(index) }                 -> same_as<typename M::const_column_type>;
} &&
readable_vector_data<typename M::const_column_type> &&
requires( const M&               m,
          typename M::tuple_type start,
          typename M::tuple_type end )                                               // G)
{
  typename M::const_submatrix_type;
  { m.submatrix(start,end) }          -> same_as<typename M::const_submatrix_type>;
  { m.submatrix(start,end) }          -> matrix;
};

// Readable matrix must be a readable matrix data
// Readable matrix must be a matrix
template < class M >
concept readable_matrix = readable_matrix_data<M> && matrix<M>;

// Writable matrix data must A) be a readable matrix
// Writable matrix data must B) be a writable tensor
// Writable matrix data must C) provide a mutable scalar view
// Writable matrix data must D) provide a mutable row view
// Writable matrix data must E) provide a mutable column view
// Writable matrix data must F) provide a mutable submatrix view
template < class M >
concept writable_matrix_data = readable_matrix_data<M> &&                      // A)
writable_tensor_data<M> &&                                                     // B)
requires( M& m, typename M::index_type index1, typename M::index_type index2 ) // C)
{
  typename M::reference_type;
  { m[index1,index2] }       noexcept -> same_as<typename M::reference_type>;
  { m.at(index1,index2) }             -> same_as<typename M::reference_type>;
} &&
requires( M& m, typename M::index_type index )                                 // D)
{
  typename M::row_type;
  { m.row(index) }                    -> same_as<typename M::row_type>;
} &&
writable_vector_data<typename M::row_type> &&
requires( M& m, typename M::index_type index )                                 // E)
{
  typename M::column_type;
  { m.column(index) }                 -> same_as<typename M::column_type>;
} &&
writable_vector_data<typename M::column_type> &&
requires( M& m,
          tuple<typename M::index_type,typename M::index_type> start,
          tuple<typename M::index_type,typename M::index_type> end )           // F)
{
  typename M::submatrix_type;
  { m.submatrix(start,end) }          -> same_as<typename M::submatrix_type>;
  { m.submatrix(start,end) }          -> matrix;
};

// Writable matrix must be a writable matrix data
// Writable matrix must be a readable matrix
template < class M >
concept writable_matrix = writable_matrix_data<M> && readable_matrix<M>;

// Dynamic matrix data must be a matrix data
// Dynamic matrix data must be a dynamic tensor data
template < class M >
concept dynamic_matrix_data = matrix_data<M> &&
dynamic_tensor_data<M>;

// Dynamic matrix must be a dynamic matrix data
// Dynamic matrix must be a matrix
template < class M >
concept dynamic_matrix = dynamic_matrix_data<M> &&
matrix<M>;

// Fixed size matrix data must be a matrix data
// Fixed size matrix data must be a fixed size tensor data
// A fixed size matrix data must support constant expression size functions.
// A fixed size matrix data must support constant expression capacity functions.
// A fixed size matrix data must have equivalent size and capacity.
template < class M >
concept fixed_size_matrix_data = matrix_data<M> &&
fixed_size_tensor_data<M> &&
detail::is_constexpr( []{ decltype( declval<M>().rows() )            nodiscard_warning = M().rows(); } ) &&
detail::is_constexpr( []{ decltype( declval<M>().columns() )         nodiscard_warning = M().columns(); } ) &&
detail::is_constexpr( []{ decltype( declval<M>().row_capacity() )    nodiscard_warning = M().row_capacity(); } ) &&
detail::is_constexpr( []{ decltype( declval<M>().column_capacity() ) nodiscard_warning = M().column_capacity(); } ) &&
( M().rows()    == M().row_capacity() ) &&
( M().columns() == M().column_capacity() );

// Fixed size matrix must be a fixed size matrix data
// Fixed size matrix must be a matrix
template < class M >
concept fixed_size_matrix = fixed_size_matrix_data<M> &&
matrix<M>;

//=================================================================================================
//  Additional Matrix Concepts
//=================================================================================================

// Matrix may be convertible
// Enforces both types are matrix which have convertible elements
// and both rows / columns must be the same size or dynamic
template < class From, class To >
concept matrix_may_be_convertible =
matrix<From> && matrix<To> &&
( is_convertible_v< typename From::value_type, typename To::value_type > ) &&
( ( From::extents_type::static_extent(0) == To::extents_type::static_extent(0) ) ||
  ( From::extents_type::static_extent(0) == std::experimental::dynamic_extent ) ||
  ( To::extents_type::static_extent(0) == std::experimental::dynamic_extent ) ) &&
( ( From::extents_type::static_extent(1) == To::extents_type::static_extent(1) ) ||
  ( From::extents_type::static_extent(1) == std::experimental::dynamic_extent ) ||
  ( To::extents_type::static_extent(1) == std::experimental::dynamic_extent ) );

// View may be convertible to matrix
// Enforces view is an mdspan of rank 2 with elements convertible to
// the matrix elements and a view which may be same size as the matrix
template < class From, class To >
concept view_may_be_convertible_to_matrix =
detail::is_mdspan_v<From> && matrix<To> &&
( From::rank() == 2 ) &&
matrix<To> &&
( is_convertible_v<typename From::value_type,typename To::value_type> ) &&
( ( From::extents_type::static_extent(0) == To::extents_type::static_extent(0) ) ||
  ( From::extents_type::static_extent(0) == std::experimental::dynamic_extent ) ||
  ( To::extents_type::static_extent(0) == std::experimental::dynamic_extent ) ) &&
( ( From::extents_type::static_extent(1) == To::extents_type::static_extent(1) ) ||
  ( From::extents_type::static_extent(1) == std::experimental::dynamic_extent ) ||
  ( To::extents_type::static_extent(1) == std::experimental::dynamic_extent ) );

}       //- concepts namespace
}       //- math namespace
}       //- std namespace

#endif  //- LINEAR_ALGEBRA_MATRIX_CONCEPTS_HPP_DEFINED
