#ifndef LINEAR_ALGEBRA_MATRIX_CONCEPTS_HPP
#define LINEAR_ALGEBRA_MATRIX_CONCEPTS_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{
namespace concepts
{

#ifdef LINALG_ENABLE_CONCEPTS

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
  #if LINALG_USE_BRACKET_OPERATOR
  { m[index1,index2] }       noexcept -> same_as<typename M::value_type>;
  #endif
  #if LINALG_USE_PAREN_OPERATOR
  { m(index1,index2) }       noexcept -> same_as<typename M::value_type>;
  #endif
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
  #if LINALG_USE_BRACKET_OPERATOR
  { m[index1,index2] }       noexcept -> same_as<typename M::reference_type>;
  #endif
  #if LINALG_USE_PAREN_OPERATOR
  { m(index1,index2) }       noexcept -> same_as<typename M::reference_type>;
  #endif
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

#else

//- Tests for aliases

// Test if T has alias const_row_type
template < class T, class = void > struct has_const_row_type : public false_type { };
template < class T > struct has_const_row_type< T, std::enable_if_t< std::is_same_v< typename T::const_row_type, typename T::const_row_type > > > : public true_type { };
template < class T > inline constexpr bool has_const_row_type_v = has_const_row_type<T>::value;

// Test if T has alias const_column_type
template < class T, class = void > struct has_const_column_type : public false_type { };
template < class T > struct has_const_column_type< T, std::enable_if_t< std::is_same_v< typename T::const_column_type, typename T::const_column_type > > > : public true_type { };
template < class T > inline constexpr bool has_const_column_type_v = has_const_column_type<T>::value;

// Test if T has alias const_submatrix_type
template < class T, class = void > struct has_const_submatrix_type : public false_type { };
template < class T > struct has_const_submatrix_type< T, std::enable_if_t< std::is_same_v< typename T::const_submatrix_type, typename T::const_submatrix_type > > > : public true_type { };
template < class T > inline constexpr bool has_const_submatrix_type_v = has_const_submatrix_type<T>::value;

// Test if T has alias row_type
template < class T, class = void > struct has_row_type : public false_type { };
template < class T > struct has_row_type< T, std::enable_if_t< std::is_same_v< typename T::row_type, typename T::row_type > > > : public true_type { };
template < class T > inline constexpr bool has_row_type_v = has_row_type<T>::value;

// Test if T has alias column_type
template < class T, class = void > struct has_column_type : public false_type { };
template < class T > struct has_column_type< T, std::enable_if_t< std::is_same_v< typename T::column_type, typename T::column_type > > > : public true_type { };
template < class T > inline constexpr bool has_column_type_v = has_column_type<T>::value;

// Test if T has alias submatrix_type
template < class T, class = void > struct has_submatrix_type : public false_type { };
template < class T > struct has_submatrix_type< T, std::enable_if_t< std::is_same_v< typename T::submatrix_type, typename T::submatrix_type > > > : public true_type { };
template < class T > inline constexpr bool has_submatrix_type_v = has_submatrix_type<T>::value;

//- Test for functions

// Test for rows function
template < class T, class = void > struct has_rows_func : public false_type { };
template < class T > struct has_rows_func< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().rows() ), typename T::size_type > > > : public true_type { };
template < class T > inline constexpr bool has_rows_func_v = has_rows_func<T>::value;

// Test for columns function
template < class T, class = void > struct has_columns_func : public false_type { };
template < class T > struct has_columns_func< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().columns() ), typename T::size_type > > > : public true_type { };
template < class T > inline constexpr bool has_columns_func_v = has_columns_func<T>::value;

// Test for row_capacity function
template < class T, class = void > struct has_row_capacity_func : public false_type { };
template < class T > struct has_row_capacity_func< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().row_capacity() ), typename T::size_type > > > : public true_type { };
template < class T > inline constexpr bool has_row_capacity_func_v = has_row_capacity_func<T>::value;

// Test for column_capacity function
template < class T, class = void > struct has_column_capacity_func : public false_type { };
template < class T > struct has_column_capacity_func< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().column_capacity() ), typename T::size_type > > > : public true_type { };
template < class T > inline constexpr bool has_column_capacity_func_v = has_column_capacity_func<T>::value;

// Test for index paren operator
template < class T, class = void > struct has_const_index_paren_oper_two : public false_type { };
template < class T > struct has_const_index_paren_oper_two< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().operator()( declval<typename T::index_type>(), declval<typename T::index_type>() ) ), typename T::value_type > > > : public true_type { };
template < class T > inline constexpr bool has_const_index_paren_oper_two_v = has_const_index_paren_oper_two<T>::value;

// Test for index bracket operator
template < class T, class = void > struct has_const_index_bracket_oper_two : public false_type { };
template < class T > struct has_const_index_bracket_oper_two< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().operator[]( declval<typename T::index_type>(), declval<typename T::index_type>() ) ), typename T::value_type > > > : public true_type { };
template < class T > inline constexpr bool has_const_index_bracket_oper_two_v = has_const_index_bracket_oper_two<T>::value;

// Test for index paren operator
template < class T, class = void > struct has_index_paren_oper_two : public false_type { };
template < class T > struct has_index_paren_oper_two< T, std::enable_if_t< std::is_same_v< decltype( declval<T>().operator()( declval<typename T::index_type>(), declval<typename T::index_type>() ) ), typename T::reference_type > > > : public true_type { };
template < class T > inline constexpr bool has_index_paren_oper_two_v = has_index_paren_oper_two<T>::value;

// Test for index bracket operator
template < class T, class = void > struct has_index_bracket_oper_two : public false_type { };
template < class T > struct has_index_bracket_oper_two< T, std::enable_if_t< std::is_same_v< decltype( declval<T>().operator[]( declval<typename T::index_type>(), declval<typename T::index_type>() ) ), typename T::reference_type > > > : public true_type { };
template < class T > inline constexpr bool has_index_bracket_oper_two_v = has_index_bracket_oper_two<T>::value;

// Test for const row function
template < class T, class = void > struct has_const_row_func : public false_type { };
template < class T > struct has_const_row_func< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().row( declval<typename T::index_type>() ) ), typename T::const_row_type > > > : public true_type { };
template < class T > inline constexpr bool has_const_row_func_v = has_const_row_func<T>::value;

// Test for row function
template < class T, class = void > struct has_row_func : public false_type { };
template < class T > struct has_row_func< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().row( declval<typename T::index_type>() ) ), typename T::row_type > > > : public true_type { };
template < class T > inline constexpr bool has_row_func_v = has_row_func<T>::value;

// Test for const column function
template < class T, class = void > struct has_const_column_func : public false_type { };
template < class T > struct has_const_column_func< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().column( declval<typename T::index_type>() ) ), typename T::const_column_type > > > : public true_type { };
template < class T > inline constexpr bool has_const_column_func_v = has_const_column_func<T>::value;

// Test for column function
template < class T, class = void > struct has_column_func : public false_type { };
template < class T > struct has_column_func< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().column( declval<typename T::index_type>() ) ), typename T::column_type > > > : public true_type { };
template < class T > inline constexpr bool has_column_func_v = has_column_func<T>::value;

// Test for const submatrix function
template < class T, class = void > struct has_const_submatrix_func : public false_type { };
template < class T > struct has_const_submatrix_func< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().submatrix( declval<typename T::tuple_type>(), declval<typename T::tuple_type>() ) ), typename T::const_submatrix_type > > > : public true_type { };
template < class T > inline constexpr bool has_const_submatrix_func_v = has_const_submatrix_func<T>::value;

// Test for submatrix function
template < class T, class = void > struct has_submatrix_func : public false_type { };
template < class T > struct has_submatrix_func< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().submatrix( declval<typename T::tuple_type>(), declval<typename T::tuple_type>() ) ), typename T::submatrix_type > > > : public true_type { };
template < class T > inline constexpr bool has_submatrix_func_v = has_submatrix_func<T>::value;

// Test for matrix multiplication
template < class T, class = void > struct has_matrix_multi_func : public false_type { };
template < class T > struct has_matrix_multi_func< T, std::enable_if_t< std::is_same_v< decltype( declval<T>() * trans( declval<T>() ) ), decltype( declval<T>() * trans( declval<T>() ) ) > > > : public true_type { };
template < class T > inline constexpr bool has_matrix_multi_func_v = has_matrix_multi_func<T>::value;

//- Test for matrices

// Matrix data
template < class M > struct matrix_data : public conditional_t< 
  tensor_data_v<M> &&
  //( M::extents_type::rank() == 2 ) &&
  has_rows_func_v<M> &&
  has_columns_func_v<M> &&
  has_row_capacity_func_v<M> &&
  has_column_capacity_func_v<M>, true_type, false_type > { };
template < class M > inline constexpr bool matrix_data_v = matrix_data<M>::value;

// Matrix
template < class M > struct matrix : public conditional_t< 
  matrix_data_v<M> &&
  tensor_v<M> &&
  has_trans_func_v<M> &&
  has_conj_func_v<M> &&
  has_matrix_multi_func_v<M>, true_type, false_type > { };
template < class M > inline constexpr bool matrix_v = matrix<M>::value;

// Readable matrix data
template < class M > struct readable_matrix_data : public conditional_t< 
  matrix_data_v<M> &&
  ( !LINALG_USE_BRACKET_OPERATOR || has_const_index_bracket_oper_two_v<M> ) &&
  ( !LINALG_USE_PAREN_OPERATOR || has_const_index_paren_oper_two_v<M> ) &&
  has_const_row_type_v<M> &&
  has_const_row_func_v<M> &&
  //readable_vector_data_v<typename M::const_row_vector_type> &&
  //readable_vector_data_v<typename M::const_column_vector_type> &&
  has_const_column_type_v<M> &&
  has_const_column_func_v<M> &&
  has_const_submatrix_type_v<M> &&
  has_const_submatrix_func_v<M>, true_type, false_type > { };
template < class M > inline constexpr bool readable_matrix_data_v = readable_matrix_data<M>::value;

// Readable matrix
template < class M > struct readable_matrix : public conditional_t< 
  readable_matrix_data_v<M> &&
  matrix_v<M>, true_type, false_type > { };
template < class M > inline constexpr bool readable_matrix_v = readable_matrix<M>::value;

// Writable matrix data
template < class M > struct writable_matrix_data : public conditional_t< 
  matrix_data_v<M> &&
  ( !LINALG_USE_BRACKET_OPERATOR || has_index_bracket_oper_two_v<M> ) &&
  ( !LINALG_USE_PAREN_OPERATOR || has_index_paren_oper_two_v<M> ) &&
  has_row_type_v<M> &&
  has_row_func_v<M> &&
  //writable_vector_data_v<typename M::row_vector_type> &&
  //writable_vector_data_v<typename M::column_vector_type> &&
  has_column_type_v<M> &&
  has_column_func_v<M> &&
  has_submatrix_type_v<M> &&
  has_submatrix_func_v<M>, true_type, false_type > { };
template < class M > inline constexpr bool writable_matrix_data_v = writable_matrix_data<M>::value;

// Writable matrix
template < class M > struct writable_matrix : public conditional_t< 
  writable_matrix_data_v<M> &&
  matrix_v<M>, true_type, false_type > { };
template < class M > inline constexpr bool writable_matrix_v = writable_matrix<M>::value;

// Dynamic matrix data
template < class M > struct dynamic_matrix_data : public conditional_t< 
  matrix_data_v<M> &&
  dynamic_tensor_data_v<M>, true_type, false_type > { };
template < class M > inline constexpr bool dynamic_matrix_data_v = dynamic_matrix_data<M>::value;

// Dynamic matrix
template < class M > struct dynamic_matrix : public conditional_t< 
  dynamic_matrix_data_v<M> &&
  matrix_v<M>, true_type, false_type > { };
template < class M > inline constexpr bool dynamic_matrix_v = dynamic_matrix<M>::value;

// Fixed size matrix data
template < class M > struct fixed_size_matrix_data : public conditional_t< 
  matrix_data_v<M> &&
  fixed_size_tensor_data_v<M> &&
  detail::is_constexpr( []{ decltype( declval<M>().rows() )            nodiscard_warning = M().rows(); } ) &&
  detail::is_constexpr( []{ decltype( declval<M>().columns() )         nodiscard_warning = M().columns(); } ) &&
  detail::is_constexpr( []{ decltype( declval<M>().row_capacity() )    nodiscard_warning = M().row_capacity(); } ) &&
  detail::is_constexpr( []{ decltype( declval<M>().column_capacity() ) nodiscard_warning = M().column_capacity(); } ) &&
  ( M().rows()    == M().row_capacity() ) &&
  ( M().columns() == M().column_capacity() ), true_type, false_type > { };
template < class M > inline constexpr bool fixed_size_matrix_data_v = fixed_size_matrix_data<M>::value;

// Fixed size matrix
template < class M > struct fixed_size_matrix : public conditional_t< 
  fixed_size_matrix_data_v<M> &&
  matrix_v<M>, true_type, false_type > { };
template < class M > inline constexpr bool fixed_size_matrix_v = fixed_size_matrix<M>::value;

// Matrix may be convertible
// Enforces both types are matrix which have convertible elements
// and both rows / columns must be the same size or dynamic
template < class From, class To >
inline constexpr bool matrix_may_be_convertible_v =
  matrix_v<From> && matrix_v<To> &&
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
inline constexpr bool view_may_be_convertible_to_matrix_v =
  detail::is_mdspan_v<From> && matrix_v<To> &&
  ( From::rank() == 2 ) &&
  matrix_v<To> &&
  ( is_convertible_v<typename From::value_type,typename To::value_type> ) &&
  ( ( From::extents_type::static_extent(0) == To::extents_type::static_extent(0) ) ||
    ( From::extents_type::static_extent(0) == std::experimental::dynamic_extent ) ||
    ( To::extents_type::static_extent(0) == std::experimental::dynamic_extent ) ) &&
  ( ( From::extents_type::static_extent(1) == To::extents_type::static_extent(1) ) ||
    ( From::extents_type::static_extent(1) == std::experimental::dynamic_extent ) ||
    ( To::extents_type::static_extent(1) == std::experimental::dynamic_extent ) );

#endif

}       //- concepts namespace
}       //- math namespace
}       //- std namespace

#endif  //- LINEAR_ALGEBRA_MATRIX_CONCEPTS_HPP_DEFINED
