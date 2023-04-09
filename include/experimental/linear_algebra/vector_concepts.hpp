#ifndef LINEAR_ALGEBRA_VECTOR_CONCEPTS_HPP
#define LINEAR_ALGEBRA_VECTOR_CONCEPTS_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{
namespace concepts
{

#ifdef LINALG_ENABLE_CONCEPTS

//=================================================================================================
//  Vector Engine Concepts
//=================================================================================================

// All vector data types must satisfy tensor data requirements
// All vector data types must be a rank 1 tensor
template < class V >
concept vector_data = tensor_data<V> &&
( V::extents_type::rank() == 1 );

// Vector must be a vector data
// Vector must be a tensor
// All vector types must support unary functions: trans and conj
// All vector types must support binary functions: inner_prod and outer_prod
template < class V >
concept vector = vector_data<V> &&
tensor<V> &&
requires ( V v ) // Unary operations
{
  trans( v );
  conj( v );
} &&
requires ( V v ) // Binary operations
{
  inner_prod( v, v );
  outer_prod( v, v );
};

// Readable vector data must A) be a vector data
// Readable vector data must B) be a readable tensor data
// Readable vector data must C) provide index access
// Readable vector data must D) provide a one dimension const span representation
// Readable vector data must E) provide a const subvector view
template < class V >
concept readable_vector_data = vector_data<V> &&     // A)
readable_tensor_data<V> &&                           // B)
requires( const V& v, typename V::index_type index ) // C)
{
  #if LINALG_USE_BRACKET_OPERATOR
  { v[index] }       noexcept -> same_as<typename V::value_type>;
  #endif
  #if LINALG_USE_PAREN_OPERATOR
  { v(index) }       noexcept -> same_as<typename V::value_type>;
  #endif
  { v.at(index) }             -> same_as<typename V::value_type>;
} &&
( V::span_type::extents_type::rank() == 1 ) &&       // D)
requires( const V&               v,
          typename V::index_type start,
          typename V::index_type end )               // E)
{
  typename V::const_subvector_type;
  { v.subvector(start,end) } -> same_as<typename V::const_subvector_type>;
  { v.subvector(start,end) } -> vector_data;
};

// Readable vector must be a readable vector data
// Readable vector must be a vector
template < class V >
concept readable_vector = readable_vector_data<V> && vector<V>;

// Writable vector data must A) be a readable vector data
// Writable vector data must B) be a writable tensor data
// Writable vector data must C) provide a mutable scalar view
// Writable vector data must D) provide a mutable subvector view
template < class V >
concept writable_vector_data = readable_vector_data<V> && // A)
writable_tensor_data<V> &&                                // B)
requires( V& v, typename V::index_type index )            // C)
{
  typename V::reference_type;
  #if LINALG_USE_BRACKET_OPERATOR
  { v[index] }       noexcept -> same_as<typename V::reference_type>;
  #endif
  #if LINALG_USE_PAREN_OPERATOR
  { v(index) }       noexcept -> same_as<typename V::reference_type>;
  #endif
  { v.at(index) }             -> same_as<typename V::reference_type>;
} &&
requires( V&                     v,
          typename V::index_type start,
          typename V::index_type end )                    // D)
{
  typename V::subvector_type;
  { v.subvector(start,end) } -> same_as<typename V::subvector_type>;
  { v.subvector(start,end) } -> vector_data;
};

// Writable vector must be a writable vector data
// Writable vector must be a readable vector
template < class V >
concept writable_vector = writable_vector_data<V> && readable_vector<V>;

// Dynamic vector data must be a vector data
// Dynamic vector data must be a dynamic tensor data
template < class V >
concept dynamic_vector_data = vector_data<V> &&
dynamic_tensor_data<V>;

// Dynamic vector must be a dynamic vector data
// Dynamic vector must be a vector
template < class V >
concept dynamic_vector = dynamic_vector_data<V> &&
vector<V>;

// Fixed size vector data must be a vector data
// Fixed size vector data must be a fixed size tensor data
template < class V >
concept fixed_size_vector_data = vector_data<V> &&
fixed_size_tensor_data<V>;

// Fixed size vector must be a fixed size vector data
// Fixed size vector must be a vector
template < class V >
concept fixed_size_vector = fixed_size_vector_data<V> &&
vector<V>;

#else

//- Tests for aliases

// Test if T has alias const_subvector_type
template < class T, class = void > struct has_const_subvector_type : public false_type { };
template < class T > struct has_const_subvector_type< T, std::enable_if_t< std::is_same_v< typename T::const_subvector_type, typename T::const_subvector_type > > > : public true_type { };
template < class T > inline constexpr bool has_const_subvector_type_v = typename has_const_subvector_type<T>::value;

// Test if T has alias subvector_type
template < class T, class = void > struct has_subvector_type : public false_type { };
template < class T > struct has_subvector_type< T, std::enable_if_t< std::is_same_v< typename T::subvector_type, typename T::subvector_type > > > : public true_type { };
template < class T > inline constexpr bool has_subvector_type_v = typename has_subvector_type<T>::value;

//- Test for functions

// Test for inner product
template < class T, class = void > struct has_inner_prod_func : public false_type { };
template < class T > struct has_inner_prod_func< T, std::enable_if_t< std::is_same_v< decltype( inner_prod( declval<T>(), declval<T>() ) ), decltype( inner_prod( declval<T>(), declval<T>() ) ) > > > : public true_type { };
template < class T > inline constexpr bool has_inner_prod_func_v = typename has_inner_prod_func<T>::value;

// Test for outer product
template < class T, class = void > struct has_outer_prod_func : public false_type { };
template < class T > struct has_outer_prod_func< T, std::enable_if_t< std::is_same_v< decltype( outer_prod( declval<T>(), declval<T>() ) ), decltype( outer_prod( declval<T>(), declval<T>() ) ) > > > : public true_type { };
template < class T > inline constexpr bool has_outer_prod_func_v = typename has_outer_prod_func<T>::value;

// Test for index paren operator
template < class T, class = void > struct has_const_index_paren_oper_one : public false_type { };
template < class T > struct has_const_index_paren_oper_one< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().operator()( declval<typename T::index_type>() ) ), typename T::value_type > > > : public true_type { };
template < class T > inline constexpr bool has_const_index_paren_oper_one_v = typename has_const_index_paren_oper_one<T>::value;

// Test for index bracket operator
template < class T, class = void > struct has_const_index_bracket_oper_one : public false_type { };
template < class T > struct has_const_index_bracket_oper_one< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().operator[]( declval<typename T::index_type>() ) ), typename T::value_type > > > : public true_type { };
template < class T > inline constexpr bool has_const_index_bracket_oper_one_v = typename has_const_index_bracket_oper_one<T>::value;

// Test for index paren operator
template < class T, class = void > struct has_index_paren_oper_one : public false_type { };
template < class T > struct has_index_paren_oper_one< T, std::enable_if_t< std::is_same_v< decltype( declval<T>().operator()( declval<typename T::index_type>() ) ), typename T::reference_type > > > : public true_type { };
template < class T > inline constexpr bool has_index_paren_oper_one_v = typename has_index_paren_oper_one<T>::value;

// Test for index bracket operator
template < class T, class = void > struct has_index_bracket_oper_one : public false_type { };
template < class T > struct has_index_bracket_oper_one< T, std::enable_if_t< std::is_same_v< decltype( declval<T>().operator[]( declval<typename T::index_type>() ) ), typename T::reference_type > > > : public true_type { };
template < class T > inline constexpr bool has_index_bracket_oper_one_v = typename has_index_bracket_oper_one<T>::value;

// Test for const subvector function
template < class T, class = void > struct has_const_subvector_func : public false_type { };
template < class T > struct has_const_subvector_func< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().subvector( declval<typename T::index_type>(), declval<typename T::index_type>() ) ), decltype( declval<const T>().subvector( declval<typename T::index_type>(), declval<typename T::index_type>() ) ) > > > : public true_type { };
template < class T > inline constexpr bool has_const_subvector_func_v = has_const_subvector_func<T>::value;

// Test for subvector function
template < class T, class = void > struct has_subvector_func : public false_type { };
template < class T > struct has_subvector_func< T, std::enable_if_t< std::is_same_v< decltype( declval<const T>().subvector( declval<typename T::index_type>(), declval<typename T::index_type>() ) ), decltype( declval<const T>().subvector( declval<typename T::index_type>(), declval<typename T::index_type>() ) ) > > > : public true_type { };
template < class T > inline constexpr bool has_subvector_func_v = has_subvector_func<T>::value;

//- Test for vectors

// Vector data
template < class V > struct vector_data : public conditional_t< 
  tensor_data_v<V> &&
( V::extents_type::rank() == 1 ), true_type, false_type > { };
template < class V > inline constexpr bool vector_data_v = vector_data<V>::value;

// Vector
template < class V > struct vector : public conditional_t< 
  vector_data_v<V> &&
  tensor_v<V> &&
  has_trans_func_v<V> &&
  has_conj_func_v<V> &&
  has_inner_prod_func_v<V> &&
  has_outer_prod_func_v<V>, true_type, false_type > { };
template < class V > inline constexpr bool vector_v = vector<V>::value;

// Readable vector data
template < class V > struct readable_vector_data : public conditional_t< 
  vector_data_v<V> &&
  ( !LINALG_USE_BRACKET_OPERATOR || has_const_index_bracket_oper_one_v<V> ) &&
  ( !LINALG_USE_PAREN_OPERATOR || has_const_index_paren_oper_one_v<V> ) &&
  has_const_subvector_type_v<V> &&
  has_const_subtensor_func_v<V>, true_type, false_type > { };
template < class V > inline constexpr bool readable_vector_data_v = readable_vector_data<V>::value;

// Readable vector
template < class V > struct readable_vector : public conditional_t< 
  readable_vector_data_v<V> &&
  vector_v<V>, true_type, false_type > { };
template < class V > inline constexpr bool readable_vector_v = readable_vector<V>::value;

// Writable vector data
template < class V > struct writable_vector_data : public conditional_t< 
  readable_vector_data_v<V> &&
  ( !LINALG_USE_BRACKET_OPERATOR || has_index_bracket_oper_one_v<V> ) &&
  ( !LINALG_USE_PAREN_OPERATOR || has_index_paren_oper_one_v<V> ) &&
  has_subvector_type_v<V> &&
  has_subtensor_func_v<V>, true_type, false_type > { };
template < class V > inline constexpr bool writable_vector_data_v = writable_vector_data<V>::value;

// Writable vector
template < class V > struct writable_vector : public conditional_t< 
  writable_vector_data_v<V> &&
  vector_v<V>, true_type, false_type > { };
template < class V > inline constexpr bool writable_vector_v = writable_vector<V>::value;

// Dynamic vector data
template < class V > struct dynamic_vector_data : public conditional_t< 
  vector_data_v<V> &&
  dynamic_tensor_data_v<V>, true_type, false_type > { };
template < class V > inline constexpr bool dynamic_vector_data_v = dynamic_vector_data<V>::value;

// Dynamic vector
template < class V > struct dynamic_vector : public conditional_t< 
  dynamic_vector_data_v<V> &&
  vector_v<V>, true_type, false_type > { };
template < class V > inline constexpr bool dynamic_vector_v = dynamic_vector<V>::value;

// Fixed size vector data
template < class V > struct fixed_size_vector_data : public conditional_t< 
  vector_data_v<V> &&
  fixed_size_tensor_data_v<V>, true_type, false_type > { };
template < class V > inline constexpr bool fixed_size_vector_data_v = fixed_size_vector_data<V>::value;

// Fixed size vector
template < class V > struct fixed_size_vector : public conditional_t< 
  fixed_size_vector_data_v<V> &&
  vector_v<V>, true_type, false_type > { };
template < class V > inline constexpr bool fixed_size_vector_v = fixed_size_vector<V>::value;

#endif

}       //- concepts namespace
}       //- math namespace
}       //- std namespace

#endif  //- LINEAR_ALGEBRA_VECTOR_CONCEPTS_HPP
