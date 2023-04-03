#ifndef LINEAR_ALGEBRA_VECTOR_CONCEPTS_HPP
#define LINEAR_ALGEBRA_VECTOR_CONCEPTS_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{
namespace concepts
{

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
  { v[index] }       noexcept -> same_as<typename V::value_type>;
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
  { v[index] }       noexcept -> same_as<typename V::reference_type>;
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

}       //- concepts namespace
}       //- math namespace
}       //- std namespace

#endif  //- LINEAR_ALGEBRA_VECTOR_CONCEPTS_HPP
