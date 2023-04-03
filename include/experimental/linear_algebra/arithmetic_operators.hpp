//==================================================================================================
//  File:       arithmetic_operators.hpp
//
//  Summary:    This header defines the overloaded operators that implement basic arithmetic
//              operations on vectors, matrices, and tensors.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_ARITHMETIC_OPERATORS_HPP
#define LINEAR_ALGEBRA_ARITHMETIC_OPERATORS_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{

//=================================================================================================
//  Unary negation operators
//=================================================================================================
template < concepts::tensor_data T >
[[nodiscard]] inline constexpr decltype(auto)
operator - ( const T& t ) noexcept( noexcept( operations::template negation<T>::negate( t ) ) )
{
  return operations::template negation<T>::negate( t );
}

//=================================================================================================
//  Unary transpose operators
//=================================================================================================
template < concepts::matrix_data M >
[[nodiscard]] inline constexpr decltype(auto)
trans( const M& m ) noexcept( noexcept( operations::template transpose_matrix<M>::trans( m ) ) )
{
  return operations::template transpose_matrix<M>::trans( m );
}

template < concepts::vector_data V >
[[nodiscard]] inline constexpr decltype(auto)
trans( const V& v ) noexcept( noexcept( operations::template transpose_vector<V>::trans( v ) ) )
{
  return operations::template transpose_vector<V>::trans( v );
}

//=================================================================================================
//  Unary conjugate transpose operators
//=================================================================================================
template < concepts::matrix_data M >
[[nodiscard]] inline constexpr decltype(auto)
conj( const M& m ) noexcept( noexcept( operations::template conjugate_matrix<M>::conjugate( m ) ) )
{
  return operations::template conjugate_matrix<M>::conjugate( m );
}

template < concepts::vector_data V >
[[nodiscard]] inline constexpr decltype(auto)
conj( const V& v ) noexcept( noexcept( operations::template conjugate_vector<V>::conjugate( v ) ) )
{
  return operations::template conjugate_vector<V>::conjugate( v );
}

//=================================================================================================
//  Binary addition operators
//=================================================================================================
template < concepts::tensor_data T1, concepts::tensor_data T2 >
[[nodiscard]] inline constexpr decltype(auto)
operator + ( const T1& t1, const T2& t2 ) noexcept( noexcept( operations::template addition<T1,T2>::add( t1, t2 ) ) )
{
  return operations::template addition<T1,T2>::add( t1, t2 );
}

//=================================================================================================
//  Binary addition assignment operators
//=================================================================================================
template < concepts::tensor_data T1, concepts::tensor_data T2 >
[[nodiscard]] inline constexpr T1&
operator += ( T1& t1, const T2& t2 ) noexcept( noexcept( operations::template addition<T1,T2>::add( t1, t2 ) ) )
  requires is_convertible_v< decltype( declval<typename T1::value_type>() + declval<typename T2::value_type>() ), typename T1::element_type >
{
  return operations::template addition<T1,T2>::add( t1, t2 );
}

//=================================================================================================
//  Binary subtraction operators
//=================================================================================================
template < concepts::tensor_data T1, concepts::tensor_data T2 >
[[nodiscard]] inline constexpr decltype(auto)
operator - ( const T1& t1, const T2& t2 ) noexcept( noexcept( operations::template subtraction<T1,T2>::subtract( t1, t2 ) ) )
{
  return operations::template subtraction<T1,T2>::subtract( t1, t2 );
}

//=================================================================================================
//  Binary subtraction assignment operators
//=================================================================================================
template < concepts::tensor_data T1, concepts::tensor_data T2 >
[[nodiscard]] inline constexpr T1&
operator -= ( T1& t1, const T2& t2 ) noexcept( noexcept( operations::template subtraction<T1,T2>::subtract( t1, t2 ) ) )
  requires is_convertible_v< decltype( declval<typename T1::value_type>() - declval<typename T2::value_type>() ), typename T1::element_type >
{
  return operations::template subtraction<T1,T2>::subtract( t1, t2 );
}

//=================================================================================================
//  Scalar pre-multiplication operators
//=================================================================================================
template < class S, concepts::tensor_data T >
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const S& s, const T& t )
  noexcept( noexcept( operations::template scalar_product<S,T>::prod(s,t) ) )
  requires requires { s * declval< typename T::value_type >(); } &&
           ( !concepts::tensor_data<S> )
{
  return operations::template scalar_product<S,T>::prod(s,t);
}

//=================================================================================================
//  Scalar post-multiplication operators
//=================================================================================================
template < concepts::tensor_data T, class S >
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const T& t, const S& s )
  noexcept( noexcept( operations::template scalar_product<S,T>::prod(t,s) ) )
  requires requires { declval< typename T::value_type >() * s; } &&
           ( !concepts::tensor_data<S> )
{
  return operations::template scalar_product<S,T>::prod(t,s);
}

//=================================================================================================
//  Scalar post-multiplication assignment operators
//=================================================================================================
template < concepts::tensor_data T, class S >
[[nodiscard]] inline constexpr T&
operator *= ( T& t, const S& s )
  noexcept( noexcept( operations::template scalar_product<S,T>::prod(t,s) ) )
  requires requires { declval< typename T::value_type >() * s; } &&
           ( !concepts::tensor_data<S> ) &&
  is_convertible_v< decltype( declval<typename T::value_type>() * declval<S>() ), typename T::element_type >
{
  return operations::template scalar_product<S,T>::prod(t,s);
}

//=================================================================================================
//  Scalar divison operators
//=================================================================================================
template < concepts::tensor_data T, class S >
[[nodiscard]] inline constexpr decltype(auto)
operator / ( const T& t, const S& s )
  noexcept( noexcept( operations::template scalar_division<T,S>::divide(t,s) ) )
  requires requires { declval< typename T::value_type >() / s; }
{
  return operations::template scalar_division<T,S>::divide(t,s);
}

//=================================================================================================
//  Scalar divison assignment operators
//=================================================================================================
template < concepts::tensor_data T, class S >
[[nodiscard]] inline constexpr T&
operator /= ( T& t, const S& s )
  noexcept( noexcept( operations::template scalar_division<T,S>::divide(t,s) ) )
  requires requires { declval< typename T::value_type >() / s; } &&
           ( !concepts::tensor_data<S> ) &&
  is_convertible_v< decltype( declval<typename T::value_type>() / declval<S>() ), typename T::element_type >
{
  return operations::template scalar_division<T,S>::divide(t,s);
}

//=================================================================================================
//  Inner product
//=================================================================================================
template < concepts::vector_data V1, concepts::vector_data V2 >
[[nodiscard]] inline constexpr decltype(auto)
inner_prod( const V1& v1, const V2& v2 ) noexcept( noexcept( operations::template inner_product<V1,V2>::prod( v1, v2 ) ) )
{
  return operations::template inner_product<V1,V2>::prod( v1, v2 );
}

//=================================================================================================
//  Outer product
//=================================================================================================
template < concepts::vector_data V1, concepts::vector_data V2 >
[[nodiscard]] inline constexpr decltype(auto)
outer_prod( const V1& v1, const V2& v2 ) noexcept( noexcept( operations::template outer_product<V1,V2>::prod( v1, v2 ) ) )
{
  return operations::template outer_product<V1,V2>::prod( v1, v2 );
}

//=================================================================================================
//  Vector Matrix product
//=================================================================================================
template < concepts::vector_data V, concepts::matrix_data M >
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const V& v, const M& m )
  noexcept( noexcept( operations::template vector_matrix_product<V,M>::prod(v,m) ) )
{
  return operations::template vector_matrix_product<V,M>::prod(v,m);
}

//=================================================================================================
//  Vector Matrix multiplication assignment
//=================================================================================================
template < concepts::vector_data V, concepts::matrix_data M >
[[nodiscard]] inline constexpr V&
operator *= ( V& v, const M& m )
  noexcept( noexcept( operations::template vector_matrix_product<V,M>::prod(v,m) ) )
  requires detail::extents_may_be_equal_v< typename V::extents_type, typename decltype( const_cast<const V&>(v) * m )::extents_type >
{
  return operations::template vector_matrix_product<V,M>::prod(v,m);
}

//=================================================================================================
//  Matrix Vector product
//=================================================================================================
template < concepts::matrix_data M, concepts::vector_data V >
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const M& m, const V& v )
  noexcept( noexcept( operations::template vector_matrix_product<V,M>::prod(m,v) ) )
{
  return operations::template vector_matrix_product<V,M>::prod(m,v);
}

//=================================================================================================
//  Matrix Matrix product
//=================================================================================================
template < concepts::matrix_data M1, concepts::matrix_data M2 >
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const M1& m1, const M2& m2 )
  noexcept( noexcept( operations::template matrix_matrix_product<M1,M2>::prod(m1,m2) ) )
{
  return operations::template matrix_matrix_product<M1,M2>::prod(m1,m2);
}

template < concepts::matrix_data M1, concepts::matrix_data M2 >
[[nodiscard]] inline constexpr M1&
operator *= ( M1& m1, const M2& m2 )
  noexcept( noexcept( operations::template matrix_matrix_product<M1,M2>::prod(m1,m2) ) )
  requires detail::extents_may_be_equal_v< typename M1::extents_type, typename decltype( const_cast<const M1&>(m1) * m2 )::extents_type >
{
  return operations::template matrix_matrix_product<M1,M2>::prod(m1,m2);
}

}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_ARITHMETIC_OPERATORS_HPP
