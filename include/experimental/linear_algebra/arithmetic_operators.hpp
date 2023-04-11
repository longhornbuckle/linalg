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
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_data T >
#else
template < class T, typename = enable_if_t< concepts::tensor_data_v<T> > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator - ( const T& t ) noexcept( noexcept( operations::template negation<T>::negate( t ) ) )
{
  return operations::template negation<T>::negate( t );
}

//=================================================================================================
//  Unary transpose operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::matrix_data M >
#else
template < class M, typename = enable_if_t< concepts::matrix_data_v<M> > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
trans( const M& m ) noexcept( noexcept( operations::template transpose_matrix<M>::trans( m ) ) )
{
  return operations::template transpose_matrix<M>::trans( m );
}

#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::vector_data V >
#else
template < class V, typename = enable_if_t< concepts::vector_data_v<V> >, typename = enable_if_t<true> >
#endif
[[nodiscard]] inline constexpr decltype(auto)
trans( const V& v ) noexcept( noexcept( operations::template transpose_vector<V>::trans( v ) ) )
{
  return operations::template transpose_vector<V>::trans( v );
}

//=================================================================================================
//  Unary conjugate transpose operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::matrix_data M >
#else
template < class M, typename = enable_if_t< concepts::matrix_data_v<M> > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
conj( const M& m ) noexcept( noexcept( operations::template conjugate_matrix<M>::conjugate( m ) ) )
{
  return operations::template conjugate_matrix<M>::conjugate( m );
}

#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::vector_data V >
#else
template < class V, typename = enable_if_t< concepts::vector_data_v<V> >, typename = enable_if_t<true> >
#endif
[[nodiscard]] inline constexpr decltype(auto)
conj( const V& v ) noexcept( noexcept( operations::template conjugate_vector<V>::conjugate( v ) ) )
{
  return operations::template conjugate_vector<V>::conjugate( v );
}

//=================================================================================================
//  Binary addition operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_data T1, concepts::tensor_data T2 >
#else
template < class T1, class T2, typename = enable_if_t< concepts::tensor_data_v<T1> && concepts::tensor_data_v<T2> > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator + ( const T1& t1, const T2& t2 ) noexcept( noexcept( operations::template addition<T1,T2>::add( t1, t2 ) ) )
{
  return operations::template addition<T1,T2>::add( t1, t2 );
}

//=================================================================================================
//  Binary addition assignment operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_data T1, concepts::tensor_data T2 >
#else
template < class T1, class T2,
           typename = enable_if_t< concepts::tensor_data_v<T1> &&
                                   concepts::tensor_data_v<T2> &&
                                   is_convertible_v< decltype( declval<typename T1::value_type>() + declval<typename T2::value_type>() ), typename T1::element_type > > >
#endif
[[nodiscard]] inline constexpr T1&
operator += ( T1& t1, const T2& t2 ) noexcept( noexcept( operations::template addition<T1,T2>::add( t1, t2 ) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires is_convertible_v< decltype( declval<typename T1::value_type>() + declval<typename T2::value_type>() ), typename T1::element_type >
#endif
{
  return operations::template addition<T1,T2>::add( t1, t2 );
}

//=================================================================================================
//  Binary subtraction operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_data T1, concepts::tensor_data T2 >
#else
template < class T1, class T2, typename = enable_if_t< concepts::tensor_data_v<T1> && concepts::tensor_data_v<T2> > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator - ( const T1& t1, const T2& t2 ) noexcept( noexcept( operations::template subtraction<T1,T2>::subtract( t1, t2 ) ) )
{
  return operations::template subtraction<T1,T2>::subtract( t1, t2 );
}

//=================================================================================================
//  Binary subtraction assignment operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_data T1, concepts::tensor_data T2 >
#else
template < class T1, class T2,
           typename = enable_if_t< concepts::tensor_data_v<T1> &&
                                   concepts::tensor_data_v<T2> &&
                                   is_convertible_v< decltype( declval<typename T1::value_type>() - declval<typename T2::value_type>() ), typename T1::element_type > > >
#endif
[[nodiscard]] inline constexpr T1&
operator -= ( T1& t1, const T2& t2 ) noexcept( noexcept( operations::template subtraction<T1,T2>::subtract( t1, t2 ) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires is_convertible_v< decltype( declval<typename T1::value_type>() - declval<typename T2::value_type>() ), typename T1::element_type >
#endif
{
  return operations::template subtraction<T1,T2>::subtract( t1, t2 );
}

//=================================================================================================
//  Scalar pre-multiplication operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < class S, concepts::tensor_data T >
#else
template < class S, class T,
           typename = enable_if_t< concepts::tensor_data_v<T> &&
                                   concepts::has_scalar_premultiply_func_v<T,S> > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const S& s, const T& t )
  noexcept( noexcept( operations::template scalar_product<S,T>::prod(s,t) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires { s * declval< typename T::value_type >(); } &&
           ( !concepts::tensor_data<S> )
#endif
{
  return operations::template scalar_product<S,T>::prod(s,t);
}

//=================================================================================================
//  Scalar post-multiplication operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_data T, class S >
#else
template < class T, class S,
           typename = enable_if_t< concepts::tensor_data_v<T> &&
                                   concepts::has_scalar_postmultiply_func_v<T,S> >,
           typename = enable_if_t<true> >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const T& t, const S& s )
  noexcept( noexcept( operations::template scalar_product<S,T>::prod(t,s) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires { declval< typename T::value_type >() * s; } &&
           ( !concepts::tensor_data<S> )
#endif
{
  return operations::template scalar_product<S,T>::prod(t,s);
}

//=================================================================================================
//  Scalar post-multiplication assignment operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_data T, class S >
#else
template < class T, class S,
           typename = enable_if_t< concepts::tensor_data_v<T> &&
                                   concepts::has_scalar_postmultiply_func_v<T,S> &&
                                   is_convertible_v< decltype( declval<typename T::value_type>() * declval<S>() ), typename T::element_type > > >
#endif
[[nodiscard]] inline constexpr T&
operator *= ( T& t, const S& s )
  noexcept( noexcept( operations::template scalar_product<S,T>::prod(t,s) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires { declval< typename T::value_type >() * s; } &&
           ( !concepts::tensor_data<S> ) &&
  is_convertible_v< decltype( declval<typename T::value_type>() * declval<S>() ), typename T::element_type >
#endif
{
  return operations::template scalar_product<S,T>::prod(t,s);
}

//=================================================================================================
//  Scalar divison operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_data T, class S >
#else
template < class T, class S,
           typename = enable_if_t< concepts::tensor_data_v<T> &&
                                   concepts::has_scalar_divide_func_v<T,S> > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator / ( const T& t, const S& s )
  noexcept( noexcept( operations::template scalar_division<T,S>::divide(t,s) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires { declval< typename T::value_type >() / s; }
#endif
{
  return operations::template scalar_division<T,S>::divide(t,s);
}

//=================================================================================================
//  Scalar divison assignment operators
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_data T, class S >
#else
template < class T, class S,
           typename = enable_if_t< concepts::tensor_data_v<T> &&
                                   concepts::has_scalar_divide_func_v<T,S> &&
                                   is_convertible_v< decltype( declval<typename T::value_type>() * declval<S>() ), typename T::element_type > > >
#endif
[[nodiscard]] inline constexpr T&
operator /= ( T& t, const S& s )
  noexcept( noexcept( operations::template scalar_division<T,S>::divide(t,s) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires { declval< typename T::value_type >() / s; } &&
           ( !concepts::tensor_data<S> ) &&
  is_convertible_v< decltype( declval<typename T::value_type>() / declval<S>() ), typename T::element_type >
#endif
{
  return operations::template scalar_division<T,S>::divide(t,s);
}

//=================================================================================================
//  Inner product
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::vector_data V1, concepts::vector_data V2 >
#else
template < class V1, class V2, typename = enable_if_t< concepts::vector_data_v<V1> && concepts::vector_data_v<V2> > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
inner_prod( const V1& v1, const V2& v2 ) noexcept( noexcept( operations::template inner_product<V1,V2>::prod( v1, v2 ) ) )
{
  return operations::template inner_product<V1,V2>::prod( v1, v2 );
}

//=================================================================================================
//  Outer product
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::vector_data V1, concepts::vector_data V2 >
#else
template < class V1, class V2, typename = enable_if_t< concepts::vector_data_v<V1> && concepts::vector_data_v<V2> > >
#endif
[[nodiscard]] inline constexpr decltype(auto)
outer_prod( const V1& v1, const V2& v2 ) noexcept( noexcept( operations::template outer_product<V1,V2>::prod( v1, v2 ) ) )
{
  return operations::template outer_product<V1,V2>::prod( v1, v2 );
}

//=================================================================================================
//  Vector Matrix product
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::vector_data V, concepts::matrix_data M >
#else
template < class V, class M,
           typename = enable_if_t< concepts::vector_data_v<V> &&
                                   concepts::matrix_data_v<M> >,
           typename = enable_if_t<true>,
           typename = enable_if_t<true> >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const V& v, const M& m )
  noexcept( noexcept( operations::template vector_matrix_product<V,M>::prod(v,m) ) )
{
  return operations::template vector_matrix_product<V,M>::prod(v,m);
}

//=================================================================================================
//  Vector Matrix multiplication assignment
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::vector_data V, concepts::matrix_data M >
#else
template < class V, class M,
           typename = enable_if_t< concepts::vector_data_v<V> &&
                                   concepts::matrix_data_v<M> &&
                                   detail::extents_may_be_equal_v< typename V::extents_type, typename decltype( declval<V>() * declval<M>() )::extents_type > >,
           typename = enable_if_t<true> >
#endif
[[nodiscard]] inline constexpr V&
operator *= ( V& v, const M& m )
  noexcept( noexcept( operations::template vector_matrix_product<V,M>::prod(v,m) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires detail::extents_may_be_equal_v< typename V::extents_type, typename decltype( const_cast<const V&>(v) * m )::extents_type >
#endif
{
  return operations::template vector_matrix_product<V,M>::prod(v,m);
}

//=================================================================================================
//  Matrix Vector product
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::matrix_data M, concepts::vector_data V >
#else
template < class M, class V,
           typename = enable_if_t< concepts::matrix_data_v<M> && concepts::vector_data_v<V> >,
           typename = enable_if_t<true>,
           typename = enable_if_t<true>,
           typename = enable_if_t<true> >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const M& m, const V& v )
  noexcept( noexcept( operations::template vector_matrix_product<V,M>::prod(m,v) ) )
{
  return operations::template vector_matrix_product<V,M>::prod(m,v);
}

//=================================================================================================
//  Matrix Matrix product
//=================================================================================================
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::matrix_data M1, concepts::matrix_data M2 >
#else
template < class M1, class M2,
           typename = enable_if_t< concepts::matrix_data_v<M1> && concepts::matrix_data_v<M2> >,
           typename = enable_if_t<true>,
           typename = enable_if_t<true>,
           typename = enable_if_t<true>,
           typename = enable_if_t<true> >
#endif
[[nodiscard]] inline constexpr decltype(auto)
operator * ( const M1& m1, const M2& m2 )
  noexcept( noexcept( operations::template matrix_matrix_product<M1,M2>::prod(m1,m2) ) )
{
  return operations::template matrix_matrix_product<M1,M2>::prod(m1,m2);
}

#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::matrix_data M1, concepts::matrix_data M2 >
#else
template < class M1, class M2,
           typename = enable_if_t< concepts::matrix_data_v<M1> &&
                                   concepts::matrix_data_v<M2> &&
                                   detail::extents_may_be_equal_v< typename M1::extents_type, typename decltype( declval<M1>() * declval<M2>() )::extents_type > > >
#endif
[[nodiscard]] inline constexpr M1&
operator *= ( M1& m1, const M2& m2 )
  noexcept( noexcept( operations::template matrix_matrix_product<M1,M2>::prod(m1,m2) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires detail::extents_may_be_equal_v< typename M1::extents_type, typename decltype( const_cast<const M1&>(m1) * m2 )::extents_type >
#endif
{
  return operations::template matrix_matrix_product<M1,M2>::prod(m1,m2);
}

}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_ARITHMETIC_OPERATORS_HPP
