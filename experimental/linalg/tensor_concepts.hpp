#ifndef LINEAR_ALGEBRA_TENSOR_CONCEPTS_HPP
#define LINEAR_ALGEBRA_TENSOR_CONCEPTS_HPP

#include <linear_algebra.hpp>

namespace std
{
namespace math
{
namespace concepts
{

//=================================================================================================
//  Tensor Concepts
//=================================================================================================

// All tensor data types must specify a value and index type
// All tensor data types must specify an extents type which must be of type std::extents<...>
// All tensor data types must support size / capacity const access functions
template < class T >
concept tensor_data = requires                       // Types
{
  typename T::value_type;
  typename T::index_type;
  typename T::size_type;
  typename T::extents_type;
} &&
( detail::is_extents_v<typename T::extents_type> ) &&
requires( T& t )                                     // Functions
{
  { t.size() }     noexcept;
  { t.capacity() } noexcept;
} &&
( is_same_v< decltype( declval<T>().size() ), typename T::extents_type > ||
  ( is_same_v< decltype( declval<T>().size() ), typename T::size_type > && ( T::extents_type::rank() == 1 ) ) ) &&
( is_same_v< decltype( declval<T>().capacity() ), typename T::extents_type > ||
  ( is_same_v< decltype( declval<T>().capacity() ), typename T::size_type > && ( T::extents_type::rank() == 1 ) ) );

// All tensor types are tensor data types
// All tensor types must support scalar multiply and divide operations
// All tensor types must support unary functions: negation
// All tensor types must support binary functions: addition and subtraction
template < class T >
concept tensor = tensor_data<T> &&
requires ( T& t, typename T::value_type v ) // Scalar functions
{
  v * t;                                    // Return type should also be a tensor or a derived type
  t * v;                                    // Return type should also be a tensor or a derived type
  t / v;                                    // Return type should also be a tensor or a derived type
} &&
requires ( T t )                            // Unary operations
{
  -t;                                       // Return type should also be a tensor or a derived type
} &&
requires( T t )                             // Binary operations
{
  t + t;                                    // Return type should also be a tensor or a derived type
  t - t;                                    // Return type should also be a tensor or a derived type
};

// Readable tensor data must A) be a tensor data
// Readable tensor data must B) provide a tuple type of extents_type::rank() index_type
// Readable tensor data must C) provide index access
// Readable tensor data must D) provide a const span representation
// Readable tensor data must E) provide an implementation defined view of the underlying memory.
// Readable tensor data must F) provide a const subtensor view
template < class T >
concept readable_tensor_data = tensor_data<T> && // A)
requires                                         // B)
{
  typename T::tuple_type;
} &&
detail::is_homogeneous_tuple_v< typename T::tuple_type > &&
( tuple_size_v< typename T::tuple_type > == T::extents_type::rank() ) &&
requires( const T& t )                           // C)
{
  // TODO: language support doesn't really exist for enforcing:
  // { e[indices...] } noexcept -> same_as<typename ET::value_type>
  // { e.at(indices...) } -> same_as<typename ET::value_type>
  //   given sizeof(indices...) = ET::extents_type::rank()
  // Parameter pack not allowed in requires parameter clause
  t;
} &&
requires( const T& t )                           // D)
{
  typename T::span_type;
  { t.span() }               noexcept -> same_as<typename T::span_type>;
} &&
detail::is_mdspan_v< typename T::span_type > && // span_type must be an mdspan
requires( const T& t )                           // E)
{
  typename T::const_underlying_span_type;
  { t.underlying_span() }    noexcept -> same_as<typename T::const_underlying_span_type>;
} &&
requires( const T&               t,
          typename T::tuple_type start,
          typename T::tuple_type end )           // F)
{
  typename T::const_subtensor_type;
  { t.subtensor(start,end) } -> same_as<typename T::const_subtensor_type>;
  { t.subtensor(start,end) } -> tensor_data;
};

// Readable tensor must be a readable tensor data
// Readable tensor must be a tensor
template < class T >
concept readable_tensor = readable_tensor_data<T> && tensor<T>;

// Writable tensor data must A) be a readable tensor
// Writable tensor data must B) provide a mutable scalar view
// Writable tensor data must C) provide an implementation defined mutable view of the underlying memory.
// Writable tensor data must D) provide a mutable subtensor view
template < class T >
concept writable_tensor_data = readable_tensor_data<T> && // A)
requires( T& t )                                          // B)
{
  typename T::reference_type;
  // TODO: language support doesn't really exist for enforcing:
  // { e[indices...] } noexcept -> same_as<typename ET::reference_type>
  // { e.at(indices...) } -> same_as<typename ET::reference_type>
  //   given sizeof(indices...) = ET::extents_type::rank()
  // Parameter pack not allowed in requires parameter clause
} &&
( !is_const_v< remove_reference_t< typename T::reference_type > > ) &&
requires( T& t )                                          // C)
{
  typename T::underlying_span_type;
  { t.underlying_span() }    noexcept -> same_as<typename T::underlying_span_type>;
} &&
requires( T&                     t,
          typename T::tuple_type start,
          typename T::tuple_type end )                    // D)
{
  typename T::subtensor_type;
  { t.subtensor(start,end) } -> same_as<typename T::subtensor_type>;
  { t.subtensor(start,end) } -> tensor_data;
};

// Writable tensor must be a writable tensor data
// Writable tensor must be a tensor
template < class T >
concept writable_tensor = writable_tensor_data<T> && readable_tensor<T>;

// Dynamic tensor data
// Dynamic tensor data must be resizable
// Dynamic tensor data must be reservable
// Dynamic tensor data must define an allocator type
// Dynamic tensor data must be constructible with an allocator
// Dynamic tensor data must be constructible with an allocator and size
// Dynamic tensor data must be constructible with an allocator and size, capacity
// Dynamic tensor data must provide mutable/const access to underlying allocator
template < class T >
concept dynamic_tensor_data = tensor_data<T> && requires
{
  typename T::allocator_type;
} &&
requires( T&                                t,
          typename T::extents_type          size,
          typename T::extents_type          cap,
          const typename T::allocator_type& alloc )
{
  { T( alloc )};
  { T( size, alloc ) };
  { T( size, cap, alloc ) };
  { t.set_allocator( alloc ) };
  { static_cast<const T&>(t).get_allocator() } -> same_as<const typename T::allocator_type&>;
} &&
( requires( T&                       t,
            typename T::extents_type size,
            typename T::extents_type cap )
  {
    { t.resize( size ) };
    { t.reserve( cap ) };
  } ||
  ( requires( T&                    t,
              typename T::size_type size,
              typename T::size_type cap )
    {
      { t.resize( size ) };
      { t.reserve( cap ) };
    } && ( T::extents_type::rank() == 1 )
  )
);

// Dynamic tensor must be a dynamic tensor data
// Dynamic tensor must be a tensor
template < class T >
concept dynamic_tensor = dynamic_tensor_data<T> && tensor<T>;

// Fixed size tensor data
// A fixed size tensor data must be compile-time constructible
// A fixed size tensor data must support constant expression size functions.
// A fixed size tensor data must support constant expression capacity functions.
// A fixed size tensor data must have equivalent size and capacity.
template < class T >
concept fixed_size_tensor_data = tensor_data<T> &&
detail::extents_is_static_v<typename T::extents_type> &&
detail::is_constexpr( []{ T(); } ) &&
detail::is_constexpr( []{ decltype( declval<T>().size() )     nodiscard_warning = T().size(); } ) &&
detail::is_constexpr( []{ decltype( declval<T>().capacity() ) nodiscard_warning = T().capacity(); } ) &&
( T().size() == T().capacity() );

// Fixed size tensor must be a fixed size tensor data
// Fixed size tensor must be a tensor
template < class T >
concept fixed_size_tensor = fixed_size_tensor_data<T> && tensor<T>;

//=================================================================================================
//  Additional Tensor Concepts
//=================================================================================================

// Tensor may be constructible
// Enforces both types are tensor which have constructible elements
// and both have potentially compatible extents
template < class From, class To >
concept tensor_may_be_constructible =
tensor<From> && tensor<To> &&
( is_constructible_v< typename From::value_type, typename To::value_type > ) &&
detail::extents_may_be_equal_v< typename From::extents_type, typename To::extents_type >;

// View may be constructible to tensor
// Enforces view is an mdspan of rank equal to tensor with elements
//   Each extent must be the same or dynamic
// View element must be constructible to the tensor elements
template < class From, class To >
concept view_may_be_constructible_to_tensor =
detail::is_mdspan_v<From> && tensor<To> &&
( is_constructible_v<typename From::value_type,typename To::value_type> ) &&
detail::extents_may_be_equal_v< typename From::extents_type, typename To::extents_type >;

// View is constructible to tensor
// Enforces view is an mdspan of rank equal to tensor with elements
//   Each extent must be the same (and not dynamic)
// View element must be constructible to the tensor elements
template < class From, class To >
concept view_is_constructible_to_tensor =
detail::is_mdspan_v<From> && tensor<To> &&
( is_constructible_v<typename From::value_type,typename To::value_type> ) &&
detail::extents_is_equal_v< typename From::extents_type, typename To::extents_type >;

// View is nothrow constructible to tensor
// Enforces view is an mdspan of rank equal to tensor with elements
//   Each extent must be the same (and not dynamic)
// View element must be nothrow constructible to the tensor elements
template < class From, class To >
concept view_is_nothrow_constructible_to_tensor =
detail::is_mdspan_v<From> && tensor<To> &&
( is_nothrow_constructible_v<typename From::value_type,typename To::value_type> ) &&
detail::extents_is_equal_v< typename From::extents_type, typename To::extents_type >;

}       //- concepts namespace
}       //- math namespace
}       //- std namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_CONCEPTS_HPP
