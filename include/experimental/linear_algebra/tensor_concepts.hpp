#ifndef LINEAR_ALGEBRA_TENSOR_CONCEPTS_HPP
#define LINEAR_ALGEBRA_TENSOR_CONCEPTS_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{
namespace concepts
{

#ifdef LINALG_ENABLE_CONCEPTS

//=================================================================================================
//  Tensor Concepts
//=================================================================================================

// All tensor data types must specify a value and index type
// All tensor data types must specify an extents type which must be of type ::std::extents<...>
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
is_same_v< decltype( ::std::declval<T>().size() ), typename T::extents_type > &&
is_same_v< decltype( ::std::declval<T>().capacity() ), typename T::extents_type >;

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
// Note: subtensor(...) does not have an easily enforceable interface
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
  { t.span() }               noexcept -> ::std::same_as<typename T::span_type>;
} &&
detail::is_mdspan_v< typename T::span_type > && // span_type must be an mdspan
requires( const T& t )                           // E)
{
  typename T::const_underlying_span_type;
  { t.underlying_span() }    noexcept -> ::std::same_as<typename T::const_underlying_span_type>;
};

// Readable tensor must be a readable tensor data
// Readable tensor must be a tensor
template < class T >
concept readable_tensor = readable_tensor_data<T> && tensor<T>;

// Writable tensor data must A) be a readable tensor
// Writable tensor data must B) provide a mutable scalar view
// Writable tensor data must C) provide an implementation defined mutable view of the underlying memory.
// Writable tensor data must D) provide a mutable subtensor view
// Note: subtensor(...) does not have an easily enforceable interface
template < class T >
concept writable_tensor_data = readable_tensor_data<T> && // A)
requires( T& t )                                          // B)
{
  typename T::reference;
  // TODO: language support doesn't really exist for enforcing:
  // { e[indices...] } noexcept -> same_as<typename ET::reference>
  // { e.at(indices...) } -> same_as<typename ET::reference>
  //   given sizeof(indices...) = ET::extents_type::rank()
  // Parameter pack not allowed in requires parameter clause
} &&
( !::std::is_const_v< ::std::remove_reference_t< typename T::reference > > ) &&
requires( T& t )                                          // C)
{
  typename T::underlying_span_type;
  { t.underlying_span() }    noexcept -> ::std::same_as<typename T::underlying_span_type>;
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
  { static_cast<const T&>(t).get_allocator() } -> ::std::same_as<const typename T::allocator_type&>;
} &&
requires( T&                       t,
          typename T::extents_type size,
          typename T::extents_type cap )
{
  { t.resize( size ) };
  { t.reserve( cap ) };
};

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
detail::is_constexpr( []{ decltype( ::std::declval<T>().size() )     nodiscard_warning = T().size(); } ) &&
detail::is_constexpr( []{ decltype( ::std::declval<T>().capacity() ) nodiscard_warning = T().capacity(); } ) &&
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
( ::std::is_constructible_v< typename From::value_type, typename To::value_type > ) &&
detail::extents_may_be_equal_v< typename From::extents_type, typename To::extents_type >;

// View may be constructible to tensor
// Enforces view is an mdspan of rank equal to tensor with elements
//   Each extent must be the same or dynamic
// View element must be constructible to the tensor elements
template < class From, class To >
concept view_may_be_constructible_to_tensor =
detail::is_mdspan_v<From> && tensor<To> &&
( ::std::is_constructible_v<typename From::value_type,typename To::value_type> ) &&
detail::extents_may_be_equal_v< typename From::extents_type, typename To::extents_type >;

// View is constructible to tensor
// Enforces view is an mdspan of rank equal to tensor with elements
//   Each extent must be the same (and not dynamic)
// View element must be constructible to the tensor elements
template < class From, class To >
concept view_is_constructible_to_tensor =
detail::is_mdspan_v<From> && tensor<To> &&
( ::std::is_constructible_v<typename From::value_type,typename To::value_type> ) &&
detail::extents_are_equal_v< typename From::extents_type, typename To::extents_type >;

// View is nothrow constructible to tensor
// Enforces view is an mdspan of rank equal to tensor with elements
//   Each extent must be the same (and not dynamic)
// View element must be nothrow constructible to the tensor elements
template < class From, class To >
concept view_is_nothrow_constructible_to_tensor =
detail::is_mdspan_v<From> && tensor<To> &&
( ::std::is_nothrow_constructible_v<typename From::value_type,typename To::value_type> ) &&
detail::extents_are_equal_v< typename From::extents_type, typename To::extents_type >;

#else

//- Tests for aliases

// Test if T has alias value_type
template < class T, class = void > struct has_value_type : public ::std::false_type { };
template < class T > struct has_value_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::value_type, typename T::value_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_value_type_v = has_value_type<T>::value;

// Test if T has alias index_type
template < class T, class = void > struct has_index_type : public ::std::false_type { };
template < class T > struct has_index_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::index_type, typename T::index_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_index_type_v = has_index_type<T>::value;

// Test if T has alias size_type
template < class T, class = void > struct has_size_type : public ::std::false_type { };
template < class T > struct has_size_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::size_type, typename T::size_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_size_type_v = has_size_type<T>::value;

// Test if T has alias extents_type
template < class T, class = void > struct has_extents_type : public ::std::false_type { };
template < class T > struct has_extents_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::extents_type, typename T::extents_type > && detail::is_extents_v< typename T::extents_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_extents_type_v = has_extents_type<T>::value;

// Test if T has alias tuple_type
template < class T, class = void > struct has_tuple_type : public ::std::false_type { };
template < class T > struct has_tuple_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::tuple_type, typename T::tuple_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_tuple_type_v = has_tuple_type<T>::value;

// Test if T has alias span_type
template < class T, class = void > struct has_span_type : public ::std::false_type { };
template < class T > struct has_span_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::span_type, typename T::span_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_span_type_v = has_span_type<T>::value;

// Test if T has alias reference
template < class T, class = void > struct has_reference : public ::std::false_type { };
template < class T > struct has_reference< T, ::std::enable_if_t< ::std::is_same_v< typename T::reference, typename T::reference > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_reference_v = has_reference<T>::value;

// Test if T has alias underlying_span_type
template < class T, class = void > struct has_underlying_span_type : public ::std::false_type { };
template < class T > struct has_underlying_span_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::underlying_span_type, typename T::underlying_span_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_underlying_span_type_v = has_underlying_span_type<T>::value;

// Test if T has alias const_underlying_span_type
template < class T, class = void > struct has_const_underlying_span_type : public ::std::false_type { };
template < class T > struct has_const_underlying_span_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::const_underlying_span_type, typename T::const_underlying_span_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_const_underlying_span_type_v = has_const_underlying_span_type<T>::value;


// Test if T has alias allocator_type
template < class T, class = void > struct has_allocator_type : public ::std::false_type { };
template < class T > struct has_allocator_type< T, ::std::enable_if_t< ::std::is_same_v< typename T::allocator_type, typename T::allocator_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_allocator_type_v = has_allocator_type<T>::value;

//- Test for functions

// Test for size function
template < class T, class = void > struct has_size_func : public ::std::false_type { };
template < class T > struct has_size_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().size() ), typename T::extents_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_size_func_v = has_size_func<T>::value;

// Test for capacity function
template < class T, class = void > struct has_capacity_func : public ::std::false_type { };
template < class T > struct has_capacity_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().capacity() ), typename T::extents_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_capacity_func_v = has_capacity_func<T>::value;

template < class T, typename = void > struct value_type_helper { using type = void; };
template < class T > struct value_type_helper< T, ::std::enable_if_t< has_value_type_v<T> > > { using type = typename T::value_type; };

// Test for scalar pre-multiply
template < class T, class S = void, class = void > struct has_scalar_premultiply_func : public ::std::false_type { };
template < class T, class S > struct has_scalar_premultiply_func< T, S, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<S>() * ::std::declval<const T>() ), decltype( ::std::declval<S>() * ::std::declval<const T>() ) > > > : public ::std::true_type { };
template < class T, class S = typename value_type_helper<T>::type > inline constexpr bool has_scalar_premultiply_func_v = has_scalar_premultiply_func<T,S>::value;

// Test for scalar post-multiply
template < class T, class S = void, class = void > struct has_scalar_postmultiply_func : public ::std::false_type { };
template < class T, class S > struct has_scalar_postmultiply_func< T, S, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>() * ::std::declval<S>() ), decltype( ::std::declval<const T>() * ::std::declval<S>() ) > > > : public ::std::true_type { };
template < class T, class S = typename value_type_helper<T>::type > inline constexpr bool has_scalar_postmultiply_func_v = has_scalar_postmultiply_func<T,S>::value;

// Test for scalar divide
template < class T, class S = void, class = void > struct has_scalar_divide_func : public ::std::false_type { };
template < class T, class S > struct has_scalar_divide_func< T, S, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>() / ::std::declval<S>() ), decltype( ::std::declval<const T>() / ::std::declval<S>() ) > > > : public ::std::true_type { };
template < class T, class S = typename value_type_helper<T>::type > inline constexpr bool has_scalar_divide_func_v = has_scalar_divide_func<T,S>::value;

// Test for negation
template < class T, class = void > struct has_negate_func : public ::std::false_type { };
template < class T > struct has_negate_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( -declval<const T>() ), decltype( -declval<const T>() ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_negate_func_v = has_negate_func<T>::value;

// Test for addition
template < class T, class = void > struct has_add_func : public ::std::false_type { };
template < class T > struct has_add_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>() + ::std::declval<const T>() ), decltype( ::std::declval<const T>() + ::std::declval<const T>() ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_add_func_v = has_add_func<T>::value;

// Test for subtraction
template < class T, class = void > struct has_subtract_func : public ::std::false_type { };
template < class T > struct has_subtract_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>() - ::std::declval<const T>() ), decltype( ::std::declval<const T>() - ::std::declval<const T>() ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_subtract_func_v = has_subtract_func<T>::value;

// Test for span function
template < class T, class = void > struct has_span_func : public ::std::false_type { };
template < class T > struct has_span_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().span() ), const typename T::span_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_span_func_v = has_span_func<T>::value;

// Test for underlying span function
template < class T, class = void > struct has_underlying_span_func : public ::std::false_type { };
template < class T > struct has_underlying_span_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<T>().underlying_span() ), typename T::underlying_span_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_underlying_span_func_v = has_underlying_span_func<T>::value;

// Test for const underlying span function
template < class T, class = void > struct has_const_underlying_span_func : public ::std::false_type { };
template < class T > struct has_const_underlying_span_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().underlying_span() ), typename T::underlying_span_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_const_underlying_span_func_v = has_const_underlying_span_func<T>::value;

// Test for get_allocator function
template < class T, class = void > struct has_get_allocator_func : public ::std::false_type { };
template < class T > struct has_get_allocator_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<const T>().get_allocator() ), const typename T::allocator_type& > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_get_allocator_func_v = has_get_allocator_func<T>::value;

// Test for set_allocator function
template < class T, class = void > struct has_set_allocator_func : public ::std::false_type { };
template < class T > struct has_set_allocator_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<T>().set_allocator( ::std::declval<const typename T::allocator_type&>() ) ), decltype( ::std::declval<T>().set_allocator( ::std::declval<const typename T::allocator_type&>() ) ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_set_allocator_func_v = has_set_allocator_func<T>::value;

// Test for resize function
template < class T, class = void > struct has_resize_func : public ::std::false_type { };
template < class T > struct has_resize_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<T>().resize( ::std::declval<typename T::extents_type>() ) ), decltype( ::std::declval<T>().resize( ::std::declval<typename T::extents_type>() ) ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_resize_func_v = has_resize_func<T>::value;

// Test for reserve function
template < class T, class = void > struct has_reserve_func : public ::std::false_type { };
template < class T > struct has_reserve_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<T>().reserve( ::std::declval<typename T::extents_type>() ) ), decltype( ::std::declval<T>().reserve( ::std::declval<typename T::extents_type>() ) ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_reserve_func_v = has_reserve_func<T>::value;

// Test for construct from allocator
template < class T, class = void > struct constructible_from_alloc : public ::std::false_type { };
template < class T > struct constructible_from_alloc< T, ::std::enable_if_t< is_constructible_v< T, typename T::allocator_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool constructible_from_alloc_v = constructible_from_alloc<T>::value;

// Test for construct from size and allocator
template < class T, class = void > struct constructible_from_size_and_alloc : public ::std::false_type { };
template < class T > struct constructible_from_size_and_alloc< T, ::std::enable_if_t< is_constructible_v< T, typename T::extents_type, typename T::allocator_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool constructible_from_size_and_alloc_v = constructible_from_size_and_alloc<T>::value;

// Test for construct from size capacity and allocator
template < class T, class = void > struct constructible_from_size_cap_and_alloc : public ::std::false_type { };
template < class T > struct constructible_from_size_cap_and_alloc< T, ::std::enable_if_t< is_constructible_v< T, typename T::extents_type, typename T::extents_type, typename T::allocator_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool constructible_from_size_cap_and_alloc_v = constructible_from_size_cap_and_alloc<T>::value;

// Test for transpose
template < class T, class = void > struct has_trans_func : public ::std::false_type { };
template < class T > struct has_trans_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( trans( ::std::declval<T>() ) ), decltype( trans( ::std::declval<T>() ) ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_trans_func_v = has_trans_func<T>::value;

// Test for conjugate
template < class T, class = void > struct has_conj_func : public ::std::false_type { };
template < class T > struct has_conj_func< T, ::std::enable_if_t< ::std::is_same_v< decltype( conj( ::std::declval<T>() ) ), decltype( conj( ::std::declval<T>() ) ) > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_conj_func_v = has_conj_func<T>::value;

//- Additional tests

// Test for extents which may be equal
template < class T, class U, class = void > struct extents_may_be_equal : public ::std::false_type { };
template < class T, class U > struct extents_may_be_equal< T, U, ::std::enable_if_t< detail::extents_may_be_equal_v< typename T::extents_type, typename U::extents_type > > > : public ::std::true_type { };
template < class T, class U > inline constexpr bool extents_may_be_equal_v = extents_may_be_equal<T,U>::value;

// Test for extents which are equal
template < class T, class U, class = void > struct extents_are_equal : public ::std::false_type { };
template < class T, class U > struct extents_are_equal< T, U, ::std::enable_if_t< detail::extents_are_equal_v< typename T::extents_type, typename U::extents_type > > > : public ::std::true_type { };
template < class T, class U > inline constexpr bool extents_are_equal_v = extents_are_equal<T,U>::value;

// Test values are convertible
template < class T, class U, class = void > struct values_are_constructible : public ::std::false_type { };
template < class T, class U > struct values_are_constructible< T, U, ::std::enable_if_t< is_constructible_v< typename T::value_type, typename U::value_type > > > : public ::std::true_type { };
template < class T, class U > inline constexpr bool values_are_constructible_v = values_are_constructible<T,U>::value;

// Test values are convertible
template < class T, class U, class = void > struct values_are_nothrow_constructible : public ::std::false_type { };
template < class T, class U > struct values_are_nothrow_constructible< T, U, ::std::enable_if_t< is_nothrow_constructible_v< typename T::value_type, typename U::value_type > > > : public ::std::true_type { };
template < class T, class U > inline constexpr bool values_are_nothrow_constructible_v = values_are_nothrow_constructible<T,U>::value;

// Test addition exists
template < class T, class U, class = void > struct addition_exists : public ::std::false_type { };
template < class T, class U > struct addition_exists< T, U, ::std::enable_if_t< is_same_v< decltype( ::std::declval<T>() + ::std::declval<U>() ), decltype( ::std::declval<T>() + ::std::declval<U>() ) > > > : public ::std::true_type { };
template < class T, class U > inline constexpr bool addition_exists_v = addition_exists<T,U>::value;

// Test subtraction exists
template < class T, class U, class = void > struct subtraction_exists : public ::std::false_type { };
template < class T, class U > struct subtraction_exists< T, U, ::std::enable_if_t< is_same_v< decltype( ::std::declval<T>() - ::std::declval<U>() ), decltype( ::std::declval<T>() - ::std::declval<U>() ) > > > : public ::std::true_type { };
template < class T, class U > inline constexpr bool subtraction_exists_v = subtraction_exists<T,U>::value;

// Test product exists
template < class T, class U, class = void > struct product_exists : public ::std::false_type { };
template < class T, class U > struct product_exists< T, U, ::std::enable_if_t< is_same_v< decltype( ::std::declval<T>() * ::std::declval<U>() ), decltype( ::std::declval<T>() * ::std::declval<U>() ) > > > : public ::std::true_type { };
template < class T, class U > inline constexpr bool product_exists_v = product_exists<T,U>::value;

// Test division exists
template < class T, class U, class = void > struct division_exists : public ::std::false_type { };
template < class T, class U > struct division_exists< T, U, ::std::enable_if_t< is_same_v< decltype( ::std::declval<T>() / ::std::declval<U>() ), decltype( ::std::declval<T>() / ::std::declval<U>() ) > > > : public ::std::true_type { };
template < class T, class U > inline constexpr bool division_exists_v = division_exists<T,U>::value;

// Test for rank N tensor
template < class T, auto N, class = void > struct is_rank : public ::std::false_type { };
template < class T, auto N > struct is_rank< T, N, ::std::enable_if_t< T::extents_type::rank() == N > > : public ::std::true_type { };
template < class T, auto N > inline constexpr bool is_rank_v = is_rank<T,N>::value;

// Test for constant tensor
template < class T, class = void > struct is_const_tensor : public ::std::false_type { };
template < class T > struct is_const_tensor< T, ::std::enable_if_t< ::std::is_const_v< ::std::remove_reference_t< typename T::reference > > > > : public ::std::true_type { };
template < class T > inline constexpr bool is_const_tensor_v = is_const_tensor<T>::value;

// Test for static extents
template < class T, class = void > struct has_static_extents : public ::std::false_type { };
template < class T > struct has_static_extents< T, ::std::enable_if_t< detail::extents_is_static_v<typename T::extents_type> > > : public ::std::true_type { };
template < class T > inline constexpr bool has_static_extents_v = has_static_extents<T>::value;

// Test for homogeneous tuple type
template < class T, class = void > struct has_homogeneous_tuple_type : public ::std::false_type { };
template < class T > struct has_homogeneous_tuple_type< T, ::std::enable_if_t< detail::is_homogeneous_tuple_v< typename T::tuple_type > > > : public ::std::true_type { };
template < class T > inline constexpr bool has_homogeneous_tuple_type_v = has_homogeneous_tuple_type<T>::value;

// Test for capabitible tuple size
template < class T, class = void > struct has_capatible_tuple_size : public ::std::false_type { };
template < class T > struct has_capatible_tuple_size< T, ::std::enable_if_t< ( tuple_size_v< typename T::tuple_type > == T::extents_type::rank() ) > > : public ::std::true_type { };
template < class T > inline constexpr bool has_capatible_tuple_size_v = has_capatible_tuple_size<T>::value;

//- Test for tensors

// Tensor data
template < class T > struct tensor_data : public ::std::conditional_t< 
  has_value_type_v<T> &&
  has_index_type_v<T> &&
  has_size_type_v<T> &&
  has_extents_type_v<T> &&
  has_size_func_v<T> &&
  has_capacity_func_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool tensor_data_v = tensor_data<T>::value;

// Tensor
template < class T > struct tensor : public ::std::conditional_t< 
  tensor_data_v<T> &&
  has_scalar_premultiply_func_v<T> &&
  has_scalar_postmultiply_func_v<T> &&
  has_scalar_divide_func_v<T> &&
  has_negate_func_v<T> &&
  has_add_func_v<T> &&
  has_subtract_func_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool tensor_v = tensor<T>::value;

// Readable tensor data
template < class T > struct readable_tensor_data : public ::std::conditional_t< 
  tensor_data_v<T> &&
  has_tuple_type_v<T> &&
  has_homogeneous_tuple_type_v<T> &&
  has_capatible_tuple_size_v<T> &&
  has_span_type_v<T> &&
  has_const_underlying_span_type_v<T> &&
  has_span_func_v<T> &&
  has_const_underlying_span_func_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool readable_tensor_data_v = readable_tensor_data<T>::value;

// Readable tensor
template < class T > struct readable_tensor : public ::std::conditional_t< 
  readable_tensor_data_v<T> &&
  tensor_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool readable_tensor_v = readable_tensor<T>::value;

// Writable tensor data
template < class T > struct writable_tensor_data : public ::std::conditional_t< 
  readable_tensor_data_v<T> &&
  has_reference_v<T> &&
  is_const_tensor_v<T> &&
  has_underlying_span_type_v<T> &&
  has_underlying_span_func_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool writable_tensor_data_v = writable_tensor_data<T>::value;

// Writable tensor
template < class T > struct writable_tensor : public ::std::conditional_t< 
  writable_tensor_data_v<T> &&
  tensor_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool writable_tensor_v = writable_tensor<T>::value;

// Dynamic tensor data
template < class T > struct dynamic_tensor_data : public ::std::conditional_t<
  tensor_data_v<T> &&
  has_allocator_type_v<T> /*&&
  constructible_from_alloc_v<T> &&
  constructible_from_size_and_alloc_v<T> &&
  constructible_from_size_cap_and_alloc_v<T>*/ &&
  has_resize_func_v<T> &&
  has_reserve_func_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool dynamic_tensor_data_v = dynamic_tensor_data<T>::value;

// Dynamic tensor
template < class T > struct dynamic_tensor : public ::std::conditional_t< 
  dynamic_tensor_data_v<T> &&
  tensor_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool dynamic_tensor_v = dynamic_tensor<T>::value;

// Fixed size tensor data
template < class T > struct fixed_size_tensor_data : public ::std::conditional_t< 
  tensor_data_v<T> &&
  has_static_extents_v<T>
#if LINALG_HAS_CXX_20
  && detail::is_constexpr( []{ T(); } ) &&
  detail::is_constexpr( []{ [[maybe_unused]] decltype( ::std::declval<T>().size() )     nodiscard_warning = T().size(); } ) &&
  detail::is_constexpr( []{ [[maybe_unused]] decltype( ::std::declval<T>().capacity() ) nodiscard_warning = T().capacity(); } ) &&
  ( T().size() == T().capacity() )
#endif
  , ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool fixed_size_tensor_data_v = fixed_size_tensor_data<T>::value;

// Fixed size tensor
template < class T > struct fixed_size_tensor : public ::std::conditional_t< 
  fixed_size_tensor_data_v<T> &&
  tensor_v<T>, ::std::true_type, ::std::false_type > { };
template < class T > inline constexpr bool fixed_size_tensor_v = fixed_size_tensor<T>::value;

// Tensor may be constructible
// Enforces both types are tensor which have constructible elements
// and both have potentially compatible extents
template < class From, class To >
inline constexpr bool tensor_may_be_constructible =
tensor_v<From> && tensor_v<To> &&
values_are_constructible_v<From,To> &&
extents_may_be_equal_v<From,To>;

// View may be constructible to tensor
// Enforces view is an mdspan of rank equal to tensor with elements
//   Each extent must be the same or dynamic
// View element must be constructible to the tensor elements
template < class From, class To >
inline constexpr bool view_may_be_constructible_to_tensor =
detail::is_mdspan_v<From> && tensor_v<To> &&
values_are_constructible_v<From,To> &&
extents_may_be_equal_v<From,To>;

// View is constructible to tensor
// Enforces view is an mdspan of rank equal to tensor with elements
//   Each extent must be the same (and not dynamic)
// View element must be constructible to the tensor elements
template < class From, class To >
inline constexpr bool view_is_constructible_to_tensor =
detail::is_mdspan_v<From> && tensor_v<To> &&
values_are_constructible_v<From,To> &&
extents_are_equal_v<From,To>;

// View is nothrow constructible to tensor
// Enforces view is an mdspan of rank equal to tensor with elements
//   Each extent must be the same (and not dynamic)
// View element must be nothrow constructible to the tensor elements
template < class From, class To >
inline constexpr bool view_is_nothrow_constructible_to_tensor =
detail::is_mdspan_v<From> && tensor_v<To> &&
values_are_nothrow_constructible_v<From,To> &&
extents_are_equal_v< From, To >;

#endif

}       //- concepts namespace
}       //- math namespace
}       //- std namespace

#endif  //- LINEAR_ALGEBRA_TENSOR_CONCEPTS_HPP
