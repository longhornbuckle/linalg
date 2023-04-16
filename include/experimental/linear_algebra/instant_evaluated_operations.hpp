//==================================================================================================
//  File:       instant_evaluated_operations.hpp
//
//  Summary:    Provides definitions for unary and binary operations on tensors, matrices, vectors.
//              Operations are defined such that the result is immediately computed.
//==================================================================================================

#ifndef LINEAR_ALGEBRA_INSTANT_EVALUATED_OPERATIONS_HPP
#define LINEAR_ALGEBRA_INSTANT_EVALUATED_OPERATIONS_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{
namespace instant_evaluated_operations
{

template < class T, class ValueType > struct default_dynamic;
template < class MDS, class ValueType >
struct default_dynamic< tensor_view< MDS >, ValueType >
{
  using type = dr_tensor< ValueType,
                          MDS::extents_type::rank(),
                          ::std::allocator< ValueType >,
                          default_layout,
                          typename detail::rebind_accessor_t<typename MDS::accessor_type,ValueType> >;
};
template < class MDS, class ValueType >
struct default_dynamic< matrix_view< MDS >, ValueType >
{
  using type = dr_matrix< ValueType,
                          ::std::allocator< ValueType >,
                          default_layout,
                          typename detail::rebind_accessor_t<typename MDS::accessor_type,ValueType> >;
};
template < class MDS, class ValueType >
struct default_dynamic< vector_view< MDS >, ValueType >
{
  using type = dr_vector< ValueType,
                          ::std::allocator< ValueType >,
                          default_layout,
                          typename detail::rebind_accessor_t<typename MDS::accessor_type,ValueType> >;
};
template < class T, class V >
using default_dynamic_t = typename default_dynamic<T,V>::type;


/// @brief Defines negation operation on a tensor
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_data T >
#else
template < class T, typename = ::std::enable_if_t< concepts::tensor_data_v<T> > >
#endif
class negation
{
  public:
    //- Types

    /// @brief Input tensor type
    using tensor_type        = T;
  private:
    // Aliases
    using result_value_type  = ::std::decay_t< decltype( - ::std::declval<typename tensor_type::value_type>() ) >;
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class U > requires ( !( concepts::fixed_size_tensor_data<U> || concepts::dynamic_tensor_data<U> ) )
    #else
    template < class U, typename = void >
    #endif
    struct Result_tensor { using type = default_dynamic_t<U,result_value_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::fixed_size_tensor_data U >
    struct Result_tensor
    #else
    template < class U  >
    struct Result_tensor< U, ::std::enable_if_t< concepts::fixed_size_tensor_data_v<U> > >
    #endif
    { using type = typename U::template rebind_t<result_value_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::dynamic_tensor_data U >
    struct Result_tensor
    #else
    template < class U >
    struct Result_tensor< U, ::std::enable_if_t< concepts::dynamic_tensor_data_v<U> > >
    #endif
    { using type = typename U::template rebind_t<result_value_type>; };
    using result_tensor_type = typename Result_tensor< tensor_type >::type;
    // Gets necessary arguments for constrution
    // If tensor type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_tensor = result_tensor_type,
               typename = ::std::enable_if_t< concepts::fixed_size_tensor_data_v<Result_tensor> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const tensor_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_tensor_data<result_tensor_type>
    #endif
    { return tuple( forward<Lambda>( lambda ) ); }
    // If the tensor type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default constructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_tensor = result_tensor_type,
               typename = ::std::enable_if_t< concepts::dynamic_tensor_data_v<Result_tensor> >, typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const tensor_type& t, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_tensor_data<result_tensor_type>
    #endif
    {
      if constexpr ( ::std::is_default_constructible_v<typename result_tensor_type::allocator_type> &&
                     ::std::allocator_traits<typename result_tensor_type::allocator_type>::is_always_equal::value )
      {
        return ::std::tuple( t.size(), t.capacity(), ::std::forward<Lambda>( lambda ) );
      }
      else
      {
        using result_alloc_type = typename ::std::allocator_traits<typename tensor_type::allocator_type>::template rebind_alloc<result_value_type>;
        return ::std::tuple( t.size(), t.capacity(), ::std::forward<Lambda>( lambda ), result_alloc_type( t.get_allocator() ) );
      }
    }
  public:
    //- Operations

    /// @brief Returns -1 * tensor
    [[nodiscard]] static constexpr auto negate( const tensor_type& t )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept( detail::make_from_tuple< result_tensor_type >(
        collect_ctor_args( ::std::declval<const tensor_type&>(),
        #ifndef LINALG_COMPILER_CLANG
                           [&t]( auto ... indices ) constexpr noexcept { return -( detail::access( t, indices ... ) ); } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                           []( auto ... indices ) constexpr noexcept { return typename tensor_type::value_type(); } ) ) ) )
        #endif
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    {
      // Define negation operation on each element
      auto negate_lambda = [&t]( auto ... indices ) constexpr noexcept { return -( detail::access( t, indices ... ) ); };
      // Construct negated tensor
      return detail::make_from_tuple<result_tensor_type>( collect_ctor_args( t, negate_lambda ) );
    }
};

/// @brief Defines addition operation on a pair of tensors
/// @tparam T1 first in pair of tensors to be added
/// @tparam T2 second in pair of tensors to be added
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_data T1, concepts::tensor_data T2 >
#else
template < class T1, class T2, typename = ::std::enable_if_t< concepts::tensor_data_v<T1> && concepts::tensor_data_v<T2> > >
#endif
class addition
{
  public:
    //- Types

    /// @brief First input tensor type
    using first_tensor_type  = T1;
    /// @brief Second input tensor type
    using second_tensor_type = T2;
  private:
    // Aliases
    using result_value_type  = ::std::decay_t< decltype( ::std::declval<typename first_tensor_type::value_type>() + ::std::declval<typename second_tensor_type::value_type>() ) >;
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class U1, class U2 > requires ( !( concepts::fixed_size_tensor_data<U1> || concepts::fixed_size_tensor_data<U2> || concepts::dynamic_tensor_data<U1> || concepts::dynamic_tensor_data<U2> ) )
    #else
    template < class U1, class U2, typename = void >
    #endif
    struct Result_tensor { using type = default_dynamic_t<U1,result_value_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::fixed_size_tensor_data U1, class U2 >
    struct Result_tensor
    #else
    template < class U1, class U2  >
    struct Result_tensor< U1, U2, ::std::enable_if_t< concepts::fixed_size_tensor_data_v<U1> > >
    #endif
    { using type = typename U1::template rebind_t<result_value_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class U1, concepts::fixed_size_tensor_data U2 > requires ( !concepts::fixed_size_tensor_data<U2> )
    struct Result_tensor
    #else
    template < class U1, class U2 >
    struct Result_tensor< U1, U2, ::std::enable_if_t< concepts::fixed_size_tensor_data_v<U2> && !concepts::fixed_size_tensor_data_v<U1> > >
    #endif
    { using type = typename U2::template rebind_t<result_value_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::dynamic_tensor_data U1, class U2 > requires ( !( concepts::fixed_size_tensor_data<U1> || concepts::fixed_size_tensor_data<U2> ) )
    struct Result_tensor
    #else
    template < class U1, class U2 >
    struct Result_tensor< U1, U2, ::std::enable_if_t< concepts::dynamic_tensor_data_v<U1> && !( concepts::fixed_size_tensor_data_v<U1> || concepts::fixed_size_tensor_data_v<U2> ) > >
    #endif
    { using type = typename U1::template rebind_t<result_value_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class U1, concepts::dynamic_tensor_data U2 > requires ( !( concepts::fixed_size_tensor_data<U1> || concepts::fixed_size_tensor_data<U2> || concepts::dynamic_tensor_data<U1> ) )
    struct Result_tensor
    #else
    template < class U1, class U2  >
    struct Result_tensor< U1, U2, ::std::enable_if_t< concepts::dynamic_tensor_data_v<U2> && !( concepts::fixed_size_tensor_data_v<U1> || concepts::fixed_size_tensor_data_v<U2> || concepts::dynamic_tensor_data_v<U1> ) > >
    #endif
    { using type = typename U2::template rebind_t<result_value_type>; };
    using result_tensor_type = typename Result_tensor< first_tensor_type, second_tensor_type >::type;
    // Gets necessary arguments for constrution
    // If tensor type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_tensor = result_tensor_type,
               typename = ::std::enable_if_t< concepts::fixed_size_tensor_data_v<Result_tensor> > >
    #endif
    [[nodiscard]] static inline constexpr auto collect_ctor_args( [[maybe_unused]] const first_tensor_type&, [[maybe_unused]] const second_tensor_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_tensor_data<result_tensor_type>
    #endif
    { return ::std::tuple( ::std::forward<Lambda>( lambda ) ); }
    // If the tensor type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default cosntructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_tensor = result_tensor_type,
               typename = ::std::enable_if_t< concepts::dynamic_tensor_data_v<Result_tensor> >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr auto collect_ctor_args( const first_tensor_type& t1, const second_tensor_type& t2, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_tensor_data<result_tensor_type>
    #endif
    {
      if constexpr ( ::std::is_default_constructible_v<typename result_tensor_type::allocator_type> &&
                     ::std::allocator_traits<typename result_tensor_type::allocator_type>::is_always_equal::value )
      {
        if constexpr ( ::std::is_same_v<first_tensor_type,result_tensor_type> )
        {
          return ::std::tuple( t1.size(), t1.capacity(), ::std::forward<Lambda>( lambda ) );
        }
        else
        {
          return ::std::tuple( t2.size(), t2.capacity(), ::std::forward<Lambda>( lambda ) );
        }
      }
      else
      {
        if constexpr ( ::std::is_same_v<first_tensor_type,result_tensor_type> )
        {
          using result_alloc_type = typename ::std::allocator_traits<typename first_tensor_type::allocator_type>::template rebind_alloc<result_value_type>;
          return ::std::tuple( t1.size(), t1.capacity(), ::std::forward<Lambda>( lambda ), result_alloc_type( t1.get_allocator() ) );
        }
        else
        {
          using result_alloc_type = typename ::std::allocator_traits<typename second_tensor_type::allocator_type>::template rebind_alloc<result_value_type>;
          return ::std::tuple( t2.size(), t2.capacity(), ::std::forward<Lambda>( lambda ), result_alloc_type( t2.get_allocator() ) );
        }
      }
    }
  public:
    //- Operations

    /// @brief Returns t1 + t2
    [[nodiscard]] static constexpr auto add( const first_tensor_type& t1, const second_tensor_type& t2 )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept( detail::make_from_tuple< result_tensor_type >(
        collect_ctor_args( ::std::declval<const first_tensor_type&>(),
                            ::std::declval<const second_tensor_type&>(),
        #ifndef LINALG_COMPILER_CLANG
                            [&t1,&t2]( auto ... indices ) constexpr noexcept
                              { return detail::access( t1, indices ... ) + detail::access( t2, indices ... ); } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                            []( auto ... indices ) constexpr noexcept
                              { return typename result_tensor_type::value_type(); } ) ) ) )
        #endif
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    {
      auto add_lambda = [&t1,&t2]( auto ... indices ) constexpr noexcept
        { return detail::access( t1, indices ... ) + detail::access( t2, indices ... ); };
      // Construct addition tensor
      return detail::make_from_tuple<result_tensor_type>( collect_ctor_args( t1, t2, add_lambda ) );
    }
    /// @brief Returns t1 += t2
    [[nodiscard]] static constexpr first_tensor_type& add( first_tensor_type& t1, const second_tensor_type& t2 )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept( detail::apply_all( t1.underlying_span(),
      #ifndef LINALG_COMPILER_CLANG
                                             [&t1,&t2]( auto ... indices ) constexpr noexcept
                                               { static_cast<void>( detail::access( t1, indices ... ) += detail::access( t2, indices ... ) ); },
      #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                                             []( auto ... indices ) constexpr noexcept
                                               { },
      #endif
                                             LINALG_EXECUTION_UNSEQ ) ) )
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    {
      auto add_lambda = [&t1,&t2]( auto ... indices ) constexpr noexcept
        { static_cast<void>( detail::access( t1, indices ... ) += detail::access( t2, indices ... ) ); };
      // Apply lamda
      detail::apply_all( t1.underlying_span(), add_lambda, LINALG_EXECUTION_UNSEQ );
      // Return updated tensor
      return t1;
    }
};

/// @brief Defines subtraction operation on a pair of tensors
/// @tparam T1 first in pair of tensors to be added
/// @tparam T2 second in pair of tensors to be added
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_data T1, concepts::tensor_data T2 >
#else
template < class T1, class T2, typename = ::std::enable_if_t< concepts::tensor_data_v<T1> && concepts::tensor_data_v<T2> > >
#endif
class subtraction
{
  public:
    //- Types

    /// @brief First input tensor type
    using first_tensor_type  = T1;
    /// @brief Second input tensor type
    using second_tensor_type = T2;
  private:
    // Aliases
    using result_value_type  = ::std::decay_t< decltype( ::std::declval<typename first_tensor_type::value_type>() + ::std::declval<typename second_tensor_type::value_type>() ) >;
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class U1, class U2 > requires ( !( concepts::fixed_size_tensor_data<U1> || concepts::fixed_size_tensor_data<U2> || concepts::dynamic_tensor_data<U1> || concepts::dynamic_tensor_data<U2> ) )
    #else
    template < class U1, class U2, typename = void >
    #endif
    struct Result_tensor { using type = default_dynamic_t<U1,result_value_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::fixed_size_tensor_data U1, class U2 >
    struct Result_tensor
    #else
    template < class U1, class U2  >
    struct Result_tensor< U1, U2, ::std::enable_if_t< concepts::fixed_size_tensor_data_v<U1> > >
    #endif
    { using type = typename U1::template rebind_t<result_value_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class U1, concepts::fixed_size_tensor_data U2 > requires ( !concepts::fixed_size_tensor_data<U2> )
    struct Result_tensor
    #else
    template < class U1, class U2 >
    struct Result_tensor< U1, U2, ::std::enable_if_t< concepts::fixed_size_tensor_data_v<U2> && !concepts::fixed_size_tensor_data_v<U1> > >
    #endif
    { using type = typename U2::template rebind_t<result_value_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::dynamic_tensor_data U1, class U2 > requires ( !( concepts::fixed_size_tensor_data<U1> || concepts::fixed_size_tensor_data<U2> ) )
    struct Result_tensor
    #else
    template < class U1, class U2 >
    struct Result_tensor< U1, U2, ::std::enable_if_t< concepts::dynamic_tensor_data_v<U1> && !( concepts::fixed_size_tensor_data_v<U1> || concepts::fixed_size_tensor_data_v<U2> ) > >
    #endif
    { using type = typename U1::template rebind_t<result_value_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class U1, concepts::dynamic_tensor_data U2 > requires ( !( concepts::fixed_size_tensor_data<U1> || concepts::fixed_size_tensor_data<U2> || concepts::dynamic_tensor_data<U1> ) )
    struct Result_tensor
    #else
    template < class U1, class U2  >
    struct Result_tensor< U1, U2, ::std::enable_if_t< concepts::dynamic_tensor_data_v<U2> && !( concepts::fixed_size_tensor_data_v<U1> || concepts::fixed_size_tensor_data_v<U2> || concepts::dynamic_tensor_data_v<U1> ) > >
    #endif
    { using type = typename U2::template rebind_t<result_value_type>; };
    using result_tensor_type = typename Result_tensor< first_tensor_type, second_tensor_type >::type;
    // Gets necessary arguments for constrution
    // If tensor type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_tensor = result_tensor_type,
               typename = ::std::enable_if_t< concepts::fixed_size_tensor_data_v<Result_tensor> > >
    #endif
    [[nodiscard]] static inline constexpr auto collect_ctor_args( [[maybe_unused]] const first_tensor_type&, [[maybe_unused]] const second_tensor_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_tensor_data<result_tensor_type>
    #endif
    { return tuple( forward<Lambda>( lambda ) ); }
    // If the tensor type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default cosntructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_tensor = result_tensor_type,
               typename = ::std::enable_if_t< concepts::dynamic_tensor_data_v<Result_tensor> >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr auto collect_ctor_args( const first_tensor_type& t1, const second_tensor_type& t2, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_tensor_data<result_tensor_type>
    #endif
    {
      if constexpr ( ::std::is_default_constructible_v<typename result_tensor_type::allocator_type> &&
                     ::std::allocator_traits<typename result_tensor_type::allocator_type>::is_always_equal::value )
      {
        if constexpr ( ::std::is_same_v<first_tensor_type,result_tensor_type> )
        {
          return ::std::tuple( t1.size(), t1.capacity(), ::std::forward<Lambda>( lambda ) );
        }
        else
        {
          return ::std::tuple( t2.size(), t2.capacity(), ::std::forward<Lambda>( lambda ) );
        }
      }
      else
      {
        if constexpr ( ::std::is_same_v<first_tensor_type,result_tensor_type> )
        {
          using result_alloc_type = typename ::std::allocator_traits<typename first_tensor_type::allocator_type>::template rebind_alloc<result_value_type>;
          return ::std::tuple( t1.size(), t1.capacity(), ::std::forward<Lambda>( lambda ), result_alloc_type( t1.get_allocator() ) );
        }
        else
        {
          using result_alloc_type = typename ::std::allocator_traits<typename second_tensor_type::allocator_type>::template rebind_alloc<result_value_type>;
          return ::std::tuple( t2.size(), t2.capacity(), ::std::forward<Lambda>( lambda ), result_alloc_type( t2.get_allocator() ) );
        }
      }
    }
  public:
    //- Operations

    /// @brief Returns t1 - t2
    [[nodiscard]] static constexpr auto subtract( const first_tensor_type& t1, const second_tensor_type& t2 )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept( detail::make_from_tuple< result_tensor_type >(
        collect_ctor_args( ::std::declval<const first_tensor_type&>(),
                           ::std::declval<const second_tensor_type&>(),
        #ifndef LINALG_COMPILER_CLANG
                           [&t1,&t2]( auto ... indices ) constexpr noexcept
                             { return detail::access( t1, indices ... ) - detail::access( t2, indices ... ); } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                             []( auto ... indices ) constexpr noexcept
                               { return typename result_tensor_type::value_type(); } ) ) ) )
        #endif
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    {
      auto subtract_lambda = [&t1,&t2]( auto ... indices ) constexpr noexcept
        { return detail::access( t1, indices ... ) - detail::access( t2, indices ... ); };
      // Construct addition tensor
      return detail::make_from_tuple<result_tensor_type>( collect_ctor_args( t1, t2, subtract_lambda ) );
    }
    /// @brief Returns t1 -= t2
    [[nodiscard]] static constexpr first_tensor_type& subtract( first_tensor_type& t1, const second_tensor_type& t2 )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept( detail::apply_all( t1.underlying_span(),
        #ifndef LINALG_COMPILER_CLANG
                                             [&t1,&t2]( auto ... indices ) constexpr noexcept
                                               { static_cast<void>( detail::access( t1, indices ... ) -= detail::access( t2, indices ... ) ); },
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                                             []( auto ... indices ) constexpr noexcept
                                               { },
        #endif
                                             LINALG_EXECUTION_UNSEQ ) ) )
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    {
      auto subtract_lambda = [&t1,&t2]( auto ... indices ) constexpr noexcept
        { static_cast<void>( detail::access( t1, indices ... ) -= detail::access( t2, indices ... ) ); };
      // Apply lamda
      detail::apply_all( t1.underlying_span(), subtract_lambda, LINALG_EXECUTION_UNSEQ );
      // Return updated tensor
      return t1;
    }
};

/// @brief Defines scalar product operation on a tensor with a scalar
/// @tparam S scalar
/// @tparam T tensor
#ifdef LINALG_ENABLE_CONCEPTS
template < class S, concepts::tensor_data T >
#else
template < class S, class T, typename = ::std::enable_if_t< concepts::tensor_data_v<T> > >
#endif
class scalar_product
{
  public:
    //- Types

    /// @brief Input tensor type
    using tensor_type        = T;
  private:
    // Aliases
    using result_value_type  = decay_t< decltype( ::std::declval<typename tensor_type::value_type>() * ::std::declval<S>() ) >;
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class U > requires ( !( concepts::fixed_size_tensor_data<U> || concepts::dynamic_tensor_data<U> ) )
    #else
    template < class U, typename = void >
    #endif
    struct Result_tensor { using type = default_dynamic_t<U,result_value_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::fixed_size_tensor_data U >
    struct Result_tensor
    #else
    template < class U  >
    struct Result_tensor< U, ::std::enable_if_t< concepts::fixed_size_tensor_data_v<U> > >
    #endif
    { using type = typename U::template rebind_t<result_value_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::dynamic_tensor_data U1, class U2 > requires ( !( concepts::fixed_size_tensor<U1> || concepts::fixed_size_tensor<U2> ) )
    struct Result_tensor
    #else
    template < class U >
    struct Result_tensor< U, ::std::enable_if_t< concepts::dynamic_tensor_data_v<U> > >
    #endif
    { using type = typename U::template rebind_t<result_value_type>; };
    using result_tensor_type = typename Result_tensor< tensor_type >::type;
    // Gets necessary arguments for constrution
    // If tensor type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_tensor = result_tensor_type,
               typename = ::std::enable_if_t< concepts::fixed_size_tensor_data_v<Result_tensor> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const tensor_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_tensor_data<result_tensor_type>
    #endif
    { return ::std::tuple( ::std::forward<Lambda>( lambda ) ); }
    // If the tensor type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default cosntructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_tensor = result_tensor_type,
               typename = ::std::enable_if_t< concepts::dynamic_tensor_data_v<Result_tensor> >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const tensor_type& t, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_tensor_data<result_tensor_type>
    #endif
    {
      if constexpr ( ::std::is_default_constructible_v<typename result_tensor_type::allocator_type> &&
                     ::std::allocator_traits<typename result_tensor_type::allocator_type>::is_always_equal::value )
      {
        return ::std::tuple( t.size(), t.capacity(), ::std::forward<Lambda>( lambda ) );
      }
      else
      {
        using result_alloc_type = typename ::std::allocator_traits<typename tensor_type::allocator_type>::template rebind_alloc<result_value_type>;
        return ::std::tuple( t.size(), t.capacity(), ::std::forward<Lambda>( lambda ), result_alloc_type( t.get_allocator() ) );
      }
    }
  public:
    //- Operations

    /// @brief Returns s * t
    [[nodiscard]] static constexpr auto prod( const S& s, const tensor_type& t )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept( detail::make_from_tuple< result_tensor_type >(
        collect_ctor_args( t,
        #ifndef LINALG_COMPILER_CLANG
                           [&s,&t]( auto ... indices ) constexpr noexcept
                             { return s * detail::access( t, indices ... ); } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                           []( auto ... indices ) constexpr noexcept
                             { return typename result_tensor_type::value_type(); } ) ) ) )
        #endif
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    {
      // Define product operation on each element
      auto prod_lambda = [&s,&t]( auto ... indices ) constexpr noexcept
        { return s * detail::access( t, indices ... ); };
      // Construct product tensor
      return detail::make_from_tuple<result_tensor_type>( collect_ctor_args( t, prod_lambda ) );
    }
    /// @brief Returns t * s
    [[nodiscard]] static constexpr auto prod( const tensor_type& t, const S& s )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept( detail::make_from_tuple< result_tensor_type >(
        collect_ctor_args( t,
        #ifndef LINALG_COMPILER_CLANG
                           [&s,&t]( auto ... indices ) constexpr noexcept
                             { return detail::access( t, indices ... ) * s; } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                           []( auto ... indices ) constexpr noexcept
                             { return typename result_tensor_type::value_type(); } ) ) ) )
        #endif
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    {
      // Define product operation on each element
      auto prod_lambda = [&s,&t]( auto ... indices ) constexpr noexcept
        { return detail::access( t, indices ... ) * s; };
      // Construct product tensor
      return detail::make_from_tuple<result_tensor_type>( collect_ctor_args( t, prod_lambda ) );
    }
    /// @brief Returns t *= s
    [[nodiscard]] static constexpr tensor_type& prod( tensor_type& t, const S& s )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept( detail::apply_all( t.underlying_span(),
      #ifndef LINALG_COMPILER_CLANG
                                             [&t,&s]( auto ... indices ) constexpr noexcept
                                               { static_cast<void>( detail::access( t, indices ... ) += s ); },
      #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                                             []( auto ... indices ) constexpr noexcept
                                               { },
      #endif
                                             LINALG_EXECUTION_UNSEQ ) ) )
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    {
      // Define product operation on each element
      auto prod_lambda = [&s,&t]( auto ... indices ) constexpr noexcept
        { static_cast<void>( detail::access( t, indices ... ) *= s ); };
      // Apply lamda
      detail::apply_all( t.underlying_span(), prod_lambda, LINALG_EXECUTION_UNSEQ );
      // Return updated tensor
      return t;
    }
};

/// @brief Defines scalar division operation on a tensor with a scalar
/// @tparam S scalar
/// @tparam T tensor
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_data T, class S >
#else
template < class T, class S, typename = ::std::enable_if_t< concepts::tensor_data_v<T> > >
#endif
struct scalar_division
{
  public:
    //- Types

    /// @brief Input tensor type
    using tensor_type        = T;
  private:
    // Aliases
    using result_value_type  = ::std::decay_t< decltype( ::std::declval<typename tensor_type::value_type>() / ::std::declval<S>() ) >;
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class U > requires ( !( concepts::fixed_size_tensor_data<U> || concepts::dynamic_tensor_data<U> ) )
    #else
    template < class U, typename = void >
    #endif
    struct Result_tensor { using type = default_dynamic_t<U,result_value_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::fixed_size_tensor_data U >
    struct Result_tensor
    #else
    template < class U  >
    struct Result_tensor< U, ::std::enable_if_t< concepts::fixed_size_tensor_data_v<U> > >
    #endif
    { using type = typename U::template rebind_t<result_value_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::dynamic_tensor_data U1, class U2 > requires ( !( concepts::fixed_size_tensor<U1> || concepts::fixed_size_tensor<U2> ) )
    struct Result_tensor
    #else
    template < class U >
    struct Result_tensor< U, ::std::enable_if_t< concepts::dynamic_tensor_data_v<U> > >
    #endif
    { using type = typename U::template rebind_t<result_value_type>; };
    using result_tensor_type = typename Result_tensor< tensor_type >::type;
    // Gets necessary arguments for constrution
    // If tensor type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_tensor = result_tensor_type,
               typename = ::std::enable_if_t< concepts::fixed_size_tensor_data_v<Result_tensor> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const tensor_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_tensor_data<result_tensor_type>
    #endif
    { return tuple( forward<Lambda>( lambda ) ); }
    // If the tensor type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default cosntructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_tensor = result_tensor_type,
               typename = ::std::enable_if_t< concepts::dynamic_tensor_data_v<Result_tensor> >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const tensor_type& t, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_tensor_data<result_tensor_type>
    #endif
    {
      if constexpr ( ::std::is_default_constructible_v<typename result_tensor_type::allocator_type> &&
                     ::std::allocator_traits<typename result_tensor_type::allocator_type>::is_always_equal::value )
      {
        return ::std::tuple( t.size(), t.capacity(), ::std::forward<Lambda>( lambda ) );
      }
      else
      {
        using result_alloc_type = typename ::std::allocator_traits<typename tensor_type::allocator_type>::template rebind_alloc<result_value_type>;
        return ::std::tuple( t.size(), t.capacity(), ::std::forward<Lambda>( lambda ), result_alloc_type( t.get_allocator() ) );
      }
    }
  public:
    //- Operations

    /// @brief Returns t / s
    [[nodiscard]] static constexpr auto divide( const tensor_type& t, const S& s )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept( detail::make_from_tuple< result_tensor_type >(
        collect_ctor_args( declval<const tensor_type&>(),
        #ifndef LINALG_COMPILER_CLANG
                            [&t,&s]( auto ... indices ) constexpr noexcept
                              { return detail::access( t, indices ... ) / s; } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                            []( auto ... indices ) constexpr noexcept
                              { return typename result_tensor_type::value_type(); } ) ) ) )
        #endif
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    {
      // Define division operation on each element
      auto divide_lambda = [&t,&s]( auto ... indices ) constexpr noexcept
        { return detail::access( t, indices ... ) / s; };
      // Construct divided tensor
      return detail::make_from_tuple<result_tensor_type>( collect_ctor_args( t, divide_lambda ) );
    }
    /// @brief Returns t /= s
    [[nodiscard]] static constexpr tensor_type& divide( tensor_type& t, const S& s )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept( detail::apply_all( t.underlying_span(),
      #ifndef LINALG_COMPILER_CLANG
                                             [&t,&s]( auto ... indices ) constexpr noexcept
                                               { static_cast<void>( t[ indices ... ] /= s ); },
      #else
                                             []( auto ... indices ) constexpr noexcept
                                               { },
      #endif
                                             LINALG_EXECUTION_UNSEQ ) ) )
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    {
      // Define product operation on each element
      auto divide_lambda = [&s,&t]( auto ... indices ) constexpr noexcept
        { static_cast<void>( detail::access( t, indices ... ) /= s ); };
      // Apply lamda
      detail::apply_all( t.underlying_span(), divide_lambda, LINALG_EXECUTION_UNSEQ );
      // Return updated tensor
      return t;
    }
};

/// @brief Defines transpose operation on a matrix
/// @tparam M matrix
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::matrix_data M >
#else
template < class M, typename = ::std::enable_if_t< concepts::matrix_data_v<M> > >
#endif
class transpose_matrix
{
  public:
    //- Types

    /// @brief Input matrix type
    using matrix_type        = M;
  private:
    // Aliases
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class U > requires ( !( concepts::fixed_size_matrix_data<U> || concepts::dynamic_matrix_data<U> ) )
    #else
    template < class U, typename = void >
    #endif
    struct Result_matrix
    { using type = dr_matrix< typename U::value_type,
                              ::std::allocator< typename U::value_type >,
                              default_layout,
                              typename detail::rebind_accessor_t<typename U::accessor_type,typename U::value_type> >; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::fixed_size_matrix_data U >
    struct Result_matrix
    #else
    template < class U  >
    struct Result_matrix< U, ::std::enable_if_t< concepts::fixed_size_matrix_data_v<U> > >
    #endif
    { using type = fs_matrix< typename U::value_type,
                              U().columns(),
                              U().rows(),
                              typename U::layout_type,
                              typename U::accessor_type >; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::dynamic_matrix_data U >
    struct Result_matrix
    #else
    template < class U >
    struct Result_matrix< U, ::std::enable_if_t< concepts::dynamic_matrix_data_v<U> > >
    #endif
    { using type = U; };
    using result_matrix_type = typename Result_matrix< matrix_type >::type;
    // Returns the allocator of matrix_type
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename Mat = matrix_type,
               typename = ::std::enable_if_t< concepts::dynamic_matrix_data_v< Mat > > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( const matrix_type& m ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( concepts::dynamic_matrix_data< matrix_type > )
    #endif
    {
      return m.get_allocator();
    }
    // Returns the std allocator
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename Mat = matrix_type,
               typename = ::std::enable_if_t< !concepts::dynamic_matrix_data_v< Mat > >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( [[maybe_unused]] const matrix_type& m ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( !concepts::dynamic_matrix_data< matrix_type > )
    #endif
    {
      return ::std::allocator< ::std::remove_cv_t<typename matrix_type::element_type> >();
    }
    // Gets necessary arguments for constrution
    // If matrix type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_matrix = result_matrix_type,
               typename = ::std::enable_if_t< concepts::fixed_size_matrix_data_v<Result_matrix> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const matrix_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_matrix_data<result_matrix_type>
    #endif
    { return ::std::tuple( ::std::forward<Lambda>( lambda ) ); }
    // If the matrix type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default constructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_matrix = result_matrix_type,
               typename = ::std::enable_if_t< concepts::dynamic_matrix_data_v<Result_matrix> >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const matrix_type& m, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_matrix_data<result_matrix_type>
    #endif
    {
      using result_extents_type = typename result_matrix_type::extents_type;
      if constexpr ( ::std::is_default_constructible_v<typename result_matrix_type::allocator_type> &&
                     ::std::allocator_traits<typename result_matrix_type::allocator_type>::is_always_equal::value )
      {
        return ::std::tuple( result_extents_type( m.size().extent(1), m.size().extent(0) ),
                             result_extents_type( m.capacity().extent(0), m.capacity().extent(0) ),
                             ::std::forward<Lambda>( lambda ) );
      }
      else
      {
        using result_alloc_type = typename result_matrix_type::allocator_type;
        return ::std::tuple( result_extents_type( m.size().extent(1), m.size().extent(0) ),
                             result_extents_type( m.capacity().extent(1), m.capacity().extent(0) ),
                             ::std::forward<Lambda>( lambda ),
                             result_alloc_type( get_allocator( m ) ) );
      }
    }
  public:
    //- Operations

    /// @brief Returns transpose( matrix )
    [[nodiscard]] static constexpr auto trans( const matrix_type& m )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept( detail::make_from_tuple< result_matrix_type >(
        collect_ctor_args( m,
        #ifndef LINALG_COMPILER_CLANG
          [&m]( auto index1, auto index2 ) constexpr noexcept
            { return detail::access( m, index2, index1 ); } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
          []( auto index1, auto index2 ) constexpr noexcept
            { return typename matrix_type::value_type(); } ) ) ) )
        #endif
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    {
      // Define negation operation on each element
      auto transpose_lambda = [&m]( auto index1, auto index2 ) constexpr noexcept
        { return detail::access( m, index2, index1 ); };
      // Construct transpose matrix
      return detail::make_from_tuple<result_matrix_type>( collect_ctor_args( m, transpose_lambda ) );
    }
};

/// @brief Defines transpose operation on a vector
/// @tparam V vector
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::vector_data V >
#else
template < class V, typename = ::std::enable_if_t< concepts::vector_data_v<V> > >
#endif
class transpose_vector
{
  public:
    //- Types

    /// @brief Input vector type
    using vector_type        = V;
  private:
    // Aliases
    using result_vector_type = vector_type;
  public:
    //- Operations

    /// @brief Returns transpose( matrix )
    [[nodiscard]] static inline constexpr auto trans( const vector_type& v ) noexcept
    {
      return v;
    }
};

/// @brief Defines conjugate transpose operation on a matrix
/// @tparam M matrix
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::matrix_data M >
#else
template < class M, typename = ::std::enable_if_t< concepts::matrix_data_v<M> > >
#endif
class conjugate_matrix
{
  public:
    //- Types

    /// @brief Input matrix type
    using matrix_type         = M;
  private:
    // Aliases
    using result_element_type = ::std::conditional_t< detail::is_complex_v< ::std::decay_t< typename matrix_type::element_type> >,
                                                      decltype( ::std::conj( ::std::declval<typename matrix_type::element_type>() ) ),
                                                      ::std::remove_cv_t< typename matrix_type::element_type > >;
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class U > requires ( !( concepts::fixed_size_matrix_data<U> || concepts::dynamic_matrix_data<U> ) )
    #else
    template < class U, typename = void >
    #endif
    struct Result_matrix
    { using type = dr_matrix< result_element_type,
                              ::std::allocator< result_element_type >,
                              default_layout,
                              typename detail::rebind_accessor_t<typename U::accessor_type,result_element_type> >; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::fixed_size_matrix_data U >
    struct Result_matrix
    #else
    template < class U  >
    struct Result_matrix< U, ::std::enable_if_t< concepts::fixed_size_matrix_data_v<U> > >
    #endif
    { using type = fs_matrix< result_element_type,
                              U().rows(),
                              U().columns(),
                              default_layout,
                              typename detail::rebind_accessor_t<typename U::accessor_type,result_element_type> >; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::dynamic_matrix_data U >
    struct Result_matrix
    #else
    template < class U >
    struct Result_matrix< U, ::std::enable_if_t< concepts::dynamic_matrix_data_v<U> > >
    #endif
    { using type = dr_matrix< result_element_type,
                              ::std::allocator< result_element_type >,
                              typename U::layout_type,
                              typename detail::rebind_accessor_t<typename U::accessor_type,result_element_type> >; };
    using result_matrix_type = typename Result_matrix< matrix_type >::type;
    // Returns the allocator of matrix_type
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename Mat = matrix_type,
               typename = ::std::enable_if_t< concepts::dynamic_matrix_data_v< Mat > > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( const matrix_type& m ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( concepts::dynamic_matrix_data< matrix_type > )
    #endif
    {
      return m.get_allocator();
    }
    // Returns the std allocator
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename Mat = matrix_type,
               typename = ::std::enable_if_t< !concepts::dynamic_matrix_data_v< Mat > >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( [[maybe_unused]] const matrix_type& m ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( !concepts::dynamic_matrix_data< matrix_type > )
    #endif
    {
      return ::std::allocator<result_element_type>();
    }
    // Gets necessary arguments for construction
    // If matrix type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_matrix = result_matrix_type,
               typename = ::std::enable_if_t< concepts::fixed_size_matrix_data_v<Result_matrix> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const matrix_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_matrix_data<result_matrix_type>
    #endif
    { return ::std::tuple( ::std::forward<Lambda>( lambda ) ); }
    // If the matrix type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default constructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_matrix = result_matrix_type,
               typename = ::std::enable_if_t< concepts::dynamic_matrix_data_v<Result_matrix> >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const matrix_type& m, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_matrix_data<result_matrix_type>
    #endif
    {
      using result_extents_type = typename result_matrix_type::extents_type;
      if constexpr ( ::std::is_default_constructible_v<typename result_matrix_type::allocator_type> &&
                     ::std::allocator_traits<typename result_matrix_type::allocator_type>::is_always_equal::value )
      {
        return ::std::tuple( result_extents_type( m.size().extent(1), m.size().extent(0) ),
                             result_extents_type( m.capacity().extent(1), m.capacity().extent(0) ),
                             ::std::forward<Lambda>( lambda ) );
      }
      else
      {
        using result_alloc_type = typename result_matrix_type::allocator_type;
        return ::std::tuple( result_extents_type( m.size().extent(1), m.size().extent(0) ),
                             result_extents_type( m.capacity().extent(1), m.capacity().extent(0) ),
                             ::std::forward<Lambda>( lambda ),
                             result_alloc_type( get_allocator( m ) ) );
      }
    }
  public:
    //- Operations

    /// @brief Returns transpose conjugate( matrix )
    [[nodiscard]] static constexpr auto conjugate( const matrix_type& m )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept( detail::make_from_tuple< result_matrix_type >(
        collect_ctor_args( m,
        #ifndef LINALG_COMPILER_CLANG
          [&m]( auto index1, auto index2 ) constexpr noexcept
            { return ::std::conj( detail::access( m, index2, index1 ) ); } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
          []( auto index1, auto index2 ) constexpr noexcept
            { return ::std::conj( typename matrix_type::value_type() ); } ) ) ) )
        #endif
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    {
      // Define negation operation on each element
      auto conjugate_lambda = [&m]( auto index1, auto index2 ) constexpr noexcept
        { return ::std::conj( detail::access( m, index2, index1 ) ); };
      // Construct conjugate transpose matrix
      return detail::make_from_tuple<result_matrix_type>( collect_ctor_args( m, conjugate_lambda ) );
    }
};

/// @brief Defines conjugate transpose operation on a vector
/// @tparam V vector
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::vector_data V >
#else
template < class V, typename = ::std::enable_if_t< concepts::vector_data_v<V> > >
#endif
class conjugate_vector
{
  public:
    //- Types

    /// @brief Input vector type
    using vector_type         = V;
  private:
    // Aliases
    using result_element_type = ::std::conditional_t< detail::is_complex_v< ::std::decay_t< typename vector_type::element_type> >,
                                                      decltype( ::std::conj( ::std::declval<typename vector_type::element_type>() ) ),
                                                      ::std::remove_cv_t< typename vector_type::element_type > >;
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class U > requires ( !( concepts::fixed_size_vector_data<U> || concepts::dynamic_vector_data<U> ) )
    #else
    template < class U, typename = void >
    #endif
    struct Result_vector
    { using type = dr_vector< result_element_type,
                              ::std::allocator< result_element_type >,
                              default_layout,
                              typename detail::rebind_accessor_t<typename U::accessor_type,result_element_type> >; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::fixed_size_tensor_data U >
    struct Result_vector
    #else
    template < class U  >
    struct Result_vector< U, ::std::enable_if_t< concepts::fixed_size_tensor_data_v<U> > >
    #endif
    { using type = typename U::template rebind_t<result_element_type>; };
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::dynamic_tensor_data U >
    struct Result_vector
    #else
    template < class U >
    struct Result_vector< U, ::std::enable_if_t< concepts::dynamic_tensor_data_v<U> > >
    #endif
    { using type = typename U::template rebind_t<result_element_type>; };
    using result_vector_type = typename Result_vector< vector_type >::type;
    // Returns the allocator of vector_type
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename Vec = vector_type,
               typename = ::std::enable_if_t< concepts::dynamic_vector_data_v< Vec > > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( const vector_type& m ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( concepts::dynamic_vector_data< vector_type > )
    #endif
    {
      return m.get_allocator();
    }
    // Returns the std allocator
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename Vec = vector_type,
               typename = ::std::enable_if_t< !concepts::dynamic_vector_data_v< Vec > >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( [[maybe_unused]] const vector_type& m ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( !concepts::dynamic_vector_data< vector_type > )
    #endif
    {
      return ::std::allocator<result_element_type>();
    }
    // Gets necessary arguments for constrution
    // If vector type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_vector = result_vector_type,
               typename = ::std::enable_if_t< concepts::fixed_size_vector_data_v<Result_vector> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const vector_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_vector_data<result_vector_type>
    #endif
    { return ::std::tuple( ::std::forward<Lambda>( lambda ) ); }
    // If the vector type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default constructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_vector = result_vector_type,
               typename = ::std::enable_if_t< concepts::dynamic_vector_data_v<Result_vector> >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const vector_type& v, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_vector_data<vector_type>
    #endif
    {
      if constexpr ( ::std::is_default_constructible_v<typename result_vector_type::allocator_type> &&
                     ::std::allocator_traits<typename result_vector_type::allocator_type>::is_always_equal::value )
      {
        return ::std::tuple( v.size(), v.capacity(), ::std::forward<Lambda>( lambda ) );
      }
      else
      {
        using result_alloc_type = typename ::std::allocator_traits<typename vector_type::allocator_type>::template rebind_alloc<result_element_type>;
        return ::std::tuple( v.size(), v.capacity(), ::std::forward<Lambda>( lambda ), result_alloc_type( get_allocator( v ) ) );
      }
    }
  public:
    //- Operations

    /// @brief Returns conjugate( vector )
    [[nodiscard]] static constexpr auto conjugate( const vector_type& v )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( !detail::is_complex_v<typename vector_type::value_type> ||
                noexcept( detail::make_from_tuple< result_vector_type >(
                  collect_ctor_args( v,
                #ifndef LINALG_COMPILER_CLANG
                                     [&v]( auto index ) constexpr noexcept { return ::std::conj( detail::access( v, index ) ); } ) ) ) )
                #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                                     []( auto index ) constexpr noexcept { return ::std::conj( typename vector_type::value_type() ); } ) ) ) )
                #endif
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    {
      if constexpr ( detail::is_complex_v<typename vector_type::value_type> )
      {
        // Define conjugate transpose operation on each element
        auto conj_lambda = [&v]( auto index ) constexpr noexcept { return ::std::conj( detail::access( v, index ) ); };
        // Construct negated vector
        return detail::make_from_tuple<result_vector_type>( collect_ctor_args( v, conj_lambda ) );
      }
      else
      {
        return v;
      }
    }
};

/// @brief Defines multiplication operation on a matrix and a vector
/// @tparam V vector
/// @tparam M matrix
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::vector_data V, concepts::matrix_data M >
#else
template < class V, class M, typename = ::std::enable_if_t< concepts::vector_data_v<V> && concepts::matrix_data_v<M> > >
#endif
class vector_matrix_product
{
  public:
    //- Types

    /// @brief Input vector type
    using vector_type = V;
    /// @brief Input matrix type
    using matrix_type = M;
  private:
    // Return the allocator
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename Mat = matrix_type,
               typename = ::std::enable_if_t< concepts::dynamic_matrix_data_v< Mat > > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( const matrix_type& m ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_matrix_data< matrix_type >
    #endif
    {
      return m.get_allocator();
    }
    // Returns the std allocator
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename Mat = matrix_type,
               typename = ::std::enable_if_t< !concepts::dynamic_matrix_data_v< Mat > >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( [[maybe_unused]] const matrix_type& m ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( !concepts::dynamic_matrix_data< matrix_type > )
    #endif
    {
      return ::std::allocator<result_value_type>();
    }
    // Aliases
    using result_value_type       = ::std::decay_t< decltype( declval<typename vector_type::value_type>() * declval<typename matrix_type::value_type>() ) >;
    using pre_result_vector_type  = ::std::conditional_t< 
                                                          #ifdef LINALG_ENABLE_CONCEPTS
                                                          concepts::fixed_size_matrix_data<matrix_type>,
                                                          #else
                                                          concepts::fixed_size_matrix_data_v<matrix_type>,
                                                          #endif
                                                          fs_vector< result_value_type,
                                                                     matrix_type::extents_type::static_extent(1),
                                                                     detail::rebind_layout_t<typename vector_type::layout_type,
                                                                                             ::std::experimental::extents<typename vector_type::size_type,
                                                                                                                          matrix_type::extents_type::static_extent(1)> >,
                                                                     detail::rebind_accessor_t<typename vector_type::accessor_type,result_value_type> >,
                                                          dr_vector< result_value_type,
                                                                     typename ::std::allocator_traits< ::std::decay_t< decltype( get_allocator( declval<matrix_type>() ) ) > >::template rebind_alloc<result_value_type>,
                                                                     detail::rebind_layout_t<typename vector_type::layout_type,
                                                                                             ::std::experimental::extents<typename matrix_type::size_type,
                                                                                                                          matrix_type::extents_type::static_extent(1)> >,
                                                                     detail::rebind_accessor_t<typename vector_type::accessor_type,result_value_type> > >;
    using post_result_vector_type = ::std::conditional_t< 
                                                          #ifdef LINALG_ENABLE_CONCEPTS
                                                          concepts::fixed_size_matrix_data<matrix_type>,
                                                          #else
                                                          concepts::fixed_size_matrix_data_v<matrix_type>,
                                                          #endif
                                                          fs_vector< result_value_type,
                                                                     matrix_type::extents_type::static_extent(0),
                                                                     detail::rebind_layout_t<typename vector_type::layout_type,
                                                                                             ::std::experimental::extents<typename matrix_type::size_type,
                                                                                                                          matrix_type::extents_type::static_extent(0) > >,
                                                                     detail::rebind_accessor_t<typename vector_type::accessor_type,result_value_type> >,
                                                          dr_vector< result_value_type,
                                                                     typename ::std::allocator_traits< decay_t< decltype( get_allocator( declval<matrix_type>() ) ) > >::template rebind_alloc<result_value_type>,
                                                                     detail::rebind_layout_t<typename vector_type::layout_type,
                                                                                             ::std::experimental::extents<typename matrix_type::size_type,
                                                                                                                          matrix_type::extents_type::static_extent(0)> >,
                                                                      detail::rebind_accessor_t<typename vector_type::accessor_type,result_value_type> > >;
    // Gets necessary arguments for construction
    // If vector type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_vector = pre_result_vector_type,
               typename = ::std::enable_if_t< concepts::fixed_size_vector_data_v<Result_vector> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const vector_type&, [[maybe_unused]] const matrix_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_vector_data<pre_result_vector_type>
    #endif
    { return ::std::tuple( ::std::forward<Lambda>( lambda ) ); }
    // If the vector type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default cosntructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_vector = pre_result_vector_type,
               typename = ::std::enable_if_t< concepts::dynamic_vector_data_v<Result_vector> >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const vector_type&, const matrix_type& m, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_vector_data<pre_result_vector_type>
    #endif
    {
      using result_vector_type = pre_result_vector_type;
      if constexpr ( ::std::is_default_constructible_v<typename result_vector_type::allocator_type> &&
                     ::std::allocator_traits<typename result_vector_type::allocator_type>::is_always_equal::value )
      {
        using result_extents_type = typename result_vector_type::extents_type;
        return ::std::tuple( result_extents_type( m.size().extent(1) ),
                             result_extents_type( m.capacity().extent(1) ),
                             ::std::forward<Lambda>( lambda ) );
      }
      else
      {
        using result_extents_type = typename result_vector_type::extents_type;
        using result_alloc_type   = typename result_vector_type::allocator_type;
        return ::std::tuple( result_extents_type( m.size().extent(1) ),
                             result_extents_type( m.capacity().extent(1) ),
                             ::std::forward<Lambda>( lambda ),
                             result_alloc_type( m.get_allocator() ) );
      }
    }
    // Gets necessary arguments for constrution
    // If vector type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_vector = post_result_vector_type,
               typename = ::std::enable_if_t< concepts::fixed_size_vector_data_v<Result_vector> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const matrix_type&, [[maybe_unused]] const vector_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_vector_data<post_result_vector_type>
    #endif
    { return ::std::tuple( ::std::forward<Lambda>( lambda ) ); }
    // If the vector type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default cosntructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_vector = post_result_vector_type,
               typename = ::std::enable_if_t< concepts::dynamic_vector_data_v<Result_vector> >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const matrix_type& m, [[maybe_unused]] const vector_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_vector_data<post_result_vector_type>
    #endif
    {
      using result_vector_type = post_result_vector_type;
      if constexpr ( ::std::is_default_constructible_v<typename result_vector_type::allocator_type> &&
                     ::std::allocator_traits<typename result_vector_type::allocator_type>::is_always_equal::value )
      {
        using result_extents_type = typename result_vector_type::extents_type;
        return ::std::tuple( result_extents_type( m.size().extent(0) ),
                             result_extents_type( m.capacity().extent(0) ),
                             ::std::forward<Lambda>( lambda ) );
      }
      else
      {
        using result_extents_type = typename result_vector_type::extents_type;
        using result_alloc_type   = typename result_vector_type::allocator_type;
        return ::std::tuple( result_extents_type( m.size().extent(0) ),
                             result_extents_type( m.capacity().extent(0) ),
                             ::std::forward<Lambda>( lambda ),
                             result_alloc_type( m.get_allocator() ) );
      }
    }
  public:
    //- Operations

    /// @brief Returns v * m
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename Vec = vector_type,
               typename Mat = matrix_type,
               typename = ::std::enable_if_t< ( ( Vec::extents_type::static_extent(0) == Mat::extents_type::static_extent(0) ) ||
                                                ( Vec::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) ||
                                                ( Mat::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) ) > >
    #endif
    [[nodiscard]] static constexpr pre_result_vector_type prod( const vector_type& v, const matrix_type& m )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept(
        detail::make_from_tuple< pre_result_vector_type >(
          collect_ctor_args( v,
                             m,
                             #ifndef LINALG_COMPILER_CLANG
                             [&v,&m]( auto index ) constexpr noexcept
                             {
                               result_value_type result = 0;
                               detail::for_each( LINALG_EXECUTION_UNSEQ,
                                                 detail::faux_index_iterator<typename vector_type::index_type>( 0 ),
                                                 detail::faux_index_iterator<typename vector_type::index_type>( v.size().extent(0) ),
                                                 [ &v, &m, &index, &result ] ( typename vector_type::index_type index2 ) constexpr noexcept
                                                   { result += detail::access( v, index2 ) * detail::access( m, index2, index ); } );
                               return result; }
                             #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                             []( auto index ) constexpr noexcept
                             { return result_value_type(); }
                             #endif
                             ) ) ) &&
                ( ( vector_type::extents_type::static_extent(0) != ::std::experimental::dynamic_extent ) &&
                  ( matrix_type::extents_type::static_extent(0) != ::std::experimental::dynamic_extent ) ) )
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( ( vector_type::extents_type::static_extent(0) == matrix_type::extents_type::static_extent(0) ) ||
                  ( vector_type::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) ||
                  ( matrix_type::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) )
    #endif
    {
      // If the extents are dynamic, then check they are compatable
      if constexpr ( ( vector_type::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) ||
                     ( matrix_type::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) )
      {
        // If sizes are not compatable, then throw exception
        if ( v.size() != m.size().extent(0) ) LINALG_UNLIKELY
        {
          throw length_error( "Matrix and vector sizes are incompatable." );
        }
      }
      // Define product operation on each element pair
      auto lambda = [&v,&m]( auto index ) constexpr noexcept
      {
        result_value_type result = 0;
        detail::for_each( LINALG_EXECUTION_UNSEQ,
                          detail::faux_index_iterator<typename vector_type::index_type>( 0 ),
                          detail::faux_index_iterator<typename vector_type::index_type>( v.size().extent(0) ),
                          [ &v, &m, &index, &result ] ( typename vector_type::index_type index2 ) constexpr noexcept
                            { result += detail::access( v, index2 ) * detail::access( m, index2, index ); } );
        return result;
      };
      // Construct multiplication vector
      return detail::make_from_tuple<pre_result_vector_type>( collect_ctor_args( v, m, lambda ) );
    }
    /// @brief Returns v *= m
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename Vec = vector_type,
               typename Mat = matrix_type,
               typename = ::std::enable_if_t< ( ( Vec::extents_type::static_extent(0) == Mat::extents_type::static_extent(0) ) &&
                                                ( ( Vec::extents_type::static_extent(0) == Mat::extents_type::static_extent(1) ) ||
                                                  ( Vec::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) ) ) > >
    #endif
    [[nodiscard]] static constexpr vector_type& prod( vector_type& v, const matrix_type& m )
      noexcept( noexcept( v = ::std::move( v * m ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( ( vector_type::extents_type::static_extent(0) == matrix_type::extents_type::static_extent(0) ) &&
                 ( ( vector_type::extents_type::static_extent(0) == matrix_type::extents_type::static_extent(1) ) ||
                   ( vector_type::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) ) )
    #endif
    {
      return v = ::std::move( v * m );
    }
    /// @brief Returns m * v
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename Vec = vector_type,
               typename Mat = matrix_type,
               typename = ::std::enable_if_t< ( ( Mat::extents_type::static_extent(1) == Vec::extents_type::static_extent(0) ) ||
                                                ( Mat::extents_type::static_extent(1) == ::std::experimental::dynamic_extent ) ||
                                                ( Vec::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) ) > >
    #endif
    [[nodiscard]] static constexpr post_result_vector_type prod( const matrix_type& m, const vector_type& v )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept(
        detail::make_from_tuple< post_result_vector_type >(
          collect_ctor_args( ::std::declval<const vector_type&>(),
                             ::std::declval<const matrix_type&>(),
                             #ifndef LINALG_COMPILER_CLANG
                             [&v,&m]( auto index ) constexpr noexcept
                             {
                               result_value_type result = 0;
                               detail::for_each( LINALG_EXECUTION_UNSEQ,
                                                 detail::faux_index_iterator<typename vector_type::index_type>( 0 ),
                                                 detail::faux_index_iterator<typename vector_type::index_type>( v.size().extent(0) ),
                                                 [ &v, &m, &index, &result ] ( typename vector_type::index_type index2 ) constexpr noexcept
                                                   { result += detail::access( m, index, index2 ) * detail::access( v, index2 ); } );
                               return result; }
                             #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                             []( auto index ) constexpr noexcept
                             { return result_value_type(); }
                             #endif
                             ) ) ) &&
                ( ( matrix_type::extents_type::static_extent(1) != ::std::experimental::dynamic_extent ) &&
                  ( vector_type::extents_type::static_extent(0) != ::std::experimental::dynamic_extent ) ) )
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( ( matrix_type::extents_type::static_extent(1) == vector_type::extents_type::static_extent(0) ) ||
                  ( matrix_type::extents_type::static_extent(1) == ::std::experimental::dynamic_extent ) ||
                  ( vector_type::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) )
    #endif
    {
      // If the extents are dynamic, then check they are compatable
      if constexpr ( ( matrix_type::extents_type::static_extent(1) == ::std::experimental::dynamic_extent ) ||
                     ( vector_type::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) )
      {
        // If sizes are not compatable, then throw exception
        if ( v.size() != m.size().extent(1) ) LINALG_UNLIKELY
        {
          throw length_error( "Matrix and vector sizes are incompatable." );
        }
      }
      // Define product operation on each element pair
      auto lambda = [&v,&m]( auto index ) constexpr noexcept
      {
        result_value_type result = 0;
        detail::for_each( LINALG_EXECUTION_UNSEQ,
                          detail::faux_index_iterator<typename vector_type::index_type>( 0 ),
                          detail::faux_index_iterator<typename vector_type::index_type>( v.size().extent(0) ),
                          [ &v, &m, &index, &result ] ( typename vector_type::index_type index2 ) constexpr noexcept
                            { result += detail::access( m, index, index2 ) * detail::access( v, index2 ); } );
        return result;
      };
      // Construct multiplication vector
      return detail::make_from_tuple<post_result_vector_type>( collect_ctor_args( m, v, lambda ) );
    }
};

/// @brief Defines multiplication operation on a pair of matrices
/// @tparam M1 matrix
/// @tparam M2 matrix
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::matrix_data M1, concepts::matrix_data M2 >
#else
template < class M1, class M2, typename = ::std::enable_if_t< concepts::matrix_data_v<M1> && concepts::matrix_data_v<M2> > >
#endif
class matrix_matrix_product
{
  public:
    //- Types

    /// @brief First input matrix type
    using first_matrix_type  = M1;
    /// @brief Second input matrix type
    using second_matrix_type = M2;
  private:
    // Return the allocator of m1
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename Mat = first_matrix_type,
               typename = ::std::enable_if_t< concepts::dynamic_matrix_data_v< Mat > > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( const first_matrix_type& m1, [[maybe_unused]] const second_matrix_type& m2 ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_matrix_data< first_matrix_type >
    #endif
    {
      return m1.get_allocator();
    }
    // Returns the allocator of m2
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename First_mat = first_matrix_type,
               typename Second_mat = second_matrix_type,
               typename = ::std::enable_if_t< concepts::dynamic_matrix_data_v< Second_mat > && !concepts::dynamic_matrix_data_v< First_mat > >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( [[maybe_unused]] const first_matrix_type& m1, const second_matrix_type& m2 ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( concepts::dynamic_matrix_data< second_matrix_type > && !concepts::dynamic_matrix_data< first_matrix_type > )
    #endif
    {
      return m2.get_allocator();
    }
    // Returns the std allocator
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename First_mat = first_matrix_type,
               typename Second_mat = second_matrix_type,
               typename = ::std::enable_if_t< !concepts::dynamic_matrix_data_v< First_mat > && !concepts::dynamic_matrix_data_v< Second_mat > >,
               typename = ::std::enable_if_t<true>,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( [[maybe_unused]] const first_matrix_type& m1, [[maybe_unused]] const second_matrix_type& m2 ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( !concepts::dynamic_matrix_data< first_matrix_type > && !concepts::dynamic_matrix_data< second_matrix_type > )
    #endif
    {
      return ::std::allocator<result_value_type>();
    }
    // Aliases
    using result_value_type  = ::std::decay_t< decltype( ::std::declval<typename first_matrix_type::value_type>() * ::std::declval<typename second_matrix_type::value_type>() ) >;
    using result_matrix_type = ::std::conditional_t< 
                                                     #ifdef LINALG_ENABLE_CONCEPTS
                                                     concepts::fixed_size_matrix_data<first_matrix_type> && concepts::fixed_size_matrix_data<second_matrix_type>,
                                                     #else
                                                     concepts::fixed_size_matrix_data_v<first_matrix_type> && concepts::fixed_size_matrix_data_v<second_matrix_type>,
                                                     #endif
                                                     fs_matrix< result_value_type,
                                                                first_matrix_type::extents_type::static_extent(0),
                                                                second_matrix_type::extents_type::static_extent(1),
                                                                detail::rebind_layout_t< typename first_matrix_type::layout_type,
                                                                                         ::std::experimental::extents< typename first_matrix_type::size_type,
                                                                                                                       first_matrix_type::extents_type::static_extent(0),
                                                                                                                       second_matrix_type::extents_type::static_extent(1) > >,
                                                               detail::rebind_accessor_t< typename first_matrix_type::accessor_type,result_value_type > >,
                                                     dr_matrix< result_value_type,
                                                                typename allocator_traits< ::std::decay_t< decltype( get_allocator( ::std::declval<first_matrix_type>(), ::std::declval<second_matrix_type>() ) ) > >::template rebind_alloc<result_value_type>,
                                                                detail::rebind_layout_t< typename first_matrix_type::layout_type,
                                                                                         ::std::experimental::extents< typename first_matrix_type::index_type,
                                                                                                                       ::std::experimental::dynamic_extent,
                                                                                                                       ::std::experimental::dynamic_extent> >,
                                                                detail::rebind_accessor_t< typename first_matrix_type::accessor_type,result_value_type > > >;
    // Gets necessary arguments for constrution
    // If matrix type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_matrix = result_matrix_type,
               typename = ::std::enable_if_t< concepts::fixed_size_matrix_data_v<Result_matrix> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const first_matrix_type&, [[maybe_unused]] const second_matrix_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_matrix_data<result_matrix_type>
    #endif
    { return tuple( forward<Lambda>( lambda ) ); }
    // If the matrix type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default constructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_matrix = result_matrix_type,
               typename = ::std::enable_if_t< concepts::dynamic_matrix_data_v<Result_matrix> >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const first_matrix_type& m1, const second_matrix_type& m2, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_matrix_data<result_matrix_type>
    #endif
    {
      if constexpr ( ::std::is_default_constructible_v<typename result_matrix_type::allocator_type> &&
                     ::std::allocator_traits<typename result_matrix_type::allocator_type>::is_always_equal::value )
      {
        using result_extents_type = typename result_matrix_type::extents_type;
        return ::std::tuple( result_extents_type( m1.size().extent(0), m2.size().extent(1) ),
                             result_extents_type( m1.capacity().extent(0), m2.capacity().extent(1) ),
                             ::std::forward<Lambda>( lambda ) );
      }
      else
      {
        using result_extents_type = typename result_matrix_type::extents_type;
        using result_alloc_type   = typename result_matrix_type::allocator_type;
        return ::std::tuple( result_extents_type( m1.size().extent(0), m2.size().extent(1) ),
                             result_extents_type( m1.capacity().extent(0), m2.capacity().extent(1) ),
                             ::std::forward<Lambda>( lambda ),
                             result_alloc_type( get_allocator( m1, m2 ) ) );
      }
    }
  public:
    //- Operations

    /// @brief computes m1 * m2
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename First_mat = first_matrix_type,
               typename Second_mat = second_matrix_type,
               typename = ::std::enable_if_t< ( ( First_mat::extents_type::static_extent(1) == Second_mat::extents_type::static_extent(0) ) ||
                                                ( First_mat::extents_type::static_extent(1) == ::std::experimental::dynamic_extent ) ||
                                                ( Second_mat::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) ) > >
    #endif
    [[nodiscard]] static constexpr result_matrix_type prod( const first_matrix_type& m1, const second_matrix_type& m2 )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept( detail::make_from_tuple< result_matrix_type >(
        collect_ctor_args( ::std::declval<const first_matrix_type&>(),
                           ::std::declval<const second_matrix_type&>(),
                           #ifndef LINALG_COMPILER_CLANG
                           [&m1,&m2]( auto index1, auto index2 ) constexpr noexcept
                           {
                             result_value_type result = 0;
                             detail::for_each( LINALG_EXECUTION_UNSEQ,
                                               detail::faux_index_iterator<typename first_matrix_type::index_type>( 0 ),
                                               detail::faux_index_iterator<typename first_matrix_type::index_type>( m1.size().extent(1) ),
                                               [ &m1, &m2, &index1, &index2, &result ] ( typename first_matrix_type::index_type index ) constexpr noexcept
                                                 { result += detail::access( m1, index1, index ) * detail::access( m2, index, index2 ); } );
                             return result; }
                          #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                           []( auto index1, auto index2 ) constexpr noexcept
                           { return result_value_type(); }
                          #endif
                           ) ) ) &&
                ( first_matrix_type::extents_type::static_extent(1) != ::std::experimental::dynamic_extent ) &&
                ( second_matrix_type::extents_type::static_extent(0) != ::std::experimental::dynamic_extent ) )
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( ( first_matrix_type::extents_type::static_extent(1) == second_matrix_type::extents_type::static_extent(0) ) ||
                 ( first_matrix_type::extents_type::static_extent(1) == ::std::experimental::dynamic_extent ) ||
                 ( second_matrix_type::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) )
    #endif
    {
      // If the extents are dynamic, then check they are compatable
      if constexpr ( ( first_matrix_type::extents_type::static_extent(1) == ::std::experimental::dynamic_extent ) ||
                     ( second_matrix_type::extents_type::static_extent(0) == ::std::experimental::dynamic_extent ) )
      {
        // If sizes are not compatable, then throw exception
        if ( m1.size().extent(1) != m2.size().extent(0) ) LINALG_UNLIKELY
        {
          throw length_error( "Matrix sizes are incompatable." );
        }
      }
      // Define product operation on each element pair
      auto lambda = [&m1,&m2]( auto index1, auto index2 ) constexpr noexcept
      {
        result_value_type result = 0;
        detail::for_each( LINALG_EXECUTION_UNSEQ,
                          detail::faux_index_iterator<typename first_matrix_type::index_type>( 0 ),
                          detail::faux_index_iterator<typename first_matrix_type::index_type>( m1.size().extent(1) ),
                          [ &m1, &m2, &index1, &index2, &result ] ( typename first_matrix_type::index_type index ) constexpr noexcept
                            { result += detail::access( m1, index1, index ) * detail::access( m2, index, index2 ); } );
        return result;
      };
      // Construct multiplication matrix
      return detail::make_from_tuple<result_matrix_type>( collect_ctor_args( m1, m2, lambda ) );
    }
    /// @brief Returns m1 *= m2
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename First_mat = first_matrix_type,
               typename Second_mat = second_matrix_type,
               typename = ::std::enable_if_t< ( ( First_mat::extents_type::static_extent(1) == Second_mat::extents_type::static_extent(0) ) &&
                                                ( ( First_mat::extents_type::static_extent(1) == Second_mat::extents_type::static_extent(1) ) ||
                                                  ( First_mat::extents_type::static_extent(1) == ::std::experimental::dynamic_extent ) ) ) > >
    #endif
    [[nodiscard]] static constexpr first_matrix_type& prod( first_matrix_type& m1, const second_matrix_type& m2 )
      noexcept( noexcept( m1 = ::std::move( m1 * m2 ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( ( first_matrix_type::extents_type::static_extent(1) == second_matrix_type::extents_type::static_extent(0) ) &&
                 ( ( first_matrix_type::extents_type::static_extent(1) == second_matrix_type::extents_type::static_extent(1) ) ||
                   ( first_matrix_type::extents_type::static_extent(1) == ::std::experimental::dynamic_extent ) ) )
    #endif
    {
      return m1 = ::std::move( m1 * m2 );
    }
};

/// @brief Defines inner product operation on a pair of vectors
/// @tparam V1 vector
/// @tparam V2 vector
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::vector_data V1, concepts::vector_data V2 >
#else
template < class V1, class V2, typename = ::std::enable_if_t< concepts::vector_data_v< V2 > > >
#endif
class inner_product
{
  public:
    //- Types

    /// @brief First input vector type
    using first_vector_type  = V1;
    /// @brief Second input vector type
    using second_vector_type = V2;
  private:
    // Aliases
    using result_type        = decltype( ::std::declval<typename first_vector_type::value_type>() * ::std::declval<typename second_vector_type::value_type>() );
  public:
    //- Operations

    /// @brief computes the inner product of v1 and v2
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename First_extents = typename first_vector_type::extents_type,
               typename Second_extents = typename second_vector_type::extents_type,
               typename = ::std::enable_if_t< detail::extents_may_be_equal_v<First_extents,Second_extents> > >
    #endif
    [[nodiscard]] static constexpr auto prod( const first_vector_type& v1, const second_vector_type& v2 )
      noexcept( detail::extents_are_equal_v<typename first_vector_type::extents_type,typename second_vector_type::extents_type> )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires detail::extents_may_be_equal_v<typename first_vector_type::extents_type,typename second_vector_type::extents_type>
    #endif
    {
      if constexpr ( !detail::extents_are_equal_v<typename first_vector_type::extents_type,typename second_vector_type::extents_type> )
      {
        // Check if sizes are equal
        if ( !( v1.size() == v2.size() ) ) LINALG_UNLIKELY
        {
          throw length_error( "Vectors must have same size." );
        }
      }
      // Store sum of inner product
      result_type result = 0;
      // Define lambda function to sum inner product
      auto inner_prod_lambda = [&v1,&v2,&result]( auto index ) constexpr noexcept
        { result += detail::access( v1, index ) * detail::access( v2, index ); };
      // Apply lambda expression
      detail::apply_all( v1.span(), inner_prod_lambda, LINALG_EXECUTION_UNSEQ );
      // Return result
      return result;
    }
};

/// @brief Defines outer product operation on a pair of vectors
/// @tparam V1 vector
/// @tparam V2 vector
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::vector_data V1, concepts::vector_data V2 >
#else
template < class V1, class V2, typename = ::std::enable_if_t< concepts::vector_data_v<V1> && concepts::vector_data_v<V2> > >
#endif
class outer_product
{
  public:
    //- Types

    /// @brief First input vector type
    using first_vector_type  = V1;
    /// @brief Second input vector type
    using second_vector_type = V2;
  private:
    // Return the allocator of v1
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename First_vector = first_vector_type,
               typename = ::std::enable_if_t< concepts::dynamic_vector_data_v< First_vector > > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( const first_vector_type& v1, [[maybe_unused]] const second_vector_type& v2 ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_vector_data< first_vector_type >
    #endif
    {
      return v1.get_allocator();
    }
    // Returns the allocator of v2
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename First_vector  = first_vector_type,
               typename Second_vector = second_vector_type,
               typename = ::std::enable_if_t< concepts::dynamic_vector_data_v< Second_vector > && !concepts::dynamic_vector_data_v< First_vector > >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( [[maybe_unused]] const first_vector_type& v1, const second_vector_type& v2 ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( concepts::dynamic_vector_data< second_vector_type > && !concepts::dynamic_vector_data< first_vector_type > )
    #endif
    {
      return v2.get_allocator();
    }
    // Returns the std allocator
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename First_vector  = first_vector_type,
               typename Second_vector = second_vector_type,
               typename = ::std::enable_if_t< !concepts::dynamic_vector_data_v< First_vector > && !concepts::dynamic_vector_data_v< Second_vector > >,
               typename = ::std::enable_if_t<true>,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( [[maybe_unused]] const first_vector_type& v1, [[maybe_unused]] const second_vector_type& v2 ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( !concepts::dynamic_vector_data< first_vector_type > && !concepts::dynamic_vector_data< second_vector_type > )
    #endif
    {
      return ::std::allocator<result_value_type>();
    }
    // Aliases
    using result_value_type  = ::std::decay_t< decltype( ::std::declval<typename first_vector_type::value_type>() * ::std::declval<typename second_vector_type::value_type>() ) >;
    using result_matrix_type = ::std::conditional_t< 
                                                     #ifdef LINALG_ENABLE_CONCEPTS
                                                     concepts::fixed_size_vector_data<first_vector_type> && concepts::fixed_size_vector_data<second_vector_type>,
                                                     #else
                                                     concepts::fixed_size_vector_data_v<first_vector_type> && concepts::fixed_size_vector_data_v<second_vector_type>,
                                                     #endif
                                                     fs_matrix< result_value_type,
                                                                first_vector_type::extents_type::static_extent(0),
                                                                second_vector_type::extents_type::static_extent(0),
                                                                detail::rebind_layout_t< typename first_vector_type::layout_type,
                                                                                         ::std::experimental::extents< typename first_vector_type::size_type,
                                                                                                                       first_vector_type::extents_type::static_extent(0),
                                                                                                                       second_vector_type::extents_type::static_extent(0)> >,
                                                                detail::rebind_accessor_t<typename first_vector_type::accessor_type,result_value_type> >,
                                                     dr_matrix< result_value_type,
                                                                typename ::std::allocator_traits< ::std::decay_t< decltype( get_allocator( ::std::declval<first_vector_type>(), ::std::declval<second_vector_type>() ) ) > >::template rebind_alloc<result_value_type>,
                                                                detail::rebind_layout_t< typename first_vector_type::layout_type,
                                                                                         ::std::experimental::extents< typename first_vector_type::size_type,
                                                                                                                       ::std::experimental::dynamic_extent,
                                                                                                                       ::std::experimental::dynamic_extent> >,
                                                                detail::rebind_accessor_t<typename first_vector_type::accessor_type,result_value_type> > >;
    // Gets necessary arguments for constrution
    // If matrix type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_matrix = result_matrix_type,
               typename = ::std::enable_if_t< concepts::fixed_size_matrix_data_v<Result_matrix> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const first_vector_type&, [[maybe_unused]] const second_vector_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_matrix_data<result_matrix_type>
    #endif
    { return ::std::tuple( ::std::forward<Lambda>( lambda ) ); }
    // If the matrix type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default cosntructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename Result_matrix = result_matrix_type,
               typename = ::std::enable_if_t< concepts::dynamic_matrix_data_v<Result_matrix> >,
               typename = ::std::enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const first_vector_type& v1, const second_vector_type& v2, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_matrix_data<result_matrix_type>
    #endif
    {
      if constexpr ( ::std::is_default_constructible_v<typename result_matrix_type::allocator_type> &&
                     ::std::allocator_traits<typename result_matrix_type::allocator_type>::is_always_equal::value )
      {
        using result_extents_type = typename result_matrix_type::extents_type;
        return ::std::tuple( result_extents_type( v1.size(), v2.size() ),
                             result_extents_type( v1.capacity(), v2.capacity() ),
                             ::std::forward<Lambda>( lambda ) );
      }
      else
      {
        using result_extents_type = typename result_matrix_type::extents_type;
        using result_alloc_type   = typename result_matrix_type::allocator_type;
        return ::std::tuple( result_extents_type( v1.size(), v2.size() ),
                             result_extents_type( v1.capacity(), v2.capacity() ),
                             ::std::forward<Lambda>( lambda ),
                             result_alloc_type( get_allocator(v1,v2) ) );
      }
    }
  public:
    /// @brief Computes the outer product of v1 and v2
    [[nodiscard]] static constexpr result_matrix_type prod( const first_vector_type& v1, const second_vector_type& v2 )
      #ifdef LINALG_UNEVALUATED_LAMBDA
      noexcept( noexcept( detail::make_from_tuple< result_matrix_type >(
        collect_ctor_args( v1,
                           v2,
                           #ifndef LINALG_COMPILER_CLANG
                           [&v1,&v2]( auto index1, auto index2 ) constexpr noexcept
                             { return detail::access( v1, index1 ) * detail::access( v2, index2 ); } ) ) ) )
                           #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                           []( auto index1, auto index2 ) constexpr noexcept
                             { return typename result_matrix_type::value_type(); } ) ) ) )
                           #endif
      #else
      // Cannot assume the constructor is noexcept. Just leave with no exception specification declared.
      #endif
    {
      // Define product operation on each element pair
      auto lambda = [&v1,&v2]( auto index1, auto index2 ) constexpr noexcept
      {
        return detail::access( v1, index1 ) * detail::access( v2, index2 );
      };
      // Construct multiplication vector
      return detail::make_from_tuple<result_matrix_type>( collect_ctor_args( v1, v2, lambda ) );
    }
};

}       //- instant_evaluated_operations
}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_OPERATION_TRAITS_HPP_DEFINED
