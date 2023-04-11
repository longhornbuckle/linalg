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

/// @brief Defines negation operation on a tensor
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_data T >
#else
template < class T, typename = enable_if_t< concepts::tensor_data_v<T> > >
#endif
class negation
{
  public:
    //- Types

    /// @brief Input tensor type
    using tensor_type        = T;
  private:
    // Aliases
    using result_value_type  = decay_t< decltype( - declval<typename tensor_type::value_type>() ) >;
    using result_tensor_type = typename tensor_type::template rebind_t<result_value_type>;
    // Gets necessary arguments for constrution
    // If engine type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::fixed_size_tensor_data_v<result_tensor_type> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const tensor_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_tensor_data<result_tensor_type>
    #endif
    { return tuple( forward<Lambda>( lambda ) ); }
    // If the engine type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default constructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::dynamic_tensor_data_v<result_tensor_type> >, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const tensor_type& t, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_tensor_data<result_tensor_type>
    #endif
    {
      if constexpr ( is_default_constructible_v<typename result_tensor_type::allocator_type> &&
                     allocator_traits<typename result_tensor_type::allocator_type>::is_always_equal::value )
      {
        return tuple( t.size(), t.capacity(), forward<Lambda>( lambda ) );
      }
      else
      {
        using result_alloc_type = typename allocator_traits<typename tensor_type::allocator_type>::template rebind_alloc<result_value_type>;
        return tuple( t.size(), t.capacity(), forward<Lambda>( lambda ), result_alloc_type( t.get_allocator() ) );
      }
    }
  public:
    //- Operations

    /// @brief Returns -1 * tensor
    [[nodiscard]] static constexpr auto negate( const tensor_type& t )
      noexcept( noexcept( detail::make_from_tuple< result_tensor_type >(
        collect_ctor_args( declval<const tensor_type&>(),
        #ifndef LINALG_COMPILER_CLANG
                           [&t]< class ... IndexType >( IndexType ... indices ) constexpr noexcept { return -( detail::access( t, indices ... ) ); } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                           []< class ... IndexType >( IndexType ... indices ) constexpr noexcept { return tensor_type::value_type(); } ) ) ) )
        #endif
    {
      // Define negation operation on each element
      auto negate_lambda = [&t]< class ... IndexType >( IndexType ... indices ) constexpr noexcept { return -( detail::access( t, indices ... ) ); };
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
template < class T1, class T2, typename = enable_if_t< concepts::tensor_data_v<T1> && concepts::tensor_data_v<T2> > >
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
    using result_value_type  = decay_t< decltype( declval<typename first_tensor_type::value_type>() + declval<typename second_tensor_type::value_type>() ) >;
    using result_tensor_type = conditional_t< 
                                              #ifdef LINALG_ENABLE_CONCEPTS
                                              concepts::dynamic_tensor_data<first_tensor_type> &&
                                                concepts::fixed_size_tensor_data<second_tensor_type>,
                                              #else
                                              concepts::dynamic_tensor_data_v<first_tensor_type> &&
                                                concepts::fixed_size_tensor_data_v<second_tensor_type>,
                                              #endif
                                              typename second_tensor_type::template rebind_t<result_value_type>,
                                              typename first_tensor_type::template rebind_t<result_value_type> >;
    // Gets necessary arguments for constrution
    // If engine type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::fixed_size_tensor_data_v<result_tensor_type> > >
    #endif
    [[nodiscard]] static inline constexpr auto collect_ctor_args( [[maybe_unused]] const first_tensor_type&, [[maybe_unused]] const second_tensor_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_tensor_data<result_tensor_type>
    #endif
    { return tuple( forward<Lambda>( lambda ) ); }
    // If the engine type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default cosntructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::dynamic_tensor_data_v<result_tensor_type> >, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr auto collect_ctor_args( const first_tensor_type& t1, const second_tensor_type& t2, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_tensor_data<result_tensor_type>
    #endif
    {
      if constexpr ( is_default_constructible_v<typename result_tensor_type::allocator_type> &&
                     allocator_traits<typename result_tensor_type::allocator_type>::is_always_equal::value )
      {
        if constexpr ( is_same_v<first_tensor_type,result_tensor_type> )
        {
          return tuple( t1.size(), t1.capacity(), forward<Lambda>( lambda ) );
        }
        else
        {
          return tuple( t2.size(), t2.capacity(), forward<Lambda>( lambda ) );
        }
      }
      else
      {
        if constexpr ( is_same_v<first_tensor_type,result_tensor_type> )
        {
          using result_alloc_type = typename allocator_traits<typename first_tensor_type::allocator_type>::template rebind_alloc<result_value_type>;
          return tuple( t1.size(), t1.capacity(), forward<Lambda>( lambda ), result_alloc_type( t1.get_allocator() ) );
        }
        else
        {
          using result_alloc_type = typename allocator_traits<typename second_tensor_type::allocator_type>::template rebind_alloc<result_value_type>;
          return tuple( t2.size(), t2.capacity(), forward<Lambda>( lambda ), result_alloc_type( t2.get_allocator() ) );
        }
      }
    }
    // LEAVING FOR REFERENCE IN CASE NEED LATER
    // template < class T1, class T2, class ... IndexType >
    // struct Lambda_helper2
    // {
    //   T1&& t1;
    //   T2&& t2;
    //   [[nodiscard]] constexpr auto add_lambda() noexcept
    //     { return [this]( IndexType ... indices ) constexpr noexcept { return this->t1[ indices ... ] + this->t2[ indices ... ]; }; }
    // };
    // template < class T1, class T2, class T3 >
    // struct Lambda_helper { };
    // template < class T1, class T2, class IndexType, IndexType ... Indices >
    // struct Lambda_helper< T1, T2, integer_sequence< IndexType, Indices ... > >
    // {
    //   using more_helper = Lambda_helper2< T1, T2, decltype( get<Indices>( make_tuple( Indices ... ) ) ) ... >;
    // };
  public:
    //- Operations

    /// @brief Returns t1 + t2
    [[nodiscard]] static constexpr auto add( const first_tensor_type& t1, const second_tensor_type& t2 )
      noexcept( noexcept( detail::make_from_tuple< result_tensor_type >(
        collect_ctor_args( declval<const first_tensor_type&>(),
                            declval<const second_tensor_type&>(),
        #ifndef LINALG_COMPILER_CLANG
                            [&t1,&t2]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                              { return detail::access( t1, indices ... ) + detail::access( t2, indices ... ); } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                            []< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                              { return typename result_tensor_type::value_type(); } ) ) ) )
        #endif
    {
      // LEAVING FOR REFERENCE IN CASE NEED LATER
      // Define addition operation on each element pair
      // auto add_lambda = typename Lambda_helper< const first_tensor_type&,
      //                                           const second_tensor_type&,
      //                                           make_integer_sequence< typename result_tensor_type::index_type,
      //                                                                  result_tensor_type::extents_type::rank() > >::more_helper
      //                     ( t1, t2 ).add_lambda();
      auto add_lambda = [&t1,&t2]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
        { return detail::access( t1, indices ... ) + detail::access( t2, indices ... ); };
      // Construct addition tensor
      return detail::make_from_tuple<result_tensor_type>( collect_ctor_args( t1, t2, add_lambda ) );
    }
    /// @brief Returns t1 += t2
    [[nodiscard]] static constexpr first_tensor_type& add( first_tensor_type& t1, const second_tensor_type& t2 )
      noexcept( noexcept( detail::apply_all( t1.underlying_span(),
      #ifndef LINALG_COMPILER_CLANG
                                             [&t1,&t2]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                                               { static_cast<void>( detail::access( t1, indices ... ) += detail::access( t2, indices ... ) ); },
      #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                                             []< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                                               { },
      #endif
                                             LINALG_EXECUTION_UNSEQ ) ) )
    {
      auto add_lambda = [&t1,&t2]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
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
template < class T1, class T2, typename = enable_if_t< concepts::tensor_data_v<T1> && concepts::tensor_data_v<T2> > >
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
    using result_value_type  = decay_t< decltype( declval<typename first_tensor_type::value_type>() + declval<typename second_tensor_type::value_type>() ) >;
    using result_tensor_type = conditional_t< 
                                              #ifdef LINALG_ENABLE_CONCEPTS
                                              concepts::dynamic_tensor_data<first_tensor_type> &&
                                                concepts::fixed_size_tensor_data<second_tensor_type>,
                                              #else
                                              concepts::dynamic_tensor_data_v<first_tensor_type> &&
                                                concepts::fixed_size_tensor_data_v<second_tensor_type>,
                                              #endif
                                              typename second_tensor_type::template rebind_t<result_value_type>,
                                              typename first_tensor_type::template rebind_t<result_value_type> >;
    // Gets necessary arguments for constrution
    // If engine type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lamda, typename = enable_if_t< concepts::fixed_size_tensor_data_v<result_tensor_type> > >
    #endif
    [[nodiscard]] static inline constexpr auto collect_ctor_args( [[maybe_unused]] const first_tensor_type&, [[maybe_unused]] const second_tensor_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_tensor_data<result_tensor_type>
    #endif
    { return tuple( forward<Lambda>( lambda ) ); }
    // If the engine type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default cosntructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lamda, typename = enable_if_t< concepts::dynamic_tensor_data_v<result_tensor_type> >, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr auto collect_ctor_args( const first_tensor_type& t1, const second_tensor_type& t2, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_tensor_data<result_tensor_type>
    #endif
    {
      if constexpr ( is_default_constructible_v<typename result_tensor_type::allocator_type> &&
                     allocator_traits<typename result_tensor_type::allocator_type>::is_always_equal::value )
      {
        if constexpr ( is_same_v<first_tensor_type,result_tensor_type> )
        {
          return tuple( t1.size(), t1.capacity(), forward<Lambda>( lambda ) );
        }
        else
        {
          return tuple( t2.size(), t2.capacity(), forward<Lambda>( lambda ) );
        }
      }
      else
      {
        if constexpr ( is_same_v<first_tensor_type,result_tensor_type> )
        {
          using result_alloc_type = typename allocator_traits<typename first_tensor_type::allocator_type>::template rebind_alloc<result_value_type>;
          return tuple( t1.size(), t1.capacity(), forward<Lambda>( lambda ), result_alloc_type( t1.get_allocator() ) );
        }
        else
        {
          using result_alloc_type = typename allocator_traits<typename second_tensor_type::allocator_type>::template rebind_alloc<result_value_type>;
          return tuple( t2.size(), t2.capacity(), forward<Lambda>( lambda ), result_alloc_type( t2.get_allocator() ) );
        }
      }
    }
  public:
    //- Operations

    /// @brief Returns t1 - t2
    [[nodiscard]] static constexpr auto subtract( const first_tensor_type& t1, const second_tensor_type& t2 )
      noexcept( noexcept( detail::make_from_tuple< result_tensor_type >(
        collect_ctor_args( declval<const first_tensor_type&>(),
                            declval<const second_tensor_type&>(),
        #ifndef LINALG_COMPILER_CLANG
                            [&t1,&t2]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                              { return detail::access( t1, indices ... ) - detail::access( t2, indices ... ); } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                            []< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                              { return typename result_tensor_type::value_type(); } ) ) ) )
        #endif
    {
      auto subtract_lambda = [&t1,&t2]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
        { return detail::access( t1, indices ... ) - detail::access( t2, indices ... ); };
      // Construct addition tensor
      return detail::make_from_tuple<result_tensor_type>( collect_ctor_args( t1, t2, subtract_lambda ) );
    }
    /// @brief Returns t1 -= t2
    [[nodiscard]] static constexpr first_tensor_type& subtract( first_tensor_type& t1, const second_tensor_type& t2 )
      noexcept( noexcept( detail::apply_all( t1.underlying_span(),
        #ifndef LINALG_COMPILER_CLANG
                                             [&t1,&t2]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                                               { static_cast<void>( detail::access( t1, indices ... ) -= detail::access( t2, indices ... ) ); },
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                                             []< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                                               { },
        #endif
                                             LINALG_EXECUTION_UNSEQ ) ) )
    {
      auto subtract_lambda = [&t1,&t2]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
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
template < class S, class T, typename = enable_if_t< concepts::tensor_data_v<T> > >
#endif
class scalar_product
{
  public:
    //- Types

    /// @brief Input tensor type
    using tensor_type        = T;
  private:
    // Aliases
    using result_value_type  = decay_t< decltype( declval<typename tensor_type::value_type>() * declval<S>() ) >;
    using result_tensor_type = typename tensor_type::template rebind_t<result_value_type>;
    // Gets necessary arguments for constrution
    // If tensor type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::fixed_size_tensor_data_v<result_tensor_type> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const tensor_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_tensor_data<result_tensor_type> { return tuple( forward<Lambda>( lambda ) ); }
    #endif
    // If the tensor type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default cosntructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::dynamic_tensor_data_v<result_tensor_type> >, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const tensor_type& t, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_tensor_data<result_tensor_type>
    #endif
    {
      if constexpr ( is_default_constructible_v<typename result_tensor_type::allocator_type> &&
                     allocator_traits<typename result_tensor_type::allocator_type>::is_always_equal::value )
      {
        return tuple( t.size(), t.capacity(), forward<Lambda>( lambda ) );
      }
      else
      {
        using result_alloc_type = typename allocator_traits<typename tensor_type::allocator_type>::template rebind_alloc<result_value_type>;
        return tuple( t.size(), t.capacity(), forward<Lambda>( lambda ), result_alloc_type( t.get_allocator() ) );
      }
    }
  public:
    //- Operations

    /// @brief Returns s * t
    [[nodiscard]] static constexpr auto prod( const S& s, const tensor_type& t )
      noexcept( noexcept( detail::make_from_tuple< result_tensor_type >(
        collect_ctor_args( t,
        #ifndef LINALG_COMPILER_CLANG
                           [&s,&t]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                             { return s * detail::access( t, indices ... ); } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                           []< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                             { return typename result_tensor_type::value_type(); } ) ) ) )
        #endif
    {
      // Define product operation on each element
      auto prod_lambda = [&s,&t]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
        { return s * detail::access( t, indices ... ); };
      // Construct product tensor
      return detail::make_from_tuple<result_tensor_type>( collect_ctor_args( t, prod_lambda ) );
    }
    /// @brief Returns t * s
    [[nodiscard]] static constexpr auto prod( const tensor_type& t, const S& s )
      noexcept( noexcept( prod( s, t ) ) )
    {
      return prod( s, t );
    }
    /// @brief Returns t *= s
    [[nodiscard]] static constexpr tensor_type& prod( tensor_type& t, const S& s )
      noexcept( noexcept( detail::apply_all( t.underlying_span(),
      #ifndef LINALG_COMPILER_CLANG
                                             [&t,&s]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                                               { static_cast<void>( detail::access( t, indices ... ) += s ); },
      #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                                             []< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                                               { },
      #endif
                                             LINALG_EXECUTION_UNSEQ ) ) )
    {
      // Define product operation on each element
      auto prod_lambda = [&s,&t]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
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
template < class T, class S, enable_if_t< concepts::tensor_data_v<T> > >
#endif
struct scalar_division
{
  public:
    //- Types

    /// @brief Input tensor type
    using tensor_type        = T;
  private:
    // Aliases
    using result_value_type  = decay_t< decltype( declval<typename tensor_type::value_type>() / declval<S>() ) >;
    using result_tensor_type = typename tensor_type::template rebind_t<result_value_type>;
    // Gets necessary arguments for constrution
    // If tensor type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_it< concepts::fixed_size_tensor_data_v<result_tensor_type> > >
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
    template < class Lambda, typename = enable_if_it< concepts::dynamic_tensor_data_v<result_tensor_type> >, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const tensor_type& t, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_tensor_data<result_tensor_type>
    #endif
    {
      if constexpr ( is_default_constructible_v<typename result_tensor_type::allocator_type> &&
                      allocator_traits<typename result_tensor_type::allocator_type>::is_always_equal::value )
      {
        return tuple( t.size(), t.capacity(), forward<Lambda>( lambda ) );
      }
      else
      {
        using result_alloc_type = typename allocator_traits<typename tensor_type::allocator_type>::template rebind_alloc<result_value_type>;
        return tuple( t.size(), t.capacity(), forward<Lambda>( lambda ), result_alloc_type( t.get_allocator() ) );
      }
    }
  public:
    //- Operations

    /// @brief Returns t / s
    [[nodiscard]] static constexpr auto divide( const tensor_type& t, const S& s )
      noexcept( noexcept( detail::make_from_tuple< result_tensor_type >(
        collect_ctor_args( declval<const tensor_type&>(),
        #ifndef LINALG_COMPILER_CLANG
                            [&t,&s]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                              { return detail::access( t, indices ... ) / s; } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                            []< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                              { return typename result_tensor_type::value_type(); } ) ) ) )
        #endif
    {
      // Define division operation on each element
      auto divide_lambda = [&t,&s]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
        { return detail::access( t, indices ... ) / s; };
      // Construct divided tensor
      return detail::make_from_tuple<result_tensor_type>( collect_ctor_args( t, divide_lambda ) );
    }
    /// @brief Returns t /= s
    [[nodiscard]] static constexpr tensor_type& divide( tensor_type& t, const S& s )
      noexcept( noexcept( detail::apply_all( t.underlying_span(),
      #ifndef LINALG_COMPILER_CLANG
                                             [&t,&s]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                                               { static_cast<void>( t[ indices ... ] /= s ); },
      #else
                                             []< class ... IndexType >( IndexType ... indices ) constexpr noexcept
                                               { },
      #endif
                                             LINALG_EXECUTION_UNSEQ ) ) )
    {
      // Define product operation on each element
      auto divide_lambda = [&s,&t]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
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
template < class M, typename = enable_if_t< concepts::matrix_data_v<M> > >
#endif
class transpose_matrix
{
  public:
    //- Types

    /// @brief Input matrix type
    using matrix_type        = M;
  private:
    // Aliases
    using result_matrix_type = typename matrix_type::transpose_type;
    // Gets necessary arguments for constrution
    // If matrix type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::fixed_size_matrix_data_v<result_matrix_type> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const matrix_type&, Lambda&& lambda ) noexcept
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
    template < class Lambda, typename = enable_if_t< concepts::dynamic_matrix_data_v<result_matrix_type> >, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const matrix_type& m, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_matrix_data<result_matrix_type>
    #endif
    {
      using result_extents_type = typename result_matrix_type::extents_type;
      if constexpr ( is_default_constructible_v<typename result_matrix_type::allocator_type> &&
                      allocator_traits<typename result_matrix_type::allocator_type>::is_always_equal::value )
      {
        return tuple( result_extents_type( m.size().extent(1), m.size().extent(0) ),
                      result_extents_type( m.capacity().extent(0), m.capacity().extent(0) ),
                      forward<Lambda>( lambda ) );
      }
      else
      {
        using result_alloc_type = typename result_matrix_type::allocator_type;
        return tuple( result_extents_type( m.size().extent(1), m.size().extent(0) ),
                      result_extents_type( m.capacity().extent(1), m.capacity().extent(0) ),
                      forward<Lambda>( lambda ),
                      result_alloc_type( m.get_allocator() ) );
      }
    }
  public:
    //- Operations

    /// @brief Returns transpose( matrix )
    [[nodiscard]] static constexpr auto trans( const matrix_type& m )
      noexcept( noexcept( detail::make_from_tuple< result_matrix_type >(
        collect_ctor_args( m,
        #ifndef LINALG_COMPILER_CLANG
          [&m]< class IndexType1, class IndexType2 >( IndexType1 index1, IndexType2 index2 ) constexpr noexcept
            { return detail::access( m, index2, index1 ); } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
          []< class IndexType1, class IndexType2 >( IndexType1 index1, IndexType2 index2 ) constexpr noexcept
            { return typename matrix_type::value_type(); } ) ) ) )
        #endif
    {
      // Define negation operation on each element
      auto transpose_lambda = [&m]< class IndexType1, class IndexType2 >( IndexType1 index1, IndexType2 index2 ) constexpr noexcept
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
template < class V, typename = enable_if_t< concepts::vector_data<T> > >
#endif
class transpose_vector
{
  public:
    //- Types

    /// @brief Input vector type
    using vector_type        = V;
  private:
    // Aliases
    using result_Vector_type = vector_type;
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
template < class M, typename = enable_if_t< concepts::matrix_data_v<M> > >
#endif
class conjugate_matrix
{
  public:
    //- Types

    /// @brief Input matrix type
    using matrix_type         = M;
  private:
    // Aliases
    using result_element_type = conditional_t< detail::is_complex_v<typename matrix_type::element_type>,
                                               decltype( conj( declval<typename matrix_type::element_type>() ) ),
                                               typename matrix_type::element_type >;
    using result_matrix_type  = typename matrix_type::transpose_type::template rebind_t<result_element_type>;
    // Gets necessary arguments for constrution
    // If matrix type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::fixed_size_matrix_data_v<result_matrix_type> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const matrix_type&, Lambda&& lambda ) noexcept
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
    template < class Lambda, typename = enable_if_t< concepts::dynamic_matrix_data_v<result_matrix_type> >, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const matrix_type& m, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_matrix_data<result_matrix_type>
    #endif
    {
      using result_extents_type = typename result_matrix_type::extents_type;
      if constexpr ( is_default_constructible_v<typename result_matrix_type::allocator_type> &&
                      allocator_traits<typename result_matrix_type::allocator_type>::is_always_equal::value )
      {
        return tuple( result_extents_type( m.size().extent(1), m.size().extent(0) ),
                      result_extents_type( m.capacity().extent(1), m.capacity().extent(0) ),
                      forward<Lambda>( lambda ) );
      }
      else
      {
        using result_alloc_type = typename result_matrix_type::allocator_type;
        return tuple( result_extents_type( m.size().extent(1), m.size().extent(0) ),
                      result_extents_type( m.capacity().extent(1), m.capacity().extent(0) ),
                      forward<Lambda>( lambda ),
                      result_alloc_type( m.get_allocator() ) );
      }
    }
  public:
    //- Operations

    /// @brief Returns transpose conjugate( matrix )
    [[nodiscard]] static constexpr auto conjugate( const matrix_type& m )
      noexcept( noexcept( detail::make_from_tuple< result_matrix_type >(
        collect_ctor_args( m,
        #ifndef LINALG_COMPILER_CLANG
          [&m]< class IndexType1, class IndexType2 >( IndexType1 index1, IndexType2 index2 ) constexpr noexcept
            { return ::std::conj( detail::access( m, index2, index1 ) ); } ) ) ) )
        #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
          []< class IndexType1, class IndexType2 >( IndexType1 index1, IndexType2 index2 ) constexpr noexcept
            { return ::std::conj( typename matrix_type::value_type() ); } ) ) ) )
        #endif
    {
      // Define negation operation on each element
      auto conjugate_lambda = [&m]< class IndexType1, class IndexType2 >( IndexType1 index1, IndexType2 index2 ) constexpr noexcept
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
template < class V, typename = enable_if_t< concepts::vector_data_v<V> > >
#endif
class conjugate_vector
{
  public:
    //- Types

    /// @brief Input vector type
    using vector_type         = V;
  private:
    // Aliases
    using result_element_type = conditional_t< detail::is_complex_v<typename vector_type::element_type>,
                                               decltype( conj( declval<typename vector_type::element_type>() ) ),
                                               typename vector_type::element_type >;
    using result_vector_type  = typename vector_type::template rebind_t<result_element_type>;
    // Gets necessary arguments for constrution
    // If vector type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::fixed_size_vector_data_v<result_vector_type> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const vector_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_vector_data<result_vector_type>
    #endif
    { return tuple( forward<Lambda>( lambda ) ); }
    // If the vector type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default constructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::dynamic_vector_data_v<result_vector_type> >, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const vector_type& v, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_vector_data<vector_type>
    #endif
    {
      if constexpr ( is_default_constructible_v<typename result_vector_type::allocator_type> &&
                     allocator_traits<typename result_vector_type::allocator_type>::is_always_equal::value )
      {
        return tuple( v.size(), v.capacity(), forward<Lambda>( lambda ) );
      }
      else
      {
        using result_alloc_type = typename allocator_traits<typename vector_type::allocator_type>::template rebind_alloc<result_element_type>;
        return tuple( v.size(), v.capacity(), forward<Lambda>( lambda ), result_alloc_type( v.get_allocator() ) );
      }
    }
  public:
    //- Operations

    /// @brief Returns conjugate( vector )
    [[nodiscard]] static constexpr auto conjugate( const vector_type& v )
      noexcept( !detail::is_complex_v<typename vector_type::value_type> ||
                noexcept( detail::make_from_tuple< result_vector_type >(
                  collect_ctor_args( v,
                #ifndef LINALG_COMPILER_CLANG
                                     [&v]< class ... IndexType >( IndexType ... indices ) constexpr noexcept { return ::std::conj( detail::access( v, indices ... ) ); } ) ) ) )
                #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                                     []< class ... IndexType >( IndexType ... indices ) constexpr noexcept { return ::std::conj( typename vector_type::value_type() ); } ) ) ) )
                #endif
    {
      if constexpr ( detail::is_complex_v<typename vector_type::value_type> )
      {
        // Define conjugate transpose operation on each element
        auto conj_lambda = [&v]< class ... IndexType >( IndexType ... indices ) constexpr noexcept { return ::std::conj( detail::access( v, indices ... ) ); };
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
template < class V, class M, typename = enable_if_t< concepts::vector_data_v<V> && concepts::matrix_data_v<M> > >
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
    template < class typename = enable_if_t< concepts::dynamic_matrix_data_v< matrix_type > > >
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
    template < class typename = enable_if_t< !concepts::dynamic_matrix_data_v< matrix_type > >, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( [[maybe_unused]] const matrix_type& m ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( !concepts::dynamic_matrix_data< matrix_type > )
    #endif
    {
      return std::allocator<result_value_type>();
    }
    // Aliases
    using result_value_type       = decay_t< decltype( declval<typename vector_type::value_type>() * declval<typename matrix_type::value_type>() ) >;
    using pre_result_vector_type  = conditional_t< 
                                                   #ifdef LINALG_ENABLE_CONCEPTS
                                                   concepts::fixed_size_matrix_data<matrix_type>,
                                                   #else
                                                   concepts::fixed_size_matrix_data_v<matrix_type>,
                                                   #endif
                                                   fs_vector< result_value_type,
                                                              matrix_type::extents_type::static_extent(1),
                                                              detail::rebind_layout_t<typename vector_type::layout_type,
                                                                                      experimental::extents<typename vector_type::size_type,
                                                                                                            matrix_type::extents_type::static_extent(1)> >,
                                                              detail::rebind_accessor_t<typename vector_type::accessor_type,result_value_type> >,
                                                   dr_vector< result_value_type,
                                                              typename allocator_traits< decay_t< decltype( get_allocator( declval<matrix_type>() ) ) > >::template rebind_alloc<result_value_type>,
                                                              detail::rebind_layout_t<typename vector_type::layout_type,
                                                                                      experimental::extents<typename matrix_type::size_type,
                                                                                                            matrix_type::extents_type::static_extent(1)> >,
                                                              detail::rebind_accessor_t<typename vector_type::accessor_type,result_value_type> > >;
    using post_result_vector_type = conditional_t< 
                                                   #ifdef LINALG_ENABLE_CONCEPTS
                                                   concepts::fixed_size_matrix_data<matrix_type>,
                                                   #else
                                                   concepts::fixed_size_matrix_data_v<matrix_type>,
                                                   #endif
                                                   fs_vector< result_value_type,
                                                              matrix_type::extents_type::static_extent(0),
                                                              detail::rebind_layout_t<typename vector_type::layout_type,
                                                                                      experimental::extents<typename matrix_type::size_type,
                                                                                                            matrix_type::extents_type::static_extent(0) > >,
                                                              detail::rebind_accessor_t<typename vector_type::accessor_type,result_value_type> >,
                                                   dr_vector< result_value_type,
                                                              typename allocator_traits< decay_t< decltype( get_allocator( declval<matrix_type>() ) ) > >::template rebind_alloc<result_value_type>,
                                                              detail::rebind_layout_t<typename vector_type::layout_type,
                                                                                      experimental::extents<typename matrix_type::size_type,
                                                                                                            matrix_type::extents_type::static_extent(0)> >,
                                                              detail::rebind_accessor_t<typename vector_type::accessor_type,result_value_type> > >;
    // Gets necessary arguments for construction
    // If vector type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::fixed_size_vector_data_v<pre_result_vector_type> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const vector_type&, [[maybe_unused]] const matrix_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_vector_data<pre_result_vector_type>
    #endif
    { return tuple( forward<Lambda>( lambda ) ); }
    // If the vector type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default cosntructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::dynamic_vector_data_v<pre_result_vector_type> >, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const vector_type&, const matrix_type& m, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_vector_data<pre_result_vector_type>
    #endif
    {
      using result_vector_type = pre_result_vector_type;
      if constexpr ( is_default_constructible_v<typename result_vector_type::allocator_type> &&
                     allocator_traits<typename result_vector_type::allocator_type>::is_always_equal::value )
      {
        using result_extents_type = typename result_vector_type::extents_type;
        return tuple( result_extents_type( m.size().extent(1) ),
                      result_extents_type( m.capacity().extent(1) ),
                      forward<Lambda>( lambda ) );
      }
      else
      {
        using result_extents_type = typename result_vector_type::extents_type;
        using result_alloc_type   = typename result_vector_type::allocator_type;
        return tuple( result_extents_type( m.size().extent(1) ),
                      result_extents_type( m.capacity().extent(1) ),
                      forward<Lambda>( lambda ),
                      result_alloc_type( m.get_allocator() ) );
      }
    }
    // Gets necessary arguments for constrution
    // If vector type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::fixed_size_vector_data_v<post_result_vector_type> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const matrix_type&, [[maybe_unused]] const vector_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_vector_data<post_result_vector_type>
    #endif
    { return tuple( forward<Lambda>( lambda ) ); }
    // If the vector type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default cosntructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::dynamic_vector_data_v<post_result_vector_type> >, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const matrix_type& m, [[maybe_unused]] const vector_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_vector_data<post_result_vector_type>
    #endif
    {
      using result_vector_type = post_result_vector_type;
      if constexpr ( is_default_constructible_v<typename result_vector_type::allocator_type> &&
                     allocator_traits<typename result_vector_type::allocator_type>::is_always_equal::value )
      {
        using result_extents_type = typename result_vector_type::extents_type;
        return tuple( result_extents_type( m.size().extent(0) ),
                      result_extents_type( m.capacity().extent(0) ),
                      forward<Lambda>( lambda ) );
      }
      else
      {
        using result_extents_type = typename result_vector_type::extents_type;
        using result_alloc_type   = typename result_vector_type::allocator_type;
        return tuple( result_extents_type( m.size().extent(0) ),
                      result_extents_type( m.capacity().extent(0) ),
                      forward<Lambda>( lambda ),
                      result_alloc_type( m.get_allocator() ) );
      }
    }
  public:
    //- Operations

    /// @brief Returns v * m
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename = enable_if_t< ( ( vector_type::extents_type::static_extent(0) == matrix_type::extents_type::static_extent(0) ) ||
                                         ( vector_type::extents_type::static_extent(0) == experimental::dynamic_extent ) ||
                                         ( matrix_type::extents_type::static_extent(0) == experimental::dynamic_extent ) ) > >
    #endif
    [[nodiscard]] static constexpr pre_result_vector_type prod( const vector_type& v, const matrix_type& m )
      noexcept( noexcept(
        detail::make_from_tuple< pre_result_vector_type >(
          collect_ctor_args( v,
                             m,
                             #ifndef LINALG_COMPILER_CLANG
                             [&v,&m]< class IndexType >( IndexType index ) constexpr noexcept
                             {
                               result_value_type result = 0;
                               detail::for_each( LINALG_EXECUTION_UNSEQ,
                                                 detail::faux_index_iterator<typename vector_type::index_type>( 0 ),
                                                 detail::faux_index_iterator<typename vector_type::index_type>( v.size().extent(0) ),
                                                 [ &v, &m, &index, &result ] ( typename vector_type::index_type index2 ) constexpr noexcept
                                                   { result += detail::access( v, index2 ) * detail::access( m, index2, index ); } );
                               return result; }
                             #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                             []< class IndexType >( IndexType index ) constexpr noexcept
                             { return result_value_type(); }
                             #endif
                             ) ) ) &&
                ( ( vector_type::extents_type::static_extent(0) != experimental::dynamic_extent ) &&
                  ( matrix_type::extents_type::static_extent(0) != experimental::dynamic_extent ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( ( vector_type::extents_type::static_extent(0) == matrix_type::extents_type::static_extent(0) ) ||
                  ( vector_type::extents_type::static_extent(0) == experimental::dynamic_extent ) ||
                  ( matrix_type::extents_type::static_extent(0) == experimental::dynamic_extent ) )
    #endif
    {
      // If the extents are dynamic, then check they are compatable
      if constexpr ( ( vector_type::extents_type::static_extent(0) == experimental::dynamic_extent ) ||
                     ( matrix_type::extents_type::static_extent(0) == experimental::dynamic_extent ) )
      {
        // If sizes are not compatable, then throw exception
        if ( v.size() != m.size().extent(0) ) [[unlikely]]
        {
          throw length_error( "Matrix and vector sizes are incompatable." );
        }
      }
      // Define product operation on each element pair
      auto lambda = [&v,&m]< class IndexType >( IndexType index ) constexpr noexcept
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
    template < typename = enable_if_t< ( ( vector_type::extents_type::static_extent(0) == matrix_type::extents_type::static_extent(0) ) &&
                                         ( ( vector_type::extents_type::static_extent(0) == matrix_type::extents_type::static_extent(1) ) ||
                                           ( vector_type::extents_type::static_extent(0) == experimental::dynamic_extent ) ) ) > >
    #endif
    [[nodiscard]] static constexpr vector_type& prod( vector_type& v, const matrix_type& m )
      noexcept( noexcept( v = move( v * m ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( ( vector_type::extents_type::static_extent(0) == matrix_type::extents_type::static_extent(0) ) &&
                 ( ( vector_type::extents_type::static_extent(0) == matrix_type::extents_type::static_extent(1) ) ||
                   ( vector_type::extents_type::static_extent(0) == experimental::dynamic_extent ) ) )
    #endif
    {
      return v = move( v * m );
    }
    /// @brief Returns m * v
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename = enable_if_t< ( ( matrix_type::extents_type::static_extent(1) == vector_type::extents_type::static_extent(0) ) ||
                                         ( matrix_type::extents_type::static_extent(1) == experimental::dynamic_extent ) ||
                                         ( vector_type::extents_type::static_extent(0) == experimental::dynamic_extent ) ) > >
    #endif
    [[nodiscard]] static constexpr post_result_vector_type prod( const matrix_type& m, const vector_type& v )
      noexcept( noexcept(
        detail::make_from_tuple< post_result_vector_type >(
          collect_ctor_args( declval<const vector_type&>(),
                             declval<const matrix_type&>(),
                             #ifndef LINALG_COMPILER_CLANG
                             [&v,&m]< class IndexType >( IndexType index ) constexpr noexcept
                             {
                               result_value_type result = 0;
                               detail::for_each( LINALG_EXECUTION_UNSEQ,
                                                 detail::faux_index_iterator<typename vector_type::index_type>( 0 ),
                                                 detail::faux_index_iterator<typename vector_type::index_type>( v.size().extent(0) ),
                                                 [ &v, &m, &index, &result ] ( typename vector_type::index_type index2 ) constexpr noexcept
                                                   { result += detail::access( m, index, index2 ) * detail::access( v, index2 ); } );
                               return result; }
                             #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                             []< class IndexType >( IndexType index ) constexpr noexcept
                             { return result_value_type(); }
                             #endif
                             ) ) ) &&
                ( ( matrix_type::extents_type::static_extent(1) != experimental::dynamic_extent ) &&
                  ( vector_type::extents_type::static_extent(0) != experimental::dynamic_extent ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( ( matrix_type::extents_type::static_extent(1) == vector_type::extents_type::static_extent(0) ) ||
                  ( matrix_type::extents_type::static_extent(1) == experimental::dynamic_extent ) ||
                  ( vector_type::extents_type::static_extent(0) == experimental::dynamic_extent ) )
    #endif
    {
      // If the extents are dynamic, then check they are compatable
      if constexpr ( ( matrix_type::extents_type::static_extent(1) == experimental::dynamic_extent ) ||
                     ( vector_type::extents_type::static_extent(0) == experimental::dynamic_extent ) )
      {
        // If sizes are not compatable, then throw exception
        if ( v.size() != m.size().extent(1) ) [[unlikely]]
        {
          throw length_error( "Matrix and vector sizes are incompatable." );
        }
      }
      // Define product operation on each element pair
      auto lambda = [&v,&m]< class IndexType >( IndexType index ) constexpr noexcept
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
template < class M1, class M2, typename = enable_if_t< concepts::matrix_data_v<M1> && concepts::matrix_data_v<M2> > >
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
    template < typename = enable_if_t< concepts::dynamic_matrix_data_v< first_matrix_type > > >
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
    template < typename = enable_if_t< concepts::dynamic_matrix_data_v< second_matrix_type > && !concepts::dynamic_matrix_data_v< first_matrix_type > >, typename = enable_if_t<true> >
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
    template < typename = enable_if_t< !concepts::dynamic_matrix_data_v< first_matrix_type > && !concepts::dynamic_matrix_data_v< second_matrix_type > >, typename = enable_if_t<true>, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( [[maybe_unused]] const first_matrix_type& m1, [[maybe_unused]] const second_matrix_type& m2 ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( !concepts::dynamic_matrix_data< first_matrix_type > && !concepts::dynamic_matrix_data< second_matrix_type > )
    #endif
    {
      return std::allocator<result_value_type>();
    }
    // Aliases
    using result_value_type  = decay_t< decltype( declval<typename first_matrix_type::value_type>() * declval<typename second_matrix_type::value_type>() ) >;
    using result_matrix_type = conditional_t< 
                                              #ifdef LINALG_ENABLE_CONCEPTS
                                              concepts::fixed_size_matrix_data<first_matrix_type> && concepts::fixed_size_matrix_data<second_matrix_type>,
                                              #else
                                              concepts::fixed_size_matrix_data_v<first_matrix_type> && concepts::fixed_size_matrix_data_v<second_matrix_type>,
                                              #endif
                                              fs_matrix< result_value_type,
                                                         first_matrix_type::extents_type::static_extent(0),
                                                         second_matrix_type::extents_type::static_extent(1),
                                                         detail::rebind_layout_t<typename first_matrix_type::layout_type,
                                                                                 experimental::extents<typename first_matrix_type::size_type,
                                                                                                       first_matrix_type::extents_type::static_extent(0),
                                                                                                       second_matrix_type::extents_type::static_extent(1)> >,
                                                         detail::rebind_accessor_t<typename first_matrix_type::accessor_type,result_value_type> >,
                                              dr_matrix< result_value_type,
                                                         typename allocator_traits< decay_t< decltype( get_allocator( declval<first_matrix_type>(), declval<second_matrix_type>() ) ) > >::template rebind_alloc<result_value_type>,
                                                         detail::rebind_layout_t<typename first_matrix_type::layout_type,
                                                                                 experimental::extents<typename first_matrix_type::index_type,
                                                                                                       experimental::dynamic_extent,
                                                                                                       experimental::dynamic_extent> >,
                                                         detail::rebind_accessor_t<typename first_matrix_type::accessor_type,result_value_type> > >;
    // Gets necessary arguments for constrution
    // If matrix type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::fixed_size_matrix_data_v<result_matrix_type> > >
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
    template < class Lambda, typename = enable_if_t< concepts::dynamic_matrix_data_v<result_matrix_type> >, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const first_matrix_type& m1, const second_matrix_type& m2, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_matrix_data<result_matrix_type>
    #endif
    {
      if constexpr ( is_default_constructible_v<typename result_matrix_type::allocator_type> &&
                     allocator_traits<typename result_matrix_type::allocator_type>::is_always_equal::value )
      {
        using result_extents_type = typename result_matrix_type::extents_type;
        return tuple( result_extents_type( m1.size().extent(0), m2.size().extent(1) ),
                      result_extents_type( m1.capacity().extent(0), m2.capacity().extent(1) ),
                      forward<Lambda>( lambda ) );
      }
      else
      {
        using result_extents_type = typename result_matrix_type::extents_type;
        using result_alloc_type   = typename result_matrix_type::allocator_type;
        return tuple( result_extents_type( m1.size().extent(0), m2.size().extent(1) ),
                      result_extents_type( m1.capacity().extent(0), m2.capacity().extent(1) ),
                      forward<Lambda>( lambda ),
                      result_alloc_type( get_allocator( m1, m2 ) ) );
      }
    }
  public:
    //- Operations

    /// @brief computes m1 * m2
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename = enable_if_t< ( ( first_matrix_type::extents_type::static_extent(1) == second_matrix_type::extents_type::static_extent(0) ) ||
                                         ( first_matrix_type::extents_type::static_extent(1) == experimental::dynamic_extent ) ||
                                         ( second_matrix_type::extents_type::static_extent(0) == experimental::dynamic_extent ) ) > >
    #endif
    [[nodiscard]] static constexpr result_matrix_type prod( const first_matrix_type& m1, const second_matrix_type& m2 )
      noexcept( noexcept( detail::make_from_tuple< result_matrix_type >(
        collect_ctor_args( declval<const first_matrix_type&>(),
                           declval<const second_matrix_type&>(),
                           #ifndef LINALG_COMPILER_CLANG
                           [&m1,&m2]< class IndexType1, class IndexType2 >( IndexType1 index1, IndexType2 index2 ) constexpr noexcept
                           {
                             result_value_type result = 0;
                             detail::for_each( LINALG_EXECUTION_UNSEQ,
                                               detail::faux_index_iterator<typename first_matrix_type::index_type>( 0 ),
                                               detail::faux_index_iterator<typename first_matrix_type::index_type>( m1.size().extent(1) ),
                                               [ &m1, &m2, &index1, &index2, &result ] ( typename first_matrix_type::index_type index ) constexpr noexcept
                                                 { result += detail::access( m1, index1, index ) * detail::access( m2, index, index2 ); } );
                             return result; }
                          #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                           []< class IndexType1, class IndexType2 >( IndexType1 index1, IndexType2 index2 ) constexpr noexcept
                           { return result_value_type(); }
                          #endif
                           ) ) ) &&
                ( first_matrix_type::extents_type::static_extent(1) != experimental::dynamic_extent ) &&
                ( second_matrix_type::extents_type::static_extent(0) != experimental::dynamic_extent ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( ( first_matrix_type::extents_type::static_extent(1) == second_matrix_type::extents_type::static_extent(0) ) ||
                 ( first_matrix_type::extents_type::static_extent(1) == experimental::dynamic_extent ) ||
                 ( second_matrix_type::extents_type::static_extent(0) == experimental::dynamic_extent ) )
    #endif
    {
      // If the extents are dynamic, then check they are compatable
      if constexpr ( ( first_matrix_type::extents_type::static_extent(1) == experimental::dynamic_extent ) ||
                     ( second_matrix_type::extents_type::static_extent(0) == experimental::dynamic_extent ) )
      {
        // If sizes are not compatable, then throw exception
        if ( m1.size().extent(1) != m2.size().extent(0) ) [[unlikely]]
        {
          throw length_error( "Matrix sizes are incompatable." );
        }
      }
      // Define product operation on each element pair
      auto lambda = [&m1,&m2]< class IndexType1, class IndexType2 >( IndexType1 index1, IndexType2 index2 ) constexpr noexcept
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
    template < typename = enable_if_t< ( ( first_matrix_type::extents_type::static_extent(1) == second_matrix_type::extents_type::static_extent(0) ) &&
                                         ( ( first_matrix_type::extents_type::static_extent(1) == second_matrix_type::extents_type::static_extent(1) ) ||
                                           ( first_matrix_type::extents_type::static_extent(1) == experimental::dynamic_extent ) ) ) > >
    #endif
    [[nodiscard]] static constexpr first_matrix_type& prod( first_matrix_type& m1, const second_matrix_type& m2 )
      noexcept( noexcept( m1 = move( m1 * m2 ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( ( first_matrix_type::extents_type::static_extent(1) == second_matrix_type::extents_type::static_extent(0) ) &&
                 ( ( first_matrix_type::extents_type::static_extent(1) == second_matrix_type::extents_type::static_extent(1) ) ||
                   ( first_matrix_type::extents_type::static_extent(1) == experimental::dynamic_extent ) ) )
    #endif
    {
      return m1 = move( m1 * m2 );
    }
};

/// @brief Defines inner product operation on a pair of vectors
/// @tparam V1 vector
/// @tparam V2 vector
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::vector_data V1, concepts::vector_data V2 >
#else
template < class V1, class V2, enable_if_t< concepts::vector_data_v< V2 > > >
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
    using result_type        = decltype( auto( declval<typename first_vector_type::value_type>() * declval<typename second_vector_type::value_type>() ) );
  public:
    //- Operations

    /// @brief computes the inner product of v1 and v2
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename = enable_if_t< detail::extents_may_be_equal_v<typename first_vector_type::extents_type,typename second_vector_type::extents_type> > >
    #endif
    [[nodiscard]] static constexpr auto prod( const first_vector_type& v1, const second_vector_type& v2 )
      noexcept( detail::extents_is_equal_v<typename first_vector_type::extents_type,typename second_vector_type::extents_type> )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires detail::extents_may_be_equal_v<typename first_vector_type::extents_type,typename second_vector_type::extents_type>
    #endif
    {
      if constexpr ( !detail::extents_is_equal_v<typename first_vector_type::extents_type,typename second_vector_type::extents_type> )
      {
        // Check if sizes are equal
        if ( !( v1.size() == v2.size() ) ) [[unlikely]]
        {
          throw length_error( "Vectors must have same size." );
        }
      }
      // Store sum of inner product
      result_type result = 0;
      // Define lambda function to sum inner product
      auto inner_prod_lambda = [&v1,&v2,&result]< class ... IndexType >( IndexType ... indices ) constexpr noexcept
        { result += detail::access( v1, indices ... ) * detail::access( v2, indices ... ); };
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
template < class V1, class V2, typename = enable_if_t< concepts::vector_data_v<V1> && concepts::vector_data_v<V2> > >
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
    template < typename = enable_if_t< concepts::dynamic_matrix_data_v< first_vector_type > > >
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
    template < typename = enable_if_t< concepts::dynamic_vector_data_v< second_vector_type > && !concepts::dynamic_vector_data_v< first_vector_type > >, typename = enable_if_t<true> >
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
    template < typename = enable_if_t< !concepts::dynamic_vector_data_v< first_vector_type > && !concepts::dynamic_vector_data_v< second_vector_type > >, typename = enable_if_t<true>, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) get_allocator( [[maybe_unused]] const first_vector_type& v1, [[maybe_unused]] const second_vector_type& v2 ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( !concepts::dynamic_vector_data< first_vector_type > && !concepts::dynamic_vector_data< second_vector_type > )
    #endif
    {
      return std::allocator<result_value_type>();
    }
    // Aliases
    using result_value_type  = decay_t< decltype( declval<typename first_vector_type::value_type>() * declval<typename second_vector_type::value_type>() ) >;
    using result_matrix_type = conditional_t< 
                                              #ifdef LINALG_ENABLE_CONCEPTS
                                              concepts::fixed_size_vector_data<first_vector_type> && concepts::fixed_size_vector_data<second_vector_type>,
                                              #else
                                              concepts::fixed_size_vector_data_v<first_vector_type> && concepts::fixed_size_vector_data_v<second_vector_type>,
                                              #endif
                                              fs_matrix< result_value_type,
                                                         first_vector_type::extents_type::static_extent(0),
                                                         second_vector_type::extents_type::static_extent(0),
                                                         detail::rebind_layout_t<typename first_vector_type::layout_type,
                                                                                 experimental::extents<typename first_vector_type::size_type,
                                                                                                       first_vector_type::extents_type::static_extent(0),
                                                                                                       second_vector_type::extents_type::static_extent(0)> >,
                                                         detail::rebind_accessor_t<typename first_vector_type::accessor_type,result_value_type> >,
                                              dr_matrix< result_value_type,
                                                         typename allocator_traits< decay_t< decltype( get_allocator( declval<first_vector_type>(), declval<second_vector_type>() ) ) > >::template rebind_alloc<result_value_type>,
                                                         detail::rebind_layout_t<typename first_vector_type::layout_type,
                                                                                 experimental::extents<typename first_vector_type::size_type,
                                                                                                       experimental::dynamic_extent,
                                                                                                       experimental::dynamic_extent> >,
                                                         detail::rebind_accessor_t<typename first_vector_type::accessor_type,result_value_type> > >;
    // Gets necessary arguments for constrution
    // If matrix type is fixed size, then the lambda expression is the only argument needed
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::fixed_size_matrix_data_v<result_matrix_type> > >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( [[maybe_unused]] const first_vector_type&, [[maybe_unused]] const second_vector_type&, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::fixed_size_matrix_data<result_matrix_type>
    #endif
    { return tuple( forward<Lambda>( lambda ) ); }
    // If the matrix type is dynamic, then size and capacity must be provided along with the lambda expression.
    // Additionally, if all allocators of the desired type are not the same or cannot be default cosntructed, then it must be
    // passed along as well.
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda, typename = enable_if_t< concepts::dynamic_matrix_data_v<result_matrix_type> >, typename = enable_if_t<true> >
    #endif
    [[nodiscard]] static inline constexpr decltype(auto) collect_ctor_args( const first_vector_type& v1, const second_vector_type& v2, Lambda&& lambda ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires concepts::dynamic_matrix_data<result_matrix_type>
    #endif
    {
      if constexpr ( is_default_constructible_v<typename result_matrix_type::allocator_type> &&
                     allocator_traits<typename result_matrix_type::allocator_type>::is_always_equal::value )
      {
        using result_extents_type = typename result_matrix_type::extents_type;
        return tuple( result_extents_type( v1.size(), v2.size() ),
                      result_extents_type( v1.capacity(), v2.capacity() ),
                      forward<Lambda>( lambda ) );
      }
      else
      {
        using result_extents_type = typename result_matrix_type::extents_type;
        using result_alloc_type   = typename result_matrix_type::allocator_type;
        return tuple( result_extents_type( v1.size(), v2.size() ),
                      result_extents_type( v1.capacity(), v2.capacity() ),
                      forward<Lambda>( lambda ),
                      result_alloc_type( get_allocator(v1,v2) ) );
      }
    }
  public:
    /// @brief Computes the outer product of v1 and v2
    [[nodiscard]] static constexpr result_matrix_type prod( const first_vector_type& v1, const second_vector_type& v2 )
      noexcept( noexcept( detail::make_from_tuple< result_matrix_type >(
        collect_ctor_args( v1,
                           v2,
                           #ifndef LINALG_COMPILER_CLANG
                           [&v1,&v2]< class IndexType1, class IndexType2 >( IndexType1 index1, IndexType2 index2 ) constexpr noexcept
                             { return detail::access( v1, index1 ) * detail::access( v2, index2 ); } ) ) ) )
                           #else // Clang does not allow use of input variables in lambda expression inside noexcept specification
                           []< class IndexType1, class IndexType2 >( IndexType1 index1, IndexType2 index2 ) constexpr noexcept
                             { return typename result_matrix_type::value_type(); } ) ) ) )
                           #endif
    {
      // Define product operation on each element pair
      auto lambda = [&v1,&v2]< class IndexType1, class IndexType2 >( IndexType1 index1, IndexType2 index2 ) constexpr noexcept
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
