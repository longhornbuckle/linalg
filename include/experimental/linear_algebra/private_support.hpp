//==================================================================================================
//  File:       private_support.hpp
//
//  Summary:    This header defines several private traits types, alias templates, variable
//              templates, and functions that support the rest of this implementation.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_IMPL_SUPPORT_HPP
#define LINEAR_ALGEBRA_IMPL_SUPPORT_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{
namespace detail
{

//==================================================================================================
//  Helper class for encapsulating macro dependent multidimensional access
//==================================================================================================
template < class MultiDimOperClass, class ... IndexType >
[[nodiscard]] LINALG_FORCE_INLINE_FUNCTION constexpr decltype(auto) access( MultiDimOperClass&& t, IndexType ... indices )
  noexcept( noexcept( 
    #if LINALG_USE_BRACKET_OPERATOR
    t[ indices ... ]
    #else
    t( indices ... )
    #endif
   ) )
{
  #if LINALG_USE_BRACKET_OPERATOR
  return t[ indices ... ];
  #else
  return t( indices ... );
  #endif
}

//==================================================================================================
// Forwards to std for_each with or without execution policy depending on configuration
//==================================================================================================
template< class ExecutionPolicy, class ForwardIt, class UnaryFunction2 >
constexpr LINALG_FORCE_INLINE_FUNCTION void
for_each( [[maybe_unused]] ExecutionPolicy&& policy,
          ForwardIt                          first,
          ForwardIt                          last,
          UnaryFunction2                     f )
{
  #if LINALG_EXECTUION_POLICY
  ::std::for_each( policy, first, last, f );
  #else
  ::std::for_each( first, last, f );
  #endif
}

//==================================================================================================
// is_nothrow_convertible_v implementation (for C++17)
//==================================================================================================
#ifdef LINALG_NO_THROW_CONVERTIBLE
template < class From, class To >
struct is_nothrow_convertible : public ::std::is_nothrow_convertible< From, To > { };
#else
template < class From, class To, typename = void >
struct is_nothrow_convertible : public ::std::conjunction< ::std::is_void< From >, ::std::is_void< To > > { };
 
template < class From, class To >
struct is_nothrow_convertible< From, To, ::std::enable_if< ::std::is_same_v< decltype( ::std::declval<void(&)(To) noexcept>()( ::std::declval<From>() ) ), decltype( ::std::declval<void(&)(To) noexcept>()( ::std::declval<From>() ) ) > > > : public ::std::true_type { };
#endif
template < class From, class To >
inline constexpr bool is_nothrow_convertible_v = is_nothrow_convertible< From, To >::value;

//==================================================================================================
//  Test if type is an mdspan
//==================================================================================================
template < class T >
struct is_mdspan : public ::std::false_type {};

template < class T, class Extents, class LayoutPolicy, class AccessorPolicy >
struct is_mdspan< ::std::experimental::mdspan<T,Extents,LayoutPolicy,AccessorPolicy> > : public ::std::true_type {};
/// @brief True iff T is an mdspan
/// @tparam T 
template < class T >
inline constexpr bool is_mdspan_v = is_mdspan<T>::value;

//==================================================================================================
//  Test if type is an extents
//==================================================================================================
template < class T >
struct is_extents : public ::std::false_type {};

template < class size_type, ::std::size_t ... Extents >
struct is_extents< ::std::experimental::extents<size_type,Extents...> > : public ::std::true_type {};
/// @brief True iff T is an extents
/// @tparam T 
template < class T >
inline constexpr bool is_extents_v = is_extents<T>::value;

//==================================================================================================
//  Test if type is a tuple of all same type
//==================================================================================================
template < class T >
struct is_homogeneous_tuple : public ::std::false_type {};

template < class... size_type >
struct is_homogeneous_tuple< ::std::tuple<size_type...> > : public ::std::true_type {};
/// @brief True iff T is a tuple of the same types
/// @tparam T 
template < class T >
inline constexpr bool is_homogeneous_tuple_v = is_homogeneous_tuple<T>::value;

//==================================================================================================
//  Test if type extents is not dynamic
//==================================================================================================
template < class T >
struct extents_is_static : public ::std::false_type {};

template < class SizeType, ::std::size_t ... Extents >
struct extents_is_static< ::std::experimental::extents<SizeType,Extents...> > : public
  ::std::conditional_t< ( ( Extents != dynamic_extent ) && ... ), ::std::true_type, ::std::false_type > {};

template < class T >
inline constexpr bool extents_is_static_v = extents_is_static<T>::value;

//==================================================================================================
//  Test if type extents may be equal
//==================================================================================================

template < class T, class U, class Seq >
struct extents_may_be_equal_impl;

template < class SizeType, class OtherSizeType, auto ... Extents, auto ... OtherExtents, auto ... Indices >
struct extents_may_be_equal_impl< ::std::experimental::extents<SizeType,Extents...>,
                                  ::std::experimental::extents<OtherSizeType,OtherExtents...>,
                                  ::std::index_sequence<Indices...> > :
  ::std::conditional_t< ( ( ( ::std::experimental::extents<SizeType,Extents...>::static_extent(Indices)      == ::std::experimental::extents<SizeType,OtherExtents...>::static_extent(Indices) ) ||
                            ( ::std::experimental::extents<SizeType,Extents...>::static_extent(Indices)      == ::std::experimental::dynamic_extent ) ||
                            ( ::std::experimental::extents<SizeType,OtherExtents...>::static_extent(Indices) == ::std::experimental::dynamic_extent ) ) && ... ),
                        ::std::true_type, ::std::false_type > {};

template < class T, class U >
struct extents_may_be_equal : public ::std::false_type {};

template < class SizeType, class OtherSizeType, auto ... Extents, auto ... OtherExtents >
struct extents_may_be_equal< ::std::experimental::extents<SizeType,Extents...>,
                             ::std::experimental::extents<OtherSizeType,OtherExtents...> > : public
  ::std::conditional_t< ( sizeof...(Extents) == sizeof...(OtherExtents) ),
                        extents_may_be_equal_impl< ::std::experimental::extents<SizeType,Extents...>, ::std::experimental::extents<OtherSizeType,OtherExtents...>, ::std::make_index_sequence<sizeof...(Extents)> >,
                        ::std::false_type > {};

template < class T, class U >
inline constexpr bool extents_may_be_equal_v = extents_may_be_equal<T,U>::value;

//==================================================================================================
//  Test if type extents is equal
//==================================================================================================
template < class T, class U >
struct extents_are_equal : public
  ::std::conditional_t< extents_may_be_equal_v<T,U> && extents_is_static_v<T> && extents_is_static_v<U>,
                        ::std::true_type,
                        ::std::false_type > {};

template < class T, class U >
inline constexpr bool extents_are_equal_v = extents_are_equal<T,U>::value;

//==================================================================================================
//  Returns the number of dimensions with non-dynamic extent greater than one
//==================================================================================================
template < class T >
struct nondynamic_rank;

template < template < class, ::std::size_t ... > class E, class T, ::std::size_t ... Extents >
struct nondynamic_rank< E<T,Extents...> > : public ::std::integral_constant< ::std::size_t, ( ( ( Extents > 1 ) && ( Extents != ::std::experimental::dynamic_extent ) ) + ... ) > {};

template < class E, typename = ::std::enable_if_t< is_extents_v<E> > >
inline constexpr ::std::size_t nondynamic_rank_v = nondynamic_rank<E>::value;

//==================================================================================================
//  Returns the number of dimensions for which the extent is greater than one
//==================================================================================================
template < class SizeType, ::std::size_t ... Extents, ::std::size_t ... Ints >
[[nodiscard]] inline constexpr ::std::size_t current_rank_impl( const ::std::experimental::extents<SizeType,Extents...>& extents,
                                                                [[maybe_unused]] ::std::index_sequence<Ints...> ) noexcept
{
  return ( ( extents.extent(Ints) > 1 ) + ... );
}
template < class SizeType, ::std::size_t ... Extents >
[[nodiscard]] inline constexpr ::std::size_t current_rank( const ::std::experimental::extents<SizeType,Extents...>& extents ) noexcept
{
  return current_rank_impl( extents, ::std::make_index_sequence<sizeof...(Extents)>() );
}

//==================================================================================================
//  Test if expression is consteval
//==================================================================================================
/// @brief True iff the lambda expression may be evaluated as a constant expression
template < class Lambda, int = ( Lambda{}(), 0 ) >
constexpr bool is_constexpr(Lambda) { return true; }
constexpr bool is_constexpr(...) { return false; }

//==================================================================================================
//  Constexpr Product
//==================================================================================================
template < class T, class ... Ts >
[[nodiscard]] constexpr auto product( T t, Ts ... ts )
{
  if constexpr ( sizeof...(Ts) == 0 )
  {
    return t;
  }
  else
  {
    return t * product( ts ... );
  }
}
//==================================================================================================
//  Faux Index Iterator allows indices to be used in std algorithms which take iterators
//==================================================================================================
template < class T >
struct faux_index_iterator
{
  using value_type        = T;
  using difference_type   = ::std::ptrdiff_t;
  using reference         = T&;
  using pointer           = T*;
  using iterator_category = ::std::random_access_iterator_tag;
  constexpr faux_index_iterator( T t ) noexcept : index(t) { }
  constexpr faux_index_iterator&              operator++  ()                                 noexcept { static_cast<void>(this->index++); return *this; }
  constexpr faux_index_iterator&              operator--  ()                                 noexcept { static_cast<void>(this->index--); return *this; }
  constexpr faux_index_iterator&              operator+=  ( const difference_type n )        noexcept { static_cast<void>(this->index+=n); return *this; }
  constexpr faux_index_iterator&              operator-=  ( const difference_type n )        noexcept { static_cast<void>(this->index-=n); return *this; }
  [[nodiscard]] constexpr faux_index_iterator operator+   ( const difference_type n )        noexcept { return faux_index_iterator( this->index + n ); }
  [[nodiscard]] constexpr faux_index_iterator operator-   ( const difference_type n )        noexcept { return faux_index_iterator( this->index - n ); }
  [[nodiscard]] constexpr difference_type     operator -  ( const faux_index_iterator& rhs ) noexcept { return difference_type( this->index - rhs.index ); }
  [[nodiscard]] constexpr reference           operator *  ()                                 noexcept { return this->index; }
  [[nodiscard]] constexpr pointer             operator -> ()                                 noexcept { return &(this->index); }
  [[nodiscard]] constexpr bool                operator == ( const faux_index_iterator& rhs ) noexcept { return ( this->index == rhs.index ); }
  [[nodiscard]] constexpr bool                operator >  ( const faux_index_iterator& rhs ) noexcept { return ( this->index > rhs.index ); }
  [[nodiscard]] constexpr bool                operator <  ( const faux_index_iterator& rhs ) noexcept { return ( this->index < rhs.index ); }
  [[nodiscard]] constexpr bool                operator != ( const faux_index_iterator& rhs ) noexcept { return !( *this == rhs ); }
  [[nodiscard]] constexpr bool                operator >= ( const faux_index_iterator& rhs ) noexcept { return !( *this > rhs ); }
  [[nodiscard]] constexpr bool                operator <= ( const faux_index_iterator& rhs ) noexcept { return !( *this < rhs ); }
  [[nodiscard]] constexpr value_type          operator[]  ( const difference_type n )        noexcept { return this->index + n; }
  T index;
};

//==================================================================================================
//  Constexpr For
//==================================================================================================
template < auto Start, auto End, auto Inc, class F >
constexpr void constexpr_for( F&& f )
{
  if constexpr ( Start < End )
  {
    f( ::std::integral_constant< decltype(Start), Start >() );
    constexpr_for< Start + Inc, End, Inc >( f );
  }
}

//==================================================================================================
//  Has Index Operator concept is met if lambda has operator()( Indices ... ) defined
//==================================================================================================
template < class Lambda, class Enable, class IndexType, IndexType ... Indices >
struct has_index_operator_impl : public ::std::false_type { };
template < class Lambda, class IndexType, IndexType ... Indices >
struct has_index_operator_impl< Lambda, ::std::enable_if_t< ::std::is_same_v< decltype( ::std::declval<Lambda&&>()( Indices ... ) ), decltype( ::std::declval<Lambda&&>()( Indices ... ) ) >, void >, IndexType, Indices ... > : public ::std::true_type { };
template < class Lambda, class IndexType, IndexType ... Indices >
struct has_index_operator : public has_index_operator_impl< Lambda, void, IndexType, Indices ... > { };
template < class Lambda, class IndexType, IndexType ... Indices >
inline constexpr bool has_index_operator_v = has_index_operator< Lambda, IndexType, Indices ... >::value;

//==================================================================================================
//  Is Defined is true if the class has been defined
//==================================================================================================
template < class ... Ts > struct make_void { typedef void type;};

template < class T, class Enabler = void >
struct is_defined : ::std::false_type {};

template < class T >
struct is_defined< T, typename make_void< decltype( sizeof(T) != 0 ) >::type > : ::std::true_type {};

template < class T >
inline constexpr bool is_defined_v = is_defined<T>::value;

//==================================================================================================
//  Is Unsequenced returns true if the Execution Policy is unsequence
//==================================================================================================
template < class T >
struct is_unsequenced : public ::std::false_type { };

#if LINALG_EXECUTION_POLICY

template < >
struct is_unsequenced< ::std::execution::parallel_unsequenced_policy > : public ::std::true_type { };

#if ! is_same_v< decltype( LINALG_EXECUTION_UNSEQ ), execution::sequenced_policy >

template < >
struct is_unsequenced< ::std::execution::unsequenced_policy > : public ::std::true_type { };

#endif

#endif

template < class T >
inline constexpr bool is_unsequenced_v = is_unsequenced<T>::value;

//==================================================================================================
//  Apply All applies the lambda expression to all elements in the view
//==================================================================================================

template < class > struct apply_all_strided_helper { };
template < class IndexType, IndexType ... Indices >
struct apply_all_strided_helper< ::std::integer_sequence<IndexType,Indices...> >
{
  [[nodiscard]] static constexpr auto zero( [[maybe_unused]] IndexType ) { return 0; }
  [[nodiscard]] static constexpr auto initial_indices() noexcept { return ::std::tuple( zero(Indices) ... ); }
};

template < class View >
struct stride_order;

template < class ElementType, class Extents, class AccessorPolicy >
struct stride_order< ::std::experimental::mdspan< ElementType, Extents, ::std::experimental::layout_left, AccessorPolicy > >
{
  template < class IndexType >
  [[nodiscard]] static constexpr auto get_nth_largest_stride_index( IndexType index ) noexcept
  {
    return ::std::experimental::mdspan< ElementType, Extents, ::std::experimental::layout_left, AccessorPolicy >::rank() - index - 1;
  }
};

template < class ElementType, class Extents, class AccessorPolicy >
struct stride_order< ::std::experimental::mdspan< ElementType, Extents, ::std::experimental::layout_right, AccessorPolicy > >
{
  template < class IndexType >
  [[nodiscard]] static constexpr auto get_nth_largest_stride_index( IndexType index ) noexcept
  {
    return index;
  }
};

// DOESN'T WORK. ORDER MUST BE DETERMINED AT COMPILE TIME
//
// template < class ElementType, class Extents, class AccessorPolicy >
// struct stride_order< ::std::experimental::mdspan< ElementType, Extents, ::std::experimental::layout_stride, AccessorPolicy > >
// {
//   constexpr stride_order( const ::std::experimental::layout_stride::mapping<Extents>& layout ) noexcept :
//     order_( layout.strides() )
//   {
//     sort( this->order_.begin(),
//           this->order_.end(),
//           [layout]( auto index1, auto index2 ) { return layout.stride(index1) <= layout.stride(index2); } );
//   }

//   template < class IndexType >
//   [[nodiscard]] constexpr auto get_nth_largest_stride_index( IndexType index ) noexcept
//   {
//     return this->order_[index];
//   }
// private :
//   decltype( declval< ::std::experimental::layout_stride::mapping<Extents> >().strides() ) order_;
// };

template < class         View,
           class         Lambda,
           class         ExecutionPolicy,
           class         IndexType,
           IndexType ... BeforeIndices,
           IndexType ... AfterIndices,
           class ...     IndicesType >
inline void apply_all_strided3_except( View&&                 view,
                                       Lambda&&               lambda,
                                       ExecutionPolicy&&      execution_policy,
                                       [[maybe_unused]] ::std::integer_sequence<IndexType,BeforeIndices...>,
                                       [[maybe_unused]] ::std::integer_sequence<IndexType,AfterIndices...>,
                                       tuple<IndicesType...>& indices )
{
  constexpr IndexType index = sizeof...(BeforeIndices);
  // Cache the last exception to be thrown
  ::std::exception_ptr eptr;
  // Attempt lambda expression on each element
  ::std::math::detail::
  for_each( execution_policy,
            faux_index_iterator<::std::decay_t<decltype( ::std::get<index>(indices) )> >(0),
            faux_index_iterator<::std::decay_t<decltype( ::std::get<index>(indices) )> >(view.extent(index)),
            [&lambda,&indices,&eptr] ( ::std::decay_t<decltype( ::std::get<index>(indices) )> curr_index ) noexcept
              {
                try { lambda( ::std::get<BeforeIndices>(indices) ..., curr_index, ::std::get<AfterIndices+index>(indices) ... ); }
                catch ( ... ) { eptr = ::std::current_exception(); }
              } );
  // If exceptions were thrown, rethrow the last
  if ( eptr ) LINALG_UNLIKELY
  {
    ::std::rethrow_exception( eptr );
  }
}

template < class         View,
           class         Lambda,
           class         ExecutionPolicy,
           class         IndexType,
           IndexType ... BeforeIndices,
           IndexType ... AfterIndices,
           class ...     IndicesType >
constexpr void apply_all_strided3( View&&                                                               view,
                                   Lambda&&                                                             lambda,
                                   ExecutionPolicy&&                                                    execution_policy,
                                   [[maybe_unused]] ::std::integer_sequence<IndexType,BeforeIndices...> before,
                                   [[maybe_unused]] ::std::integer_sequence<IndexType,AfterIndices...>  after,
                                   ::std::tuple<IndicesType...>&                                        indices )
  noexcept( noexcept( lambda( ::std::get< BeforeIndices >( indices ) ...,
                              ::std::declval< ::std::decay_t< decltype( get< sizeof...(BeforeIndices) >( indices ) ) > >(),
                              ::std::get< AfterIndices + sizeof...(BeforeIndices) >( indices ) ... ) ) )
{
  constexpr IndexType index = sizeof...(BeforeIndices);
  // If lambda expression is noexcept, then just attempt to call using whatever execution policy
  constexpr bool is_noexcept = noexcept( lambda( ::std::get< BeforeIndices >( indices ) ...,
                                                 ::std::declval< decay_t< decltype( ::std::get< sizeof...(BeforeIndices) >( indices ) ) > >(),
                                                 ::std::get< AfterIndices + sizeof...(BeforeIndices) >( indices ) ... ) );
  if constexpr ( is_noexcept )
  {
    ::std::math::detail::
    for_each( execution_policy,
              faux_index_iterator<::std::decay_t<decltype( ::std::get<index>(indices) )> >(0),
              faux_index_iterator<::std::decay_t<decltype( ::std::get<index>(indices) )> >(view.extent(index)),
              [&lambda,&indices] ( ::std::decay_t<decltype( ::std::get<index>(indices) )> curr_index ) constexpr noexcept
                {
                  lambda( ::std::get<BeforeIndices>(indices) ..., curr_index, ::std::get<AfterIndices+index>(indices) ... );
                } );
  }
  else
  {
    apply_all_strided3_except( view, lambda, execution_policy, before, after, indices );
  }
}

template < ::std::size_t Index,
           class         View,
           class         Lambda,
           class         ExecutionPolicy,
           class ...     IndexType >
[[nodiscard]] constexpr bool apply_all_strided2_is_noexcept() noexcept
{
  constexpr ::std::size_t index_stride = stride_order< ::std::decay_t<View> >::get_nth_largest_stride_index( Index );
  if constexpr ( Index == sizeof...(IndexType) - 1 )
  {
    return noexcept( apply_all_strided3( ::std::declval<View&&>(),
                                         ::std::declval<Lambda&&>(),
                                         ::std::declval<ExecutionPolicy&&>(),
                                         ::std::make_index_sequence<index_stride>(),
                                         ::std::make_index_sequence<sizeof...(IndexType)-index_stride-1>(),
                                         ::std::declval<::std::tuple<IndexType...>&>() ) );
  }
  else
  {
    return apply_all_strided2_is_noexcept<Index+1,View,Lambda,ExecutionPolicy,IndexType ...>();
  }
}

template < ::std::size_t Index,
           class         View,
           class         Lambda,
           class         ExecutionPolicy,
           class ...     IndexType >
constexpr void apply_all_strided2( View&&                      view,
                                   Lambda&&                    lambda,
                                   ExecutionPolicy&&           execution_policy,
                                   ::std::tuple<IndexType...>& indices )
  noexcept( apply_all_strided2_is_noexcept<Index,View,Lambda,ExecutionPolicy,IndexType ...>() );

template < ::std::size_t Index,
           class         View,
           class         Lambda,
           class         ExecutionPolicy,
           class ...     IndexType >
inline void apply_all_strided2_except( View&&                      view,
                                       Lambda&&                    lambda,
                                       ExecutionPolicy&&           execution_policy,
                                       ::std::tuple<IndexType...>& indices )
{
  constexpr ::std::size_t index_stride = stride_order< ::std::decay_t<View> >::get_nth_largest_stride_index( Index );
  // Cache the last exception to be thrown
  ::std::exception_ptr eptr;
  // Attempt lambda expression on each element
  ::std::math::detail::
  for_each( execution_policy,
            faux_index_iterator<::std::decay_t<decltype( ::std::get<index_stride>(indices) )> >(0),
            faux_index_iterator<::std::decay_t<decltype( ::std::get<index_stride>(indices) )> >(view.extent(index_stride)),
            [&view,&lambda,&execution_policy,&indices,&eptr,index_stride] ( ::std::decay_t<decltype( ::std::get<index_stride>(indices) )> index ) noexcept
              {
                ::std::get<index_stride>(indices) = index;
                try { apply_all_strided2<Index+1>( view, lambda, execution_policy, indices ); }
                catch ( ... ) { eptr = ::std::current_exception(); }
              } );
  // If exceptions were thrown, rethrow the last
  if ( eptr ) LINALG_UNLIKELY
  {
    ::std::rethrow_exception( eptr );
  }
}

template < ::std::size_t Index,
           class         View,
           class         Lambda,
           class         ExecutionPolicy,
           class ...     IndexType >
constexpr void apply_all_strided2( View&&                      view,
                                   Lambda&&                    lambda,
                                   ExecutionPolicy&&           execution_policy,
                                   ::std::tuple<IndexType...>& indices )
  noexcept( apply_all_strided2_is_noexcept<Index,View,Lambda,ExecutionPolicy,IndexType ...>() )
{
  constexpr ::std::size_t index_stride = stride_order< ::std::decay_t<View> >::get_nth_largest_stride_index( Index );
  if constexpr ( Index == sizeof...(IndexType) - 1 )
  {
    apply_all_strided3( view,
                        lambda,
                        execution_policy,
                        make_index_sequence<index_stride>(),
                        make_index_sequence<sizeof...(IndexType)-index_stride-1>(),
                        indices );
  }
  else
  {
    if constexpr ( apply_all_strided2_is_noexcept<Index+1,View,Lambda,ExecutionPolicy,IndexType ...>() )
    {
      ::std::math::detail::
      for_each( execution_policy,
                faux_index_iterator<decay_t<decltype( get<index_stride>(indices) )> >(0),
                faux_index_iterator<decay_t<decltype( get<index_stride>(indices) )> >(view.extent(index_stride)),
                [&view,&lambda,&execution_policy,&indices,index_stride] ( ::std::decay_t<decltype( ::std::get<index_stride>(indices) )> index ) constexpr noexcept
                  {
                    ::std::get<index_stride>(indices) = index;
                    apply_all_strided2<Index+1>( view, lambda, execution_policy, indices );
                  } );
    }
    else
    {
      apply_all_strided2_except<Index>( view, lambda, execution_policy, indices );
    }
  }
}

template < class View,
           class Lambda,
           class ExecutionPolicy >
[[nodiscard]] constexpr bool apply_all_strided_is_noexcept() noexcept
{
  constexpr auto rank = ::std::decay_t<View>::rank();
  auto indices = apply_all_strided_helper< ::std::make_integer_sequence< typename ::std::decay_t<View>::size_type, rank > >::initial_indices();
  return noexcept( apply_all_strided2<0>( ::std::declval<View&&>(),
                                          ::std::declval<Lambda&&>(),
                                          ::std::declval<ExecutionPolicy&&>(),
                                          indices ) );
}

template < class View,
           class Lambda,
           class ExecutionPolicy >
constexpr void apply_all_strided( View&&            view,
                                  Lambda&&          lambda,
                                  ExecutionPolicy&& execution_policy )
  noexcept( apply_all_strided_is_noexcept<View,Lambda,ExecutionPolicy>() )
{
  constexpr auto rank = ::std::decay_t<View>::rank();
  auto indices = apply_all_strided_helper< ::std::make_integer_sequence< typename ::std::decay_t<View>::size_type, rank > >::initial_indices();
  apply_all_strided2<0>( view,
                         lambda,
                         execution_policy,
                         indices );
};

template < class     View,
           class     Lambda,
           class     ExecutionPolicy,
           class ... BeforeIndexType,
           class     ExtentType >
inline void apply_all_impl2_except( View&&              view,
                                    Lambda&&            lambda,
                                    ExecutionPolicy&&   execution_policy,
                                    ExtentType          dim,
                                    BeforeIndexType ... before_indices )
{
  // Cache the last exception to be thrown
  ::std::exception_ptr eptr;
  // Attempt lambda expression on each element
  ::std::math::detail::
  for_each( execution_policy,
            faux_index_iterator<typename ::std::decay_t<View>::size_type>( 0 ),
            faux_index_iterator<typename ::std::decay_t<View>::size_type>( view.extent(dim) ),
            [ &lambda, &before_indices..., &eptr ]( typename ::std::decay_t<View>::size_type index ) noexcept
              { try { lambda( before_indices ..., index ); } catch ( ... ) { eptr = ::std::current_exception(); } } );
  // If exceptions were thrown, rethrow the last
  if ( eptr ) LINALG_UNLIKELY
  {
    ::std::rethrow_exception( eptr );
  }
}

template < class     View,
           class     Lambda,
           class     ExecutionPolicy,
           class ... BeforeIndexType,
           class     ExtentType >
constexpr void apply_all_impl2( View&&              view,
                                Lambda&&            lambda,
                                ExecutionPolicy&&   execution_policy,
                                ExtentType          dim,
                                BeforeIndexType ... before_indices )
  noexcept( noexcept( lambda( before_indices ..., ::std::declval<typename ::std::decay_t<View>::size_type>() ) ) )
{
  constexpr bool is_noexcept = noexcept( lambda( before_indices ..., ::std::declval<typename ::std::decay_t<View>::size_type>() ) );
  // If lambda expression is noexcept, then just attempt to call using whatever execution policy
  if constexpr ( is_noexcept )
  {
    ::std::math::detail::
    for_each( execution_policy,
              faux_index_iterator<typename ::std::decay_t<View>::size_type>( 0 ),
              faux_index_iterator<typename ::std::decay_t<View>::size_type>( view.extent(dim) ),
              [ &lambda, &before_indices... ]( typename ::std::decay_t<View>::size_type index ) constexpr noexcept
                { lambda( before_indices ..., index ); } );
  }
  else
  {
    apply_all_impl2_except( view, lambda, execution_policy, dim, before_indices ... );
  }
}

template < class           View,
           class           Lambda,
           class           ExecutionPolicy,
           class ...       SizeType,
           class           ExtentsType,
           ExtentsType     FirstExtents,
           ExtentsType ... Extents >
[[nodiscard]] constexpr bool apply_all_impl_is_noexcept( [[maybe_unused]] ::std::integer_sequence<ExtentsType,FirstExtents,Extents...> ) noexcept
{
  if constexpr ( sizeof...(Extents) == 0 )
  {
    return noexcept( apply_all_impl2( ::std::forward<View>( ::std::declval<View&&>() ),
                                      ::std::forward<Lambda>( ::std::declval<Lambda&&>() ),
                                      ::std::forward<ExecutionPolicy>( ::std::declval<ExecutionPolicy&&>() ),
                                      FirstExtents, ::std::declval<SizeType>() ... ) );
  }
  else
  {
    return apply_all_impl_is_noexcept<View,Lambda,ExecutionPolicy,SizeType...,typename ::std::decay_t<View>::size_type>
             ( ::std::integer_sequence<ExtentsType,Extents...>{} );
  }
}

template < class           View,
           class           Lambda,
           class           ExecutionPolicy,
           class ...       SizeType,
           class           ExtentsType,
           ExtentsType     FirstExtents,
           ExtentsType ... Extents >
constexpr void apply_all_impl( View&&                                                                        view,
                               Lambda&&                                                                      lambda,
                               ExecutionPolicy&&                                                             execution_policy,
                               [[maybe_unused]] ::std::integer_sequence<ExtentsType,FirstExtents,Extents...> first_extents,
                               SizeType ...                                                                  indices )
  noexcept( apply_all_impl_is_noexcept<View,Lambda,ExecutionPolicy,SizeType...>( ::std::integer_sequence<ExtentsType,FirstExtents,Extents...>{} ) );

template < class           View,
           class           Lambda,
           class           ExecutionPolicy,
           class ...       SizeType,
           class           ExtentsType,
           ExtentsType     FirstExtents,
           ExtentsType ... Extents >
inline void apply_all_impl_except( View&&            view,
                                   Lambda&&          lambda,
                                   ExecutionPolicy&& execution_policy,
                                   [[maybe_unused]] ::std::integer_sequence<ExtentsType,FirstExtents,Extents...>,
                                   SizeType ...      indices )
{
  // Lambda expression iterates over the next index
  auto for_each_lambda = [&view,&lambda,&execution_policy,&indices...] ( typename ::std::decay_t<View>::size_type index )
  { apply_all_impl( ::std::forward<View>( view ),
                    ::std::forward<Lambda>( lambda ),
                    ::std::forward<ExecutionPolicy>( execution_policy ),
                    ::std::integer_sequence<ExtentsType,Extents...>{}, indices ..., index ); };
  // Cache the last exception to be thrown
  ::std::exception_ptr eptr;
  // Attempt lambda expression
  ::std::math::detail::
  for_each( execution_policy,
            faux_index_iterator<typename ::std::decay_t<View>::size_type>( 0 ),
            faux_index_iterator<typename ::std::decay_t<View>::size_type>( view.extent(FirstExtents) ),
            [ &for_each_lambda, &eptr ] ( typename ::std::decay_t<View>::size_type index ) noexcept
              { try { for_each_lambda(index); } catch ( ... ) { eptr = ::std::current_exception(); } } );
  // If exceptions were thrown, rethrow the last
  if ( eptr ) LINALG_UNLIKELY
  {
    ::std::rethrow_exception( eptr );
  }
}

template < class           View,
           class           Lambda,
           class           ExecutionPolicy,
           class ...       SizeType,
           class           ExtentsType,
           ExtentsType     FirstExtents,
           ExtentsType ... Extents >
constexpr void apply_all_impl( View&&                                                                        view,
                               Lambda&&                                                                      lambda,
                               ExecutionPolicy&&                                                             execution_policy,
                               [[maybe_unused]] ::std::integer_sequence<ExtentsType,FirstExtents,Extents...> first_extents,
                               SizeType ...                                                                  indices )
  noexcept( apply_all_impl_is_noexcept<View,Lambda,ExecutionPolicy,SizeType...>( ::std::integer_sequence<ExtentsType,FirstExtents,Extents...>{} ) )
{
  if constexpr ( sizeof...(Extents) == 0 )
  {
    apply_all_impl2( ::std::forward<View>( view ),
                     ::std::forward<Lambda>( lambda ),
                     ::std::forward<ExecutionPolicy>( execution_policy ),
                     FirstExtents, indices ... );
  }
  else
  {
    // Determines if the following lambda expression is noexcept
    constexpr bool is_noexcept =
      noexcept( apply_all_impl( ::std::forward<View>( view ),
                                ::std::forward<Lambda>( lambda ),
                                ::std::forward<ExecutionPolicy>( execution_policy ),
                                ::std::integer_sequence<ExtentsType,Extents...>{}, indices ..., ::std::declval<typename ::std::decay_t<View>::size_type>() ) );
    // If lambda expression is no except then no need to try to catch exception pointers
    if constexpr ( is_noexcept )
    {
      // Lambda expression iterates over the next index
      auto for_each_lambda = [&view,&lambda,&execution_policy,&indices...] ( typename ::std::decay_t<View>::size_type index )
        constexpr noexcept( is_noexcept )
        { apply_all_impl( ::std::forward<View>( view ),
                          ::std::forward<Lambda>( lambda ),
                          ::std::forward<ExecutionPolicy>( execution_policy ),
                          ::std::integer_sequence<ExtentsType,Extents...>{}, indices ..., index ); };
      ::std::math::detail::
      for_each( execution_policy,
                faux_index_iterator<typename ::std::decay_t<View>::size_type>( 0 ),
                faux_index_iterator<typename ::std::decay_t<View>::size_type>( view.extent(FirstExtents) ),
                for_each_lambda );
    }
    else
    {
      apply_all_impl_except( view, lambda, execution_policy, first_extents, indices ... );
    }
  }
}

template < bool >
struct apply_all_maybe_strided_helper
{
  template < class View,
             class Lambda,
             class ExecutionPolicy >
  static constexpr void apply_all( View&&            view,
                                   Lambda&&          lambda,
                                   ExecutionPolicy&& execution_policy )
  {
    apply_all_strided( view, lambda, execution_policy );
  }
};

template < >
struct apply_all_maybe_strided_helper< false >
{
  template < class View,
             class Lambda,
             class ExecutionPolicy >
  static constexpr void apply_all( View&&            view,
                                   Lambda&&          lambda,
                                   ExecutionPolicy&& execution_policy )
    noexcept( noexcept( apply_all_impl( ::std::forward<View>( ::std::declval<View&&>() ),
                                        ::std::forward<Lambda>( ::std::declval<Lambda&&>() ),
                                        ::std::forward<ExecutionPolicy>( ::std::declval<ExecutionPolicy&&>() ),
                                        ::std::make_integer_sequence<typename ::std::decay_t<View>::extents_type::rank_type,::std::decay_t<View>::extents_type::rank()>{} ) ) )
  {
    apply_all_impl( ::std::forward<View>( view ),
                    ::std::forward<Lambda>( lambda ),
                    ::std::forward<ExecutionPolicy>( execution_policy ),
                    ::std::make_integer_sequence<typename ::std::decay_t<View>::extents_type::rank_type,::std::decay_t<View>::extents_type::rank()>{} );
  }
};

template < class View,
           class Lambda,
           class ExecutionPolicy >
constexpr void apply_all( View&&            view,
                          Lambda&&          lambda,
                          ExecutionPolicy&& execution_policy )
  noexcept( noexcept( apply_all_impl( ::std::forward<View>( ::std::declval<View&&>() ),
                                      ::std::forward<Lambda>( ::std::declval<Lambda&&>() ),
                                      ::std::forward<ExecutionPolicy>( ::std::declval<ExecutionPolicy&&>() ),
                                      ::std::make_integer_sequence<typename ::std::decay_t<View>::extents_type::rank_type,::std::decay_t<View>::extents_type::rank()>{} ) ) )
{
  return apply_all_maybe_strided_helper< is_defined_v< stride_order< decay_t< View > > > &&
                                         is_unsequenced_v< decay_t< ExecutionPolicy > > >::
    apply_all( view, lambda, execution_policy );
}

//==================================================================================================
//  Submdspan calls submdspan using a pair of tuples instead of a parameter pack
//==================================================================================================
template < ::std::size_t Index, class ... IndexType >
[[nodiscard]] LINALG_FORCE_INLINE_FUNCTION constexpr auto submdspan_impl2( ::std::tuple<IndexType...> start,
                                                                           ::std::tuple<IndexType...> end ) noexcept
{
  return ::std::tuple( ::std::get<Index>(start), ::std::get<Index>(end) );
}

template < class T, class E, class L, class A, class ... IndexType, ::std::size_t ... Indices >
[[nodiscard]] LINALG_FORCE_INLINE_FUNCTION constexpr auto submdspan_impl( ::std::experimental::mdspan<T,E,L,A> mds,
                                                                          ::std::tuple<IndexType...>           start,
                                                                          ::std::tuple<IndexType...>           end,
                                                                          [[maybe_unused]] ::std::index_sequence<Indices...> ) noexcept
{
  return ::std::experimental::submdspan( mds, submdspan_impl2<Indices>( start, end ) ... );
}

template < class T, class E, class L, class A, class ... IndexType >
[[nodiscard]] LINALG_FORCE_INLINE_FUNCTION constexpr auto submdspan( ::std::experimental::mdspan<T,E,L,A> mds,
                                                                     ::std::tuple<IndexType...>           start,
                                                                     ::std::tuple<IndexType...>           end ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(IndexType) == E::rank() )
#endif
{
  return submdspan_impl( mds, start, end, ::std::make_index_sequence<E::rank()>{} );
}

//==================================================================================================
//  make_from_tuple with noexcept
//==================================================================================================
template < class T, class Tuple, ::std::size_t ... Indices >
[[nodiscard]] constexpr T make_from_tuple_impl( Tuple&& t, ::std::index_sequence< Indices ... > )
  noexcept( ::std::is_nothrow_constructible_v< T, decltype( ::std::get<Indices>( ::std::declval<Tuple>() ) ) ... > )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_constructible_v< T, decltype( ::std::get<Indices>( ::std::declval<Tuple>() ) ) ... >
#endif
{
  return T( ::std::get<Indices>( ::std::forward<Tuple>(t) ) ... );
}
 
template<class T, class Tuple>
[[nodiscard]] constexpr T make_from_tuple( Tuple&& t )
  noexcept( noexcept( make_from_tuple_impl<T>( ::std::forward<Tuple>( t ), ::std::make_index_sequence< ::std::tuple_size_v< ::std::remove_reference_t< Tuple > > >{} ) ) )
{
  return make_from_tuple_impl<T>(
    ::std::forward<Tuple>( t ),
    ::std::make_index_sequence< ::std::tuple_size_v< ::std::remove_reference_t< Tuple > > >{} );
}

//==================================================================================================
//  Sufficient Extents tests if the first extents encompasses the second
//==================================================================================================
template < class SizeType, class OtherSizeType, ::std::size_t ... Extents, ::std::size_t ... OtherExtents >
[[nodiscard]] constexpr bool sufficient_extents( const ::std::experimental::extents<SizeType,Extents...>&           extents,
                                                 const ::std::experimental::extents<OtherSizeType,OtherExtents...>& other_extents ) noexcept
{
  if constexpr ( sizeof...(Extents) == sizeof...(OtherExtents) )
  {
    // Iterate over each dimension
    bool sufficient = true;
    for( ::std::size_t dim = 0; ( dim < ::std::experimental::extents<SizeType,Extents...>::rank() ) && sufficient; ++dim )
    {
      // Set to false if size is not large enough
      sufficient = ( extents.extent( dim ) >= other_extents.extent( dim ) );
    }
    return sufficient;
  }
  else
  {
    return false;
  }
}

//==================================================================================================
//  Assign View assigns views with disparate but compatable types
//==================================================================================================
template < class ToView, class FromView
#ifdef LINALG_ENABLE_CONCEPTS
  > requires is_mdspan_v< ToView > && is_mdspan_v< FromView > &&
             ::std::is_convertible_v<typename FromView::reference,typename ToView::element_type> &&
             extents_may_be_equal_v<typename FromView::extents_type,typename ToView::extents_type>
#else
  , typename = ::std::enable_if_t< ( is_mdspan_v< ToView > && is_mdspan_v< FromView > &&
                                   ::std::is_convertible_v<typename FromView::reference,typename ToView::element_type> &&
                                   extents_may_be_equal_v<typename FromView::extents_type,typename ToView::extents_type> ) > >
#endif
constexpr ToView&
assign_view( ToView& to_view, const FromView& from_view )
  noexcept( extents_are_equal_v<typename FromView::extents_type,typename ToView::extents_type> &&
            is_nothrow_convertible_v<typename FromView::reference,typename ToView::element_type> )
{
  if constexpr ( extents_are_equal_v<typename FromView::extents_type,typename ToView::extents_type> )
  {
    apply_all( from_view,
               [ &to_view, &from_view ]( auto ... indices )
                 constexpr noexcept( is_nothrow_convertible_v<typename ::std::decay_t<FromView>::reference,typename ::std::decay_t<ToView>::reference> )
                 { access( to_view, indices ... ) = access( from_view, indices ... ); },
               LINALG_EXECUTION_UNSEQ );
  }
  else
  {
    if ( sufficient_extents( to_view.extents(), from_view.extents() ) ) LINALG_LIKELY
    {
      apply_all( from_view,
                 [ &to_view, &from_view ]( auto ... indices )
                   constexpr noexcept( is_nothrow_convertible_v<typename ::std::decay_t<FromView>::reference,typename ::std::decay_t<ToView>::reference> )
                   { access( to_view, indices ... ) = access( from_view, indices ... ); },
                 LINALG_EXECUTION_UNSEQ );
    }
    else LINALG_UNLIKELY
    {
      throw length_error( "Multi-dimensional spans mismatch." );
    }
  }
  return to_view;
}

//==================================================================================================
//  Copy View copies inplace views with disparate but compatable types
//==================================================================================================
template < class ToView, class FromView
#ifdef LINALG_ENABLE_CONCEPTS
  > requires is_mdspan_v< ToView > && is_mdspan_v< FromView > &&
             ::std::is_convertible_v<typename FromView::reference,typename ToView::element_type> &&
             extents_may_be_equal_v<typename FromView::extents_type,typename ToView::extents_type>
#else
  , typename = ::std::enable_if_t< ( is_mdspan_v< ToView > && is_mdspan_v< FromView > &&
                                     ::std::is_convertible_v<typename FromView::reference,typename ToView::element_type> &&
                                     extents_may_be_equal_v<typename FromView::extents_type,typename ToView::extents_type> ) > >
#endif
constexpr void
copy_view( ToView& to_view, const FromView& from_view )
  noexcept( extents_are_equal_v<typename FromView::extents_type,typename ToView::extents_type> &&
            is_nothrow_convertible_v<typename FromView::reference,typename ToView::element_type> )
{
  if constexpr ( extents_are_equal_v<typename FromView::extents_type,typename ToView::extents_type> )
  {
    apply_all( ::std::forward<ToView>( to_view ),
              [ &to_view, &from_view ]( auto ... indices )
                constexpr noexcept( is_nothrow_convertible_v<typename ::std::decay_t<FromView>::reference,typename ::std::decay_t<ToView>::reference> )
                { ::new ( ::std::addressof( access( to_view, indices ... ) ) ) typename ToView::element_type( access( from_view, indices ... ) ); },
              LINALG_EXECUTION_UNSEQ );
  }
  else
  {
    if ( sufficient_extents( to_view.extents(), from_view.extents() ) ) LINALG_LIKELY
    {
      apply_all( forward<ToView>( to_view ),
                [ &to_view, &from_view ]( auto ... indices )
                  constexpr noexcept( is_nothrow_convertible_v<typename ::std::decay_t<FromView>::reference,typename ::std::decay_t<ToView>::reference> )
                  { ::new ( ::std::addressof( access( to_view, indices ... ) ) ) typename ToView::element_type( access( from_view, indices ... ) ); },
                LINALG_EXECUTION_UNSEQ );
    }
    else LINALG_UNLIKELY
    {
      throw length_error( "Multi-dimensional spans mismatch." );
    }
  }
}

//==================================================================================================
//  Is Complex returns true if the type is a complex type
//==================================================================================================
template < class T >
struct is_complex : public ::std::false_type {};

template < class T >
struct is_complex< complex<T> > : public ::std::true_type {};

template < class T >
inline constexpr bool is_complex_v = is_complex<T>::value;

//==================================================================================================
//  Rebind Accessor rebinds the accessor to a new value type
//==================================================================================================
template < class Access, class ValueType >
struct rebind_accessor { using type = Access; };

template < template < class > class Access, class OtherType, class ValueType >
struct rebind_accessor< Access<OtherType>, ValueType > { using type = Access<ValueType>; };

template < class Access, class ValueType >
using rebind_accessor_t = typename rebind_accessor<Access,ValueType>::type;

//==================================================================================================
//  Rebind Layout rebinds the layout to a new extents type
//==================================================================================================

template < class LayoutType, class ExtentsType >
struct rebind_layout { using type = LayoutType; };

template < template < class > class LayoutType, class OtherType, class ExtentsType >
struct rebind_layout< LayoutType<OtherType>, ExtentsType > { using type = LayoutType<ExtentsType>; };

template < class LayoutType, class ExtentsType >
using rebind_layout_t = typename rebind_layout<LayoutType,ExtentsType>::type;

//==================================================================================================
//  Helper class for manipulating extents
//==================================================================================================
template < class U > class extents_helper_impl {};
template < class U, ::std::size_t ... Indices >
class extents_helper_impl< ::std::integer_sequence<U,Indices...> >
{
  private:
    static constexpr U dyn_ext( [[maybe_unused]] U index ) noexcept { return ::std::experimental::dynamic_extent; }
  public:
    using extents_type = ::std::experimental::extents<U,dyn_ext(Indices)...>;
    using tuple_type   = ::std::tuple<decltype(Indices)...>;
    [[nodiscard]] static constexpr U size( const extents_type& e ) noexcept { return ( e.extent(Indices) * ... ); }
    [[nodiscard]] static constexpr extents_type zero() noexcept { return extents_type( U( 0 * Indices ) ... ); }
    [[nodiscard]] static constexpr auto zero_tuple() noexcept { return ::std::tuple( U( 0 * Indices ) ... ); }
    [[nodiscard]] static constexpr auto to_tuple( const extents_type& e ) noexcept { return ::std::tuple( U( e.extent(Indices) ) ... ); }
};
template < class U, ::std::size_t R >
using extents_helper = extents_helper_impl< ::std::make_integer_sequence<U,R> >;

}       //- detail namespace
}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_PRIVATE_SUPPORT_HPP
