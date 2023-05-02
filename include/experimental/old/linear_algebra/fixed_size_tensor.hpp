//==================================================================================================
//  File:       fixed_size_tensor.hpp
//
//  Summary:    This header defines a fixed-size tensor.  In this context, fixed-size
//              means that the row and column extents of such objects are known at compile-time.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_FIXED_SIZE_TENSOR_HPP
#define LINEAR_ALGEBRA_FIXED_SIZE_TENSOR_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace experimental
{
namespace math
{

// TODO: There doesn't seem to be a good way to map multidimensional
//       view or indices back into one dimensional order. Several
//       constructors would benefit from mapping into an initializer
//       list and avoiding the default construction of the array.
// TODO: Lacking support for P0478 (rejected), syntax is awkward
//       and doesn't support natural defaults.
//       Could capture indices in an extent, but this isn't as natural.
/// @brief Fixed-size, fixed-capacity tensor.
//         Implementation satisfies the following concepts:
//         concepts::fixed_size_tensor
//         concepts::writable_tensor
/// @tparam T element_type
/// @tparam L layout defines the ordering of elements in memory
/// @tparam A accessor policy defines how elements are accessed
/// @tparam Ds... length of each dimension of the tensor
template < class             T,
           class             L,
           class             A,
           ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... ) // Each dimension must be >= 0
#endif
class fs_tensor
{
  public:
    //- Types

    /// @brief Type used to define memory layout
    using layout_type                = L;
    /// @brief Type used to define access into memory
    using accessor_type              = A;
    /// @brief Type contained by the tensor
    using element_type               = typename accessor_type::element_type;
    /// @brief Type returned by const index access
    using value_type                 = ::std::remove_cv_t<element_type>;
    /// @brief Type used for indexing
    using index_type                 = ::std::ptrdiff_t;
    /// @brief Type used for size along any dimension
    using size_type                  = ::std::size_t;
    /// @brief Type used to express size of tensor
    using extents_type               = ::std::experimental::extents<size_type,Ds...>;
    /// @brief Type used to represent a node in the tensor
    using tuple_type                 = tuple<decltype(Ds) ...>;
    /// @brief Type used to const view memory
    using const_underlying_span_type = ::std::experimental::mdspan<const element_type,extents_type,layout_type,detail::rebind_accessor_t<accessor_type,const element_type> >;
    /// @brief Type used to view memory
    using underlying_span_type       = ::std::experimental::mdspan<element_type,extents_type,layout_type,accessor_type>;
    /// @brief Type used to portray tensor as an N dimensional view
    using span_type                  = const_underlying_span_type;
    /// @brief Type returned by mutable index access
    using reference                  = typename accessor_type::reference;

    //- Rebind

    /// @brief Rebind defines a type for a rebinding a fized size tensor to the new type parameters
    /// @tparam ValueType  rebound value type
    /// @tparam LayoutType rebound layout policy
    /// @tparam AccessType rebound access policy
    template < class ValueType,
               class LayoutType   = layout_type,
               class AccessorType = accessor_type >
    class rebind
    {
    private:
      using rebind_accessor_type = detail::rebind_accessor_t<AccessorType,ValueType>;
      using rebind_element_type  = typename rebind_accessor_type::element_type;
    public:
      using type = fs_tensor< ValueType,
                              LayoutType,
                              rebind_accessor_type,
                              Ds ... >;
    };
    /// @brief Helper for defining rebound tensor type
    /// @tparam ValueType  rebound value type
    /// @tparam LayoutType rebound layout policy
    /// @tparam AccessType rebound access policy
    template < class ValueType,
               class LayoutType   = layout_type,
               class AccessorType = accessor_type >
    using rebind_t = typename rebind< ValueType, LayoutType, AccessorType >::type;

    //- Destructor / Constructors / Assignments

    /// @brief Default constructor
    constexpr fs_tensor()                   noexcept( ::std::is_nothrow_default_constructible_v<element_type> );
    /// @brief Default move constructor
    /// @param fs_tensor to be moved
    constexpr fs_tensor( fs_tensor&& )      noexcept( ::std::is_nothrow_move_constructible_v<element_type> );
    /// @brief Default copy constructor
    /// @param fs_tensor to be copied
    constexpr fs_tensor( const fs_tensor& ) noexcept( ::std::is_nothrow_copy_constructible_v<element_type> );
    // TODO: Define noexcept specification
    /// @brief Template copy constructor
    /// @tparam tensor to be copied
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::tensor_may_be_constructible< fs_tensor > T2 >
    #else
    template < class T2, typename = ::std::enable_if_t< concepts::tensor_may_be_constructible< T2, fs_tensor > > >
    #endif
    explicit constexpr fs_tensor( const T2& rhs );
    /// @brief Construct from a view
    /// @tparam An N dimensional view
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< fs_tensor > MDS >
    #else
    template < class MDS, typename = ::std::enable_if_t< concepts::view_may_be_constructible_to_tensor<MDS,fs_tensor> >, typename = ::std::enable_if_t<true> >
    #endif
    explicit constexpr fs_tensor( const MDS& view ) noexcept( concepts::view_is_nothrow_constructible_to_tensor<MDS,fs_tensor> );
    /// @brief Construct by applying lambda to every element in the tensor
    /// @tparam lambda lambda expression encapsulating operation to be performed on each element
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = ::std::enable_if_t< ::std::is_convertible_v< decltype( declval<Lambda&&>().operator()( Ds ... ) ), element_type > > >
    #endif
    explicit constexpr fs_tensor( Lambda&& lambda ) noexcept( noexcept( ::std::declval<Lambda&&>()( Ds ... ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires requires { { ::std::declval<Lambda&&>().operator()( Ds ... ) } -> ::std::convertible_to<element_type>; };
    #else
      ;
    #endif
    /// @brief Default move assignment
    /// @param  fs_tensor to be moved
    /// @return self
    constexpr fs_tensor& operator = ( fs_tensor&& )      = default;
    /// @brief Default copy assignment
    /// @param  fs_tensor to be copied
    /// @return self
    constexpr fs_tensor& operator = ( const fs_tensor& ) = default;
    // TODO: Define noexcept specification.
    /// @brief Template copy assignment
    /// @tparam type of tensor to be copied
    /// @param  tensor to be copied
    /// @returns self
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::tensor_may_be_constructible< fs_tensor > T2 >
    #else
    template < class T2, typename = ::std::enable_if_t< concepts::tensor_may_be_constructible< T2, fs_tensor > > >
    #endif
    constexpr fs_tensor& operator = ( const T2& rhs );
    // TODO: Define noexcept specification.
    /// @brief Construct from a two dimensional view
    /// @tparam type of view to be copied
    /// @param  view to be copied
    /// @returns self
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< fs_tensor > MDS >
    #else
    template < class MDS, typename = ::std::enable_if_t< concepts::view_may_be_constructible_to_tensor<MDS,fs_tensor> >, typename = ::std::enable_if_t<true> >
    #endif
    constexpr fs_tensor& operator = ( const MDS& view );

    //- Size / Capacity

    /// @brief Returns the current number of (rows,columns,depth,etc..)
    /// @return number of (rows,columns)
    [[nodiscard]] constexpr extents_type size() const noexcept;
    /// @brief Returns the current capacity of (rows,columns,depth,etc...)
    /// @return capacity of (rows,columns)
    [[nodiscard]] constexpr extents_type capacity() const noexcept;

    //- Data access
    
    /// @brief returns a const N dimensional span of extents(Ds...)
    /// @returns const N dimensional span of extents(Ds...) 
    [[nodiscard]] constexpr span_type                  span() const noexcept;
    /// @returns mutable N dimensional span of extents(Ds...) 
    /// @brief returns a mutable N dimensional span of extents(Ds...)
    [[nodiscard]] constexpr underlying_span_type       underlying_span() noexcept;
    /// @brief returns a const N dimensional span of extents(Ds...)
    /// @returns const N dimensional span of extents(Ds...) 
    [[nodiscard]] constexpr const_underlying_span_type underlying_span() const noexcept;

    //- Const views

    /// @brief Returns the value at (indices...) without index bounds checking
    /// @param indices set indices representing a node in the tensor
    /// @returns value at row i, column j, depth k, etc.
    #if LINALG_USE_BRACKET_OPERATOR
    [[nodiscard]] constexpr value_type operator[]( decltype(Ds) ... indices ) const noexcept;
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    [[nodiscard]] constexpr value_type operator()( decltype(Ds) ... indices ) const noexcept;
    #endif
    /// @brief Returns a const vector view
    /// @tparam ...SliceArgs argument types used to get a const vector view
    /// @param ...args aguments to get a const vector view
    /// @return const vector view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto subvector( SliceArgs ... args ) const
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 1 );
    #else
      ;
    #endif
    /// @brief Returns a const matrix view
    /// @tparam ...SliceArgs argument types used to get a const matrix view
    /// @param ...args aguments to get a const matrix view
    /// @return const matrix view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto submatrix( SliceArgs ... args ) const
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 2 );
    #else
      ;
    #endif
    /// @brief Returns a const view of the specified subtensor
    /// @tparam ...SliceArgs argument types used to get a tensor view
    /// @param ...args aguments to get a tensor view
    /// @return const tensor view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto subtensor( SliceArgs ... args ) const;

    //- Mutable views

    /// @brief Returns a mutable value at (indices...) without index bounds checking
    /// @param indices set indices representing a node in the tensor
    /// @returns mutable value at row i, column j, depth k, etc.
    #if LINALG_USE_BRACKET_OPERATOR
    [[nodiscard]] constexpr reference operator[]( decltype(Ds) ... indices ) noexcept;
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    [[nodiscard]] constexpr reference operator()( decltype(Ds) ... indices ) noexcept;
    #endif
    /// @brief Returns a vector view
    /// @tparam ...SliceArgs argument types used to get a vector view
    /// @param ...args aguments to get a vector view
    /// @return vector view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto subvector( SliceArgs ... args )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 1 );
    #else
      ;
    #endif
    /// @brief Returns a matrix view
    /// @tparam ...SliceArgs argument types used to get a matrix view
    /// @param ...args aguments to get a matrix view
    /// @return matrix view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto submatrix( SliceArgs ... args )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 2 );
    #else
      ;
    #endif
    /// @brief Returns a mutable view of the specified subtensor
    /// @tparam ...SliceArgs argument types used to get a tensor view
    /// @param ...args aguments to get a tensor view
    /// @return mutable tensor view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto subtensor( SliceArgs ... args );

  private:
    //- Data

    /// @brief Number of elements in array
    static const size_type nelems_ = detail::product( Ds ... );
    /// @brief Array of elements
    ::std::array<element_type,nelems_> elems_;
};

//----------------------------------------------
// Implementation of fs_tensor<T,R,C,L,A>
//----------------------------------------------

//- Destructor / Constructors / Assignments

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
constexpr fs_tensor<T,L,A,Ds...>::fs_tensor()
  noexcept( ::std::is_nothrow_default_constructible_v<typename fs_tensor<T,L,A,Ds...>::element_type> ) :
  elems_()
{
}

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
constexpr fs_tensor<T,L,A,Ds...>::fs_tensor( fs_tensor&& rhs )
  noexcept( ::std::is_nothrow_move_constructible_v<element_type> ) :
  elems_( ::std::move( rhs.elems_ ) )
{
}

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
constexpr fs_tensor<T,L,A,Ds...>::fs_tensor( const fs_tensor& rhs )
  noexcept( ::std::is_nothrow_copy_constructible_v<element_type> ) :
  elems_( rhs.elems_ )
{
}

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_may_be_constructible< fs_tensor<T,L,A,Ds...> > T2 >
#else
template < class T2, typename >
#endif
constexpr fs_tensor<T,L,A,Ds...>::fs_tensor( const T2& rhs )
{
  underlying_span_type this_view { this->underlying_span() };
  static_cast<void>( detail::assign_view( this_view, rhs.span() ) );
}

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::view_may_be_constructible_to_tensor< fs_tensor<T,L,A,Ds...> > MDS >
#else
template < class MDS, typename, typename >
#endif
constexpr fs_tensor<T,L,A,Ds...>::fs_tensor( const MDS& view )
  noexcept( concepts::view_is_nothrow_constructible_to_tensor< MDS, fs_tensor<T,L,A,Ds...> > )
{
  underlying_span_type this_view { this->underlying_span() };
  static_cast<void>( detail::assign_view( this_view, view ) );
}

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
#ifdef LINALG_ENABLE_CONCEPTS
template < class Lambda >
#else
template < class Lambda, typename >
#endif
constexpr fs_tensor<T,L,A,Ds...>::fs_tensor( Lambda&& lambda ) noexcept( noexcept( ::std::declval<Lambda&&>()( Ds ... ) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires { { ::std::declval<Lambda&&>().operator()( Ds ... ) } -> ::std::convertible_to<typename fs_tensor<T,L,A,Ds...>::element_type>; }
#endif
{
  // If expression is no except, then no need to capture last exception
  constexpr bool lambda_is_noexcept = detail::is_nothrow_convertible_v< decltype( ::std::declval<Lambda&&>()( Ds ... ) ), element_type >;
  // Construct all elements from lambda output
  auto ctor = [this,&lambda]( auto ... indices ) constexpr noexcept( lambda_is_noexcept )
  {
    #if LINALG_USE_BRACKET_OPERATOR
    this->underlying_span()[ indices ... ] = lambda( indices ... );
    #else
    this->underlying_span()( indices ... ) = lambda( indices ... );
    #endif
  };
  detail::apply_all( this->underlying_span(), ctor, LINALG_EXECUTION_UNSEQ );
}

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_may_be_constructible< fs_tensor<T,L,A,Ds...> > T2 >
#else
template < class T2, typename >
#endif
constexpr fs_tensor<T,L,A,Ds...>& fs_tensor<T,L,A,Ds...>::operator = ( const T2& rhs )
{
  underlying_span_type this_view { this->underlying_span() };
  static_cast<void>( detail::assign_view( this_view, rhs.span() ) );
  return *this;
}

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::view_may_be_constructible_to_tensor< fs_tensor<T,L,A,Ds...> > MDS >
#else
template < class MDS, typename, typename >
#endif
constexpr fs_tensor<T,L,A,Ds...>& fs_tensor<T,L,A,Ds...>::operator = ( const MDS& view )
{
  underlying_span_type this_view = { this->underlying_span() };
  static_cast<void>( detail::assign_view( this_view, view ) );
  return *this;
}

//- Size / Capacity

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
[[nodiscard]] constexpr typename fs_tensor<T,L,A,Ds...>::extents_type
fs_tensor<T,L,A,Ds...>::size() const noexcept
{
  return extents_type();
}

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
[[nodiscard]] constexpr typename fs_tensor<T,L,A,Ds...>::extents_type
fs_tensor<T,L,A,Ds...>::capacity() const noexcept
{
  return this->size();
}

//- Const views

#if LINALG_USE_BRACKET_OPERATOR
template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
[[nodiscard]] constexpr typename fs_tensor<T,L,A,Ds...>::value_type
fs_tensor<T,L,A,Ds...>::operator[]( decltype(Ds) ... indices ) const noexcept
{
  return this->underlying_span()[ indices ... ];
}
#endif

#if LINALG_USE_PAREN_OPERATOR
template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
[[nodiscard]] constexpr typename fs_tensor<T,L,A,Ds...>::value_type
fs_tensor<T,L,A,Ds...>::operator()( decltype(Ds) ... indices ) const noexcept
{
  return this->underlying_span()( indices ... );
}
#endif

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
template < class ... SliceArgs >
[[nodiscard]] constexpr auto fs_tensor<T,L,A,Ds...>::subvector( SliceArgs ... args ) const
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 1 )
#endif
{
  using subspan_type = decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
  return vector_view<subspan_type>( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
}

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
template < class ... SliceArgs >
[[nodiscard]] constexpr auto fs_tensor<T,L,A,Ds...>::submatrix( SliceArgs ... args ) const
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 2 )
#endif
{
  using subspan_type = decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
  return matrix_view<subspan_type>( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
}

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
template < class ... SliceArgs >
[[nodiscard]] constexpr auto fs_tensor<T,L,A,Ds...>::subtensor( SliceArgs ... args ) const
{
  using subspan_type = decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
  return tensor_view<subspan_type>( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
}

//- Mutable views

#if LINALG_USE_BRACKET_OPERATOR
template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
[[nodiscard]] constexpr typename fs_tensor<T,L,A,Ds...>::reference
fs_tensor<T,L,A,Ds...>::operator[]( decltype(Ds) ... indices ) noexcept
{
  return this->underlying_span()[ indices ... ];
}
#endif

#if LINALG_USE_PAREN_OPERATOR
template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
[[nodiscard]] constexpr typename fs_tensor<T,L,A,Ds...>::reference
fs_tensor<T,L,A,Ds...>::operator()( decltype(Ds) ... indices ) noexcept
{
  return this->underlying_span()( indices ... );
}
#endif

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
template < class ... SliceArgs >
[[nodiscard]] constexpr auto fs_tensor<T,L,A,Ds...>::subvector( SliceArgs ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 1 )
#endif
{
  using subspan_type = decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
  return vector_view<subspan_type>( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
}

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
template < class ... SliceArgs >
[[nodiscard]] constexpr auto fs_tensor<T,L,A,Ds...>::submatrix( SliceArgs ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 2 )
#endif
{
  using subspan_type = decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
  return matrix_view<subspan_type>( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
}

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
template < class ... SliceArgs >
[[nodiscard]] constexpr auto fs_tensor<T,L,A,Ds...>::subtensor( SliceArgs ... args )
{
  using subspan_type = decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
  return tensor_view<subspan_type>( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
}

//- Data access

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
[[nodiscard]] constexpr typename fs_tensor<T,L,A,Ds...>::span_type
fs_tensor<T,L,A,Ds...>::span() const noexcept
{
  return this->underlying_span();
}

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
[[nodiscard]] constexpr typename fs_tensor<T,L,A,Ds...>::underlying_span_type
fs_tensor<T,L,A,Ds...>::underlying_span() noexcept
{
  return underlying_span_type( &this->elems_[0], extents_type() );
}

template < class T, class L, class A, ::std::size_t ... Ds >
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( ( Ds >= 0 ) && ... )
#endif
[[nodiscard]] constexpr typename fs_tensor<T,L,A,Ds...>::const_underlying_span_type
fs_tensor<T,L,A,Ds...>::underlying_span() const noexcept
{
  return const_underlying_span_type( &this->elems_[0], extents_type() );
}

}       //- math namespace
}       //- experimental namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_FIXED_SIZE_TENSOR_HPP
