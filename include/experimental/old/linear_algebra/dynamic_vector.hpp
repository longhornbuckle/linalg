//==================================================================================================
//  File:       dynamic_vector.hpp
//
//  Summary:    This header defines a dynamic vector. In this context, dynamic
//              means the length of such objects are unknown at compile-time.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_DYNAMIC_VECTOR_HPP
#define LINEAR_ALGEBRA_DYNAMIC_VECTOR_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace experimental
{
namespace math
{

/// @brief Dynamic-size, dynamic-capacity vector.
//         Implementation satisfies the following concepts:
//         concepts::dynamic_vector
//         concepts::writable_vector
/// @tparam T      element_type
/// @tparam Alloc  allocator_type
/// @tparam L      layout defines the ordering of elements in memory
/// @tparam Access accessor policy defines how elements are accessed
template < class T,
           class Alloc,
           class L,
           class Access >
class dr_vector : public dr_tensor<T,1,Alloc,L,Access>
{
  private:
    // Base tensor type
    using base_type = dr_tensor<T,1,Alloc,L,Access>;
  public:
    //- Types

    /// @brief Type used to define memory layout
    using layout_type                = typename base_type::layout_type;
    /// @brief Type used to define access into memory
    using accessor_type              = typename base_type::accessor_type;
    /// @brief Type contained by the vector
    using element_type               = typename base_type::element_type;
    /// @brief Type returned by const index access
    using value_type                 = typename base_type::value_type;
    /// @brief Type of allocator used to get memory
    using allocator_type             = typename base_type::allocator_type;
    /// @brief Type used for indexing
    using index_type                 = typename base_type::index_type;
    /// @brief Type used for size along any dimension
    using size_type                  = typename base_type::size_type;
    /// @brief Type used to express size of tensor
    using extents_type               = typename base_type::extents_type;
    /// @brief Type used to represent a node in the tensor
    using tuple_type                 = typename base_type::tuple_type;
    /// @brief Type used to const view memory
    using const_underlying_span_type = typename base_type::const_underlying_span_type;
    /// @brief Type used to view memory
    using underlying_span_type       = typename base_type::underlying_span_type;
    /// @brief Type used to portray tensor as an N dimensional view
    using span_type                  = typename base_type::span_type;
    /// @brief Type returned by mutable index access
    using reference                  = typename base_type::reference;
    /// @brief mutable view of a subtensor
    using subvector_type             = vector_view<decltype( ::std::experimental::submdspan( ::std::declval<underlying_span_type>(), ::std::declval< ::std::tuple<index_type,index_type> >() ) ) >;
    /// @brief const view of a subtensor
    using const_subvector_type       = vector_view<decltype( ::std::experimental::submdspan( ::std::declval<const_underlying_span_type>(), ::std::declval< ::std::tuple<index_type,index_type> >() ) ) >;

    //- Rebind

    /// @brief Rebind defines a type for a rebinding a dynamic vector to the new type parameters
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
      using type = dr_vector< ValueType,
                              typename ::std::allocator_traits<allocator_type>::template rebind_alloc<rebind_element_type>,
                              LayoutType,
                              rebind_accessor_type >;
    };
    /// @brief Helper for defining rebound type
    /// @tparam ValueType  rebound value type
    /// @tparam LayoutType rebound layout policy
    /// @tparam AccessType rebound access policy
    template < class  ValueType,
               class  LayoutType   = layout_type,
               class  AccessorType = accessor_type >
    using rebind_t = typename rebind< ValueType, LayoutType, AccessorType >::type;

    //- Destructor / Constructors / Assignments

    /// @brief Default destructor
    ~dr_vector()                            = default;
    /// @brief Default constructor
    constexpr dr_vector()                   = default;
    /// @brief Default move constructor
    /// @param dr_vector to be moved
    constexpr dr_vector( dr_vector&& )      = default;
    /// @brief Default copy constructor
    /// @param dr_vector to be copied
    constexpr dr_vector( const dr_vector& ) = default;
    /// @brief Template copy constructor
    /// @tparam vector to be copied
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::tensor_may_be_constructible< dr_vector > V2 >
    #else
    template < class V2, typename = ::std::enable_if_t< concepts::tensor_may_be_constructible< V2, dr_vector > > >
    #endif
    explicit constexpr dr_vector( const V2& rhs ) noexcept( noexcept( base_type(rhs) ) );
    /// @brief Construct from a view
    /// @tparam A two dimensional view
    /// @param view view of vector elements
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< dr_vector > MDS >
    #else
    template < class MDS, typename = ::std::enable_if_t< concepts::view_may_be_constructible_to_tensor<MDS,dr_vector> && ::std::is_default_constructible_v<allocator_type> >, typename = ::std::enable_if_t<true> >
    #endif
    explicit constexpr dr_vector( const MDS& view ) noexcept( noexcept( base_type(view) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ::std::is_default_constructible_v<allocator_type>;
    #else
      ;
    #endif
    /// @brief Attempt to allocate sufficient resources for a size s vector and construct
    /// @param s defines the length of the vector
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename = ::std::enable_if_t< ::std::is_default_constructible_v<allocator_type> > >
    #endif
    explicit constexpr dr_vector( extents_type s ) noexcept( noexcept( base_type(s) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ::std::is_default_constructible_v<allocator_type>;
    #else
      ;
    #endif
    /// @brief Attempt to allocate sufficient resources for a size s vector with the input capacity and construct
    /// @param s defines the length of the vector
    /// @param cap defines the capacity of the vector
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename = ::std::enable_if_t< ::std::is_default_constructible_v<allocator_type> > >
    #endif
    constexpr dr_vector( extents_type s, extents_type cap ) noexcept( noexcept( base_type(s,cap) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ::std::is_default_constructible_v<allocator_type>;
    #else
      ;
    #endif
    /// @brief Construct by applying lambda to every element in the vector
    /// @tparam Lambda lambda expression with an operator()( index ) defined
    /// @param s defines the length of the vector
    /// @param lambda lambda expression to be performed on each element
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = ::std::enable_if_t< ::std::is_default_constructible_v<allocator_type> &&
                                              ::std::is_convertible_v< decltype( ::std::declval<Lambda&&>()( ::std::declval<index_type>() ) ), element_type > > >
    #endif
    constexpr dr_vector( extents_type s, Lambda&& lambda ) noexcept( noexcept( base_type(s,lambda) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ::std::is_default_constructible_v<allocator_type> &&
               requires { { ::std::declval<Lambda&&>()( ::std::declval<index_type>() ) } -> ::std::convertible_to<element_type>; };
    #else
      ;
    #endif
    /// @brief Construct by applying lambda to every element in the vector
    /// @tparam Lambda lambda expression with an operator()( index ) defined
    /// @param s defines the length of the vector
    /// @param cap defines the capacity of the vector
    /// @param lambda lambda expression to be performed on each element
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = ::std::enable_if_t< ::std::is_default_constructible_v<allocator_type> &&
                                              ::std::is_convertible_v< decltype( ::std::declval<Lambda&&>()( ::std::declval<index_type>() ) ), element_type > > >
    #endif
    constexpr dr_vector( extents_type s, extents_type cap, Lambda&& lambda ) noexcept( noexcept( base_type(s,cap,lambda) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ::std::is_default_constructible_v<allocator_type> &&
               requires { { ::std::declval<Lambda&&>()( ::std::declval<index_type>() ) } -> ::std::convertible_to<element_type>; };
    #else
      ;
    #endif
    /// @brief Construct empty dimensionless vector with an allocator
    /// @param alloc allocator to construct with
    explicit constexpr dr_vector( const allocator_type& alloc ) noexcept( noexcept( base_type(alloc) ) );
    /// @brief Construct from a view
    /// @tparam An N dimensional view
    /// @param view view of vector elements
    /// @param alloc allocator to construct with
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< dr_vector > MDS >
    #else
    template < class MDS, typename = ::std::enable_if_t< concepts::view_may_be_constructible_to_tensor< MDS, dr_vector > > >
    #endif
    explicit constexpr dr_vector( const MDS& view, const allocator_type& alloc ) noexcept( noexcept( base_type(view,alloc) ) );
    /// @brief Attempt to allocate sufficient resources for a size vector and construct
    /// @param s defines the length of the vector
    /// @param alloc allocator used to construct with
    constexpr dr_vector( extents_type s, const allocator_type& alloc ) noexcept( noexcept( base_type(s,alloc) ) );
    /// @brief Attempt to allocate sufficient resources for a size s vector with the input capacity and construct
    /// @param s defines the length of the vector
    /// @param cap defines the capacity of the vector
    /// @param alloc allocator used to construct with
    constexpr dr_vector( extents_type s, extents_type cap, const allocator_type& alloc ) noexcept( noexcept( base_type(s,cap,alloc) ) );
    /// @brief Construct by applying lambda to every element in the vector
    /// @tparam Lambda lambda expression with an operator()( index ) defined
    /// @param s defines the length of the vector
    /// @param lambda lambda expression to be performed on each element
    /// @param alloc allocator used to construct with
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = ::std::enable_if_t< ::std::is_default_constructible_v<allocator_type> &&
                                              ::std::is_convertible_v< decltype( ::std::declval<Lambda&&>()( ::std::declval<index_type>() ) ), element_type > > >
    #endif
    constexpr dr_vector( extents_type s, Lambda&& lambda, const allocator_type& alloc ) noexcept( noexcept( base_type(s,lambda,alloc) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires requires { { ::std::declval<Lambda&&>()( ::std::declval<index_type>() ) } -> ::std::convertible_to<element_type>; };
    #else
      ;
    #endif
    /// @brief Construct by applying lambda to every element in the vector
    /// @tparam Lambda lambda expression with an operator()( index ) defined
    /// @param s defines the length of the vector
    /// @param cap defines the capacity of the vector
    /// @param lambda lambda expression to be performed on each element
    /// @param alloc allocator used to construct with
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = ::std::enable_if_t< ::std::is_default_constructible_v<allocator_type> &&
                                              ::std::is_convertible_v< decltype( ::std::declval<Lambda&&>()( ::std::declval<index_type>() ) ), element_type > > >
    #endif
    constexpr dr_vector( extents_type s, extents_type cap, Lambda&& lambda, const allocator_type& alloc ) noexcept( noexcept( base_type(s,cap,lambda,alloc) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires requires { { ::std::declval<Lambda&&>()( ::std::declval<index_type>() ) } -> ::std::convertible_to<element_type>; };
    #else
      ;
    #endif
    /// @brief Default move constructor
    /// @param  dr_vector to be moved
    /// @return self
    constexpr dr_vector& operator = ( dr_vector&& )      = default;
    /// @brief Default copy constructor
    /// @param  dr_vector to be copied
    /// @return self
    constexpr dr_vector& operator = ( const dr_vector& ) = default;
    /// @brief Template copy constructor
    /// @tparam type of vector to be copied
    /// @param  vector to be copied
    /// @returns self
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::tensor_may_be_constructible< dr_vector > V2 >
    #else
    template < class V2, typename = ::std::enable_if_t< concepts::tensor_may_be_constructible< V2, dr_vector > > >
    #endif
    constexpr dr_vector& operator = ( const V2& rhs ) noexcept( noexcept( ::std::declval<base_type>() = rhs ) );
    /// @brief Construct from a two dimensional view
    /// @tparam type of view to be copied
    /// @param  view to be copied
    /// @returns self
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< dr_vector > MDS >
    #else
    template < class MDS, typename = ::std::enable_if_t< concepts::view_may_be_constructible_to_tensor<MDS,dr_vector> && ::std::is_default_constructible_v<allocator_type> >, typename = ::std::enable_if_t<true> >
    #endif
    constexpr dr_vector& operator = ( const MDS& view ) noexcept( noexcept( ::std::declval<base_type>() = view ) );

    //- Size / Capacity

    using base_type::size;
    using base_type::capacity;
    using base_type::resize;
    using base_type::reserve;

    //- Const views

    #if LINALG_USE_BRACKET_OPERATOR
    using base_type::operator[]; // Brings into scope const and mutable
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    using base_type::operator(); // Brings into scope const and mutable
    #endif

    /// @brief Returns a const view of the specified subvector
    /// @param start (row,column) start of subvector
    /// @param end (row,column) end of subvector
    /// @returns const view of the specified subvector
    [[nodiscard]] constexpr const_subvector_type subvector( index_type start,
                                                            index_type end ) const;

    //- Mutable views

    /// @brief Returns a mutable view of the specified subvector
    /// @param start (row,column) start of subvector
    /// @param end (row,column) end of subvector
    /// @returns mutable view of the specified subvector
    [[nodiscard]] constexpr subvector_type subvector( index_type start,
                                                      index_type end );

    //- Data access

    using base_type::span;
    using base_type::underlying_span;

};

//----------------------------------------------
// Implementation of dr_vector<T,Alloc,L,Access>
//----------------------------------------------

//- Destructor / Constructors / Assignments

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_may_be_constructible< dr_vector<T,Alloc,L,Access> > V2 >
#else
template < class V2, typename >
#endif
constexpr dr_vector<T,Alloc,L,Access>::dr_vector( const V2& rhs )
  noexcept( noexcept( typename dr_vector<T,Alloc,L,Access>::base_type( rhs ) ) ) :
  dr_vector<T,Alloc,L,Access>::base_type( rhs )
{
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::view_may_be_constructible_to_tensor< dr_vector<T,Alloc,L,Access> > MDS >
#else
template < class MDS, typename, typename >
#endif
constexpr dr_vector<T,Alloc,L,Access>::dr_vector( const MDS& view )
  noexcept( noexcept( dr_vector<T,Alloc,L,Access>::base_type(view) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_default_constructible_v<typename dr_vector<T,Alloc,L,Access>::allocator_type> :
#else
  :
#endif
  dr_vector<T,Alloc,L,Access>::base_type(view)
{
}
  
template < class T, class Alloc, class L, class Access >
#ifndef LINALG_ENABLE_CONCEPTS
template < typename >
#endif
constexpr dr_vector<T,Alloc,L,Access>::dr_vector( extents_type s )
  noexcept( noexcept( dr_vector<T,Alloc,L,Access>::base_type(s) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_default_constructible_v<typename dr_vector<T,Alloc,L,Access>::allocator_type> :
#else
  :
#endif
  dr_vector<T,Alloc,L,Access>::base_type(s)
{
}

template < class T, class Alloc, class L, class Access >
#ifndef LINALG_ENABLE_CONCEPTS
template < typename >
#endif
constexpr dr_vector<T,Alloc,L,Access>::dr_vector( extents_type s, extents_type cap )
  noexcept( noexcept( dr_vector<T,Alloc,L,Access>::base_type(s,cap) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_default_constructible_v<typename dr_vector<T,Alloc,L,Access>::allocator_type> :
#else
  :
#endif
  dr_vector<T,Alloc,L,Access>::base_type(s,cap)
{
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < class Lambda >
#else
template < class Lambda, typename >
#endif
constexpr dr_vector<T,Alloc,L,Access>::dr_vector( extents_type s, Lambda&& lambda )
  noexcept( noexcept( dr_vector<T,Alloc,L,Access>::base_type(s,lambda) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_default_constructible_v<typename dr_vector<T,Alloc,L,Access>::allocator_type> &&
           requires { { ::std::declval<Lambda&&>()( ::std::declval<typename dr_vector<T,Alloc,L,Access>::index_type>() ) }
                      -> ::std::convertible_to<typename dr_vector<T,Alloc,L,Access>::element_type>; } :
#else
  :
#endif
  dr_vector<T,Alloc,L,Access>::base_type(s,lambda)
{
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < class Lambda >
#else
template < class Lambda, typename >
#endif
constexpr dr_vector<T,Alloc,L,Access>::dr_vector( extents_type s, extents_type cap, Lambda&& lambda )
  noexcept( noexcept( dr_vector<T,Alloc,L,Access>::base_type(s,cap,lambda) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_default_constructible_v<typename dr_vector<T,Alloc,L,Access>::allocator_type> &&
           requires { { ::std::declval<Lambda&&>()( ::std::declval<typename dr_vector<T,Alloc,L,Access>::index_type>() ) }
                      -> ::std::convertible_to<typename dr_vector<T,Alloc,L,Access>::element_type>; } :
#else
  :
#endif
  dr_vector<T,Alloc,L,Access>::base_type(s,cap,lambda)
{
}

template < class T, class Alloc, class L, class Access >
constexpr dr_vector<T,Alloc,L,Access>::dr_vector( const allocator_type& alloc )
  noexcept( noexcept( dr_vector<T,Alloc,L,Access>::base_type(alloc) ) ) :
  dr_vector<T,Alloc,L,Access>::base_type(alloc)
{
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::view_may_be_constructible_to_tensor< dr_vector<T,Alloc,L,Access> > MDS >
#else
template < class MDS, typename >
#endif
constexpr dr_vector<T,Alloc,L,Access>::dr_vector( const MDS& view, const allocator_type& alloc )
  noexcept( noexcept( dr_vector<T,Alloc,L,Access>::base_type(view,alloc) ) ) :
  dr_vector<T,Alloc,L,Access>::base_type(view,alloc)
{
}

template < class T, class Alloc, class L, class Access >
constexpr dr_vector<T,Alloc,L,Access>::dr_vector( extents_type s, const allocator_type& alloc )
  noexcept( noexcept( dr_vector<T,Alloc,L,Access>::base_type(s,alloc) ) ) :
  dr_vector<T,Alloc,L,Access>::base_type(s,alloc)
{
}

template < class T, class Alloc, class L, class Access >
constexpr dr_vector<T,Alloc,L,Access>::dr_vector( extents_type s, extents_type cap, const allocator_type& alloc )
  noexcept( noexcept( dr_vector<T,Alloc,L,Access>::base_type(s,cap,alloc) ) ) :
  dr_vector<T,Alloc,L,Access>::base_type(s,cap,alloc)
{
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < class Lambda >
#else
template < class Lambda, typename >
#endif
constexpr dr_vector<T,Alloc,L,Access>::dr_vector( extents_type s, Lambda&& lambda, const allocator_type& alloc )
  noexcept( noexcept( dr_vector<T,Alloc,L,Access>::base_type(s,lambda,alloc) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires { { ::std::declval<Lambda&&>()( ::std::declval<typename dr_vector<T,Alloc,L,Access>::index_type>() ) }
                    -> ::std::convertible_to<typename dr_vector<T,Alloc,L,Access>::element_type>; } :
#else
  :
#endif
  dr_vector<T,Alloc,L,Access>::base_type(s,lambda,alloc)
{
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < class Lambda >
#else
template < class Lambda, typename >
#endif
constexpr dr_vector<T,Alloc,L,Access>::dr_vector( extents_type s, extents_type cap, Lambda&& lambda, const allocator_type& alloc )
  noexcept( noexcept( dr_vector<T,Alloc,L,Access>::base_type(s,cap,lambda,alloc) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires { { ::std::declval<Lambda&&>()( ::std::declval<typename dr_vector<T,Alloc,L,Access>::index_type>() ) }
                      -> ::std::convertible_to<typename dr_vector<T,Alloc,L,Access>::element_type>; } :
#else
  :
#endif
  dr_vector<T,Alloc,L,Access>::base_type(s,cap,lambda,alloc)
{
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_may_be_constructible< dr_vector<T,Alloc,L,Access> > V2 >
#else
template < class V2, typename >
#endif
constexpr dr_vector<T,Alloc,L,Access>& dr_vector<T,Alloc,L,Access>::operator = ( const V2& rhs )
  noexcept( noexcept( ::std::declval<typename dr_vector<T,Alloc,L,Access>::base_type>() = rhs ) )
{
  static_cast<void>( this->base_type::operator=(rhs) );
  return *this;
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::view_may_be_constructible_to_tensor< dr_vector<T,Alloc,L,Access> > MDS >
#else
template < class MDS, typename, typename >
#endif
constexpr dr_vector<T,Alloc,L,Access>& dr_vector<T,Alloc,L,Access>::operator = ( const MDS& view )
  noexcept( noexcept( ::std::declval<typename dr_vector<T,Alloc,L,Access>::base_type>() = view ) )
{
  static_cast<void>( this->base_type::operator=(view) );
  return *this;
}

//- Const views

template < class T, class Alloc, class L, class Access >
[[nodiscard]] constexpr typename dr_vector<T,Alloc,L,Access>::const_subvector_type
dr_vector<T,Alloc,L,Access>::subvector( index_type start,
                                        index_type end ) const
{
  return const_subvector_type { ::std::experimental::submdspan( this->underlying_span(), ::std::tuple( start, end ) ) };
}

//- Mutable views

template < class T, class Alloc, class L, class Access >
[[nodiscard]] constexpr typename dr_vector<T,Alloc,L,Access>::subvector_type
dr_vector<T,Alloc,L,Access>::subvector( index_type start,
                                        index_type end )
{
  return subvector_type { ::std::experimental::submdspan( this->underlying_span(), ::std::tuple( start, end ) ) };
}

}       //- math namespace
}       //- experimental namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_FIXED_SIZE_MATRIX_ENGINE_HPP
