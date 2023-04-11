//==================================================================================================
//  File:       fixed_size_vector.hpp
//
//  Summary:    This header defines a fixed-size vector.  In this context, fixed-size
//              means the length of such objects are known at compile-time.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_FIXED_SIZE_VECTOR_HPP
#define LINEAR_ALGEBRA_FIXED_SIZE_VECTOR_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{

/// @brief Fixed-size, fixed-capacity vector.
//         Implementation satisfies the following concepts:
//         concepts::fixed_size_vector
//         concepts::writable_vector
/// @tparam T element_type
/// @tparam N number of elements
/// @tparam L layout defines the ordering of elements in memory
/// @tparam A accessor policy defines how elements are accessed
template < class  T,
           size_t N,
           class  L = experimental::layout_right,
           class  A = experimental::default_accessor<T>
#ifdef LINALG_ENABLE_CONCEPTS
           > requires ( N >= 0 ) // Number of elements must be >= 0
#else
           , typename = enable_if_t< ( N >= 0 ) > >
#endif
class fs_vector : public fs_tensor<T,L,A,N>
{
  private:
    // Base tensor type
    using base_type = fs_tensor<T,L,A,N>;
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
    using reference_type             = typename base_type::reference_type;
    /// @brief mutable view of a subtensor
    using subvector_type             = vector_view<decltype( experimental::submdspan( declval<underlying_span_type>(), declval< tuple<index_type,index_type> >() ) ) >;
    /// @brief const view of a subtensor
    using const_subvector_type       = vector_view<decltype( experimental::submdspan( declval<const_underlying_span_type>(), declval< tuple<index_type,index_type> >() ) ) >;

    //- Rebind

    /// @brief Rebind defines a type for a rebinding a fixed size vector to the new type parameters
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
      using type = fs_vector< ValueType,
                              N,
                              LayoutType,
                              rebind_accessor_type >;
    };
    /// @brief Helper for defining rebound type
    /// @tparam ValueType  rebound value type
    /// @tparam LayoutType rebound layout policy
    /// @tparam AccessType rebound access policy
    template < class ValueType,
               class LayoutType   = layout_type,
               class AccessorType = accessor_type >
    using rebind_t = typename rebind< ValueType, LayoutType, AccessorType >::type;

    //- Destructor / Constructors / Assignments

    /// @brief Default destructor
    ~fs_vector()                  = default;
    /// @brief Default constructor
    fs_vector()                   = default;
    /// @brief Default move constructor
    /// @param fs_vector to be moved
    fs_vector( fs_vector&& )      = default;
    /// @brief Default copy constructor
    /// @param fs_vector to be copied
    fs_vector( const fs_vector& ) = default;
    /// @brief Template copy constructor
    /// @tparam vector to be copied
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::tensor_may_be_constructible< fs_vector > V2 >
    #else
    template < class V2, typename = enable_if_t< concepts::tensor_may_be_constructible< V2, fs_vector > > >
    #endif
    explicit constexpr fs_vector( const V2& rhs ) noexcept( noexcept( base_type(rhs) ) );
    /// @brief Construct from a view
    /// @tparam A two dimensional view
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< fs_vector > MDS >
    #else
    template < class MDS, typename = enable_if_t< concepts::view_may_be_constructible_to_tensor<MDS,fs_vector> >, typename = enable_if_t<true> >
    #endif
    explicit constexpr fs_vector( const MDS& view ) noexcept( noexcept( base_type(view) ) );
    /// @brief Construct by applying lambda to every element in the vector
    /// @tparam lambda lambda expression encapsulating operation to be performed on each element
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = enable_if_t< is_convertible_v< decltype( declval<Lambda&&>()( declval<index_type>() ) ), element_type > > >
    #endif
    explicit constexpr fs_vector( Lambda&& lambda ) noexcept( noexcept( declval<Lambda&&>()( declval<index_type>() ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires requires { { declval<Lambda&&>()( declval<index_type>() ) } -> convertible_to<element_type>; };
    #else
      ;
    #endif
    /// @brief Default move constructor
    /// @param  fs_vector to be moved
    /// @return self
    fs_vector& operator = ( fs_vector&& )      = default;
    /// @brief Default copy constructor
    /// @param  fs_vector to be copied
    /// @return self
    fs_vector& operator = ( const fs_vector& ) = default;
    /// @brief Template copy constructor
    /// @tparam type of vector to be copied
    /// @param  vector to be copied
    /// @returns self
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::tensor_may_be_constructible< fs_vector > V2 >
    #else
    template < class V2, typename = enable_if_t< concepts::tensor_may_be_constructible< V2, fs_vector > > >
    #endif
    constexpr fs_vector& operator = ( const V2& rhs ) noexcept( noexcept( declval<base_type>() = rhs ) );
    /// @brief Construct from a one dimensional view
    /// @tparam type of view to be copied
    /// @param  view to be copied
    /// @returns self
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< fs_vector > MDS >
    #else
    template < class MDS, typename = enable_if_t< concepts::view_may_be_constructible_to_tensor<MDS,fs_vector> >, typename = enable_if_t<true> >
    #endif
    constexpr fs_vector& operator = ( const MDS& view ) noexcept( noexcept( declval<base_type>() = view ) );

    //- Size / Capacity

    using base_type::size;
    using base_type::capacity;

    //- Const views

    #if LINALG_USE_BRACKET_OPERATOR
    using base_type::operator[]; // Brings into scope const and mutable
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    using base_type::operator(); // Brings into scope const and mutable
    #endif
    using base_type::at;         // Brings into scope const and mutable

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
// Implementation of fs_vector<T,R,C,L,A>
//----------------------------------------------

//- Destructor / Constructors / Assignments

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t N, class L, class A > requires ( N >= 0 )
template < concepts::tensor_may_be_constructible< fs_vector<T,N,L,A> > V2 >
constexpr fs_vector<T,N,L,A>::
#else
template < class T, size_t N, class L, class A, typename Dummy >
template < class V2, typename >
constexpr fs_vector<T,N,L,A,Dummy>::
#endif
fs_vector( const V2& rhs )
#ifdef LINALG_ENABLE_CONCEPTS
  noexcept( noexcept( fs_vector<T,N,L,A>::base_type( rhs ) ) ) :
  fs_vector<T,N,L,A>::base_type( rhs )
#else
  noexcept( noexcept( fs_vector<T,N,L,A,Dummy>::base_type( rhs ) ) ) :
  fs_vector<T,N,L,A,Dummy>::base_type( rhs )
#endif
{
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t N, class L, class A > requires ( N >= 0 )
template < concepts::view_may_be_constructible_to_tensor< fs_vector<T,N,L,A> > MDS >
constexpr fs_vector<T,N,L,A>::
#else
template < class T, size_t N, class L, class A, typename Dummy >
template < class MDS, typename, typename >
constexpr fs_vector<T,N,L,A,Dummy>::
#endif
#ifdef LINALG_ENABLE_CONCEPTS
#else
#endif
fs_vector( const MDS& view )
#ifdef LINALG_ENABLE_CONCEPTS
  noexcept( noexcept( fs_vector<T,N,L,A>::base_type( view ) ) ) :
  fs_vector<T,N,L,A>::base_type( view )
#else
  noexcept( noexcept( fs_vector<T,N,L,A,Dummy>::base_type( view ) ) ) :
  fs_vector<T,N,L,A,Dummy>::base_type( view )
#endif
{
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t N, class L, class A > requires ( N >= 0 )
template < class Lambda >
constexpr fs_vector<T,N,L,A>::
#else
template < class T, size_t N, class L, class A, typename Dummy >
template < class Lambda, typename >
constexpr fs_vector<T,N,L,A,Dummy>::
#endif
fs_vector( Lambda&& lambda )
#ifdef LINALG_ENABLE_CONCEPTS
  noexcept( noexcept( declval<Lambda&&>()( declval<typename fs_vector<T,N,L,A>::index_type>() ) ) )
  requires requires { { declval<Lambda&&>()( declval<typename fs_vector<T,N,L,A>::index_type>() ) } -> convertible_to<typename fs_vector<T,N,L,A>::element_type>; } :
  fs_vector<T,N,L,A>::base_type( lambda )
#else
  noexcept( noexcept( declval<Lambda&&>()( declval<typename fs_vector<T,N,L,A,Dummy>::index_type>() ) ) ) :
  fs_vector<T,N,L,A,Dummy>::base_type( lambda )
#endif
{
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t N, class L, class A > requires ( N >= 0 )
template < concepts::tensor_may_be_constructible< fs_vector<T,N,L,A> > V2 >
constexpr fs_vector<T,N,L,A>& fs_vector<T,N,L,A>::
#else
template < class T, size_t N, class L, class A, typename Dummy >
template < class V2, typename >
constexpr fs_vector<T,N,L,A,Dummy>& fs_vector<T,N,L,A,Dummy>::
#endif
operator = ( const V2& rhs )
#ifdef LINALG_ENABLE_CONCEPTS
  noexcept( noexcept( declval<typename fs_vector<T,N,L,A>::base_type>() = rhs ) )
#else
  noexcept( noexcept( declval<typename fs_vector<T,N,L,A,Dummy>::base_type>() = rhs ) )
#endif
{
  static_cast<void>( this->base_type::operator=(rhs) );
  return *this;
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t N, class L, class A > requires ( N >= 0 )
template < concepts::view_may_be_constructible_to_tensor< fs_vector<T,N,L,A> > MDS >
constexpr fs_vector<T,N,L,A>& fs_vector<T,N,L,A>::
#else
template < class T, size_t N, class L, class A, typename Dummy >
template < class MDS, typename, typename >
constexpr fs_vector<T,N,L,A,Dummy>& fs_vector<T,N,L,A,Dummy>::
#endif
operator = ( const MDS& view )
#ifdef LINALG_ENABLE_CONCEPTS
  noexcept( noexcept( declval<typename fs_vector<T,N,L,A>::base_type>() = view ) )
#else
  noexcept( noexcept( declval<typename fs_vector<T,N,L,A,Dummy>::base_type>() = view ) )
#endif
{
  static_cast<void>( this->base_type::operator=(view) );
  return *this;
}

//- Const views

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t N, class L, class A > requires ( N >= 0 )
[[nodiscard]] constexpr typename fs_vector<T,N,L,A>::const_subvector_type fs_vector<T,N,L,A>::
#else
template < class T, size_t N, class L, class A, typename Dummy >
[[nodiscard]] constexpr typename fs_vector<T,N,L,A,Dummy>::const_subvector_type fs_vector<T,N,L,A,Dummy>::
#endif
subvector( index_type start,
           index_type end ) const
{
  return const_subvector_type { experimental::submdspan( this->underlying_span(), tuple( start, end ) ) };
}

//- Mutable views

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t N, class L, class A > requires ( N >= 0 )
[[nodiscard]] constexpr typename fs_vector<T,N,L,A>::subvector_type fs_vector<T,N,L,A>::
#else
template < class T, size_t N, class L, class A, typename Dummy >
[[nodiscard]] constexpr typename fs_vector<T,N,L,A,Dummy>::subvector_type fs_vector<T,N,L,A,Dummy>::
#endif
subvector( index_type start,
           index_type end )
{
  return subvector_type { experimental::submdspan( this->underlying_span(), tuple( start, end ) ) };
}

}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_FIXED_SIZE_MATRIX_HPP
