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
           class  A = experimental::default_accessor<T> > requires 
  ( N >= 0 ) // Number of elements must be >= 0
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
    ~fs_vector()                         = default;
    /// @brief Default constructor
    fs_vector()                          = default;
    /// @brief Default move constructor
    /// @param fs_vector to be moved
    fs_vector( fs_vector&& )      = default;
    /// @brief Default copy constructor
    /// @param fs_vector to be copied
    fs_vector( const fs_vector& ) = default;
    /// @brief Template copy constructor
    /// @tparam vector to be copied
    template < concepts::tensor_may_be_constructible< fs_vector > T2 >
    explicit constexpr fs_vector( const T2& rhs ) noexcept( noexcept( base_type(rhs) ) );
    /// @brief Construct from a view
    /// @tparam A two dimensional view
    template < concepts::view_may_be_constructible_to_tensor< fs_vector > MDS >
    explicit constexpr fs_vector( const MDS& view ) noexcept( noexcept( base_type(view) ) );
    /// @brief Construct by applying lambda to every element in the vector
    /// @tparam lambda lambda expression encapsulating operation to be performed on each element
    template < class Lambda >
    explicit constexpr fs_vector( Lambda&& lambda ) noexcept( noexcept( declval<Lambda&&>()( declval<index_type>() ) ) )
      requires requires { { declval<Lambda&&>()( declval<index_type>() ) } -> convertible_to<element_type>; };
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
    template < concepts::tensor_may_be_constructible< fs_vector > T2 >
    constexpr fs_vector& operator = ( const T2& rhs ) noexcept( noexcept( declval<base_type>() = rhs ) );
    /// @brief Construct from a one dimensional view
    /// @tparam type of view to be copied
    /// @param  view to be copied
    /// @returns self
    template < concepts::view_may_be_constructible_to_tensor< fs_vector > MDS >
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

template < class T, size_t N, class L, class A > requires ( N >= 0 )
template < concepts::tensor_may_be_constructible< fs_vector<T,N,L,A> > T2 >
constexpr fs_vector<T,N,L,A>::fs_vector( const T2& rhs )
  noexcept( noexcept( fs_vector<T,N,L,A>::base_type( rhs ) ) ) :
  fs_vector<T,N,L,A>::base_type( rhs )
{
}

template < class T, size_t N, class L, class A > requires ( N >= 0 )
template < concepts::view_may_be_constructible_to_tensor< fs_vector<T,N,L,A> > MDS >
constexpr fs_vector<T,N,L,A>::fs_vector( const MDS& view )
  noexcept( noexcept( fs_vector<T,N,L,A>::base_type( view ) ) ) :
  fs_vector<T,N,L,A>::base_type( view )
{
}

template < class T, size_t N, class L, class A > requires ( N >= 0 )
template < class Lambda >
constexpr fs_vector<T,N,L,A>::fs_vector( Lambda&& lambda ) noexcept( noexcept( declval<Lambda&&>()( declval<index_type>() ) ) )
  requires requires { { declval<Lambda&&>()( declval<index_type>() ) } -> convertible_to<element_type>; } :
  fs_vector<T,N,L,A>::base_type( lambda )
{
}

template < class T, size_t N, class L, class A > requires ( N >= 0 )
template < concepts::tensor_may_be_constructible< fs_vector<T,N,L,A> > T2 >
constexpr fs_vector<T,N,L,A>& fs_vector<T,N,L,A>::operator = ( const T2& rhs )
  noexcept( noexcept( declval<typename fs_vector<T,N,L,A>::base_type>() = rhs ) )
{
  static_cast<void>( this->base_type::operator=(rhs) );
  return *this;
}

template < class T, size_t N, class L, class A > requires ( N >= 0 )
template < concepts::view_may_be_constructible_to_tensor< fs_vector<T,N,L,A> > MDS >
constexpr fs_vector<T,N,L,A>& fs_vector<T,N,L,A>::operator = ( const MDS& view )
  noexcept( noexcept( declval<typename fs_vector<T,N,L,A>::base_type>() = view ) )
{
  static_cast<void>( this->base_type::operator=(view) );
  return *this;
}

//- Const views

template < class T, size_t N, class L, class A > requires ( N >= 0 )
[[nodiscard]] constexpr typename fs_vector<T,N,L,A>::const_subvector_type
fs_vector<T,N,L,A>::subvector( index_type start,
                                      index_type end ) const
{
  return const_subvector_type { experimental::submdspan( this->underlying_span(), tuple( start, end ) ) };
}

//- Mutable views

template < class T, size_t N, class L, class A > requires ( N >= 0 )
[[nodiscard]] constexpr typename fs_vector<T,N,L,A>::subvector_type
fs_vector<T,N,L,A>::subvector( index_type start,
                                      index_type end )
{
  return subvector_type { experimental::submdspan( this->underlying_span(), tuple( start, end ) ) };
}

}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_FIXED_SIZE_MATRIX_HPP
