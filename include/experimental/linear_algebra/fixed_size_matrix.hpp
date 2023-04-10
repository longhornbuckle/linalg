//==================================================================================================
//  File:       fixed_size_matrix.hpp
//
//  Summary:    This header defines a fixed-size matrix.  In this context, fixed-size
//              means that the row and column extents of such objects are known at compile-time.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_FIXED_SIZE_MATRIX_HPP
#define LINEAR_ALGEBRA_FIXED_SIZE_MATRIX_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{

/// @brief Fixed-size, fixed-capacity matrix.
//         Implementation satisfies the following concepts:
//         concepts::fixed_size_matrix
//         concepts::writable_matrix
/// @tparam T element_type
/// @tparam R number of rows
/// @tparam C number of columns
/// @tparam L layout defines the ordering of elements in memory
/// @tparam A accessor policy defines how elements are accessed
template < class  T,
           size_t R,
           size_t C,
           class  L = experimental::layout_right,
           class  A = experimental::default_accessor<T>
#ifdef LINALG_ENABLE_CONCEPTS
           > requires ( ( R >= 0 ) && ( C >= 0 ) ) // Row and column must be >= 0
#else
           , typename = enable_if_t< ( ( R >= 0 ) && ( C >= 0 ) ) > >
#endif
class fs_matrix : public fs_tensor<T,L,A,R,C>
{
  private:
    // Base tensor type
    using base_type = fs_tensor<T,L,A,R,C>;
  public:
    //- Types

    /// @brief Type used to define memory layout
    using layout_type                = typename base_type::layout_type;
    /// @brief Type used to define access into memory
    using accessor_type              = typename base_type::accessor_type;
    /// @brief Type contained by the matrix
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
    /// @brief mutable view of a column vector
    using column_type                = vector_view<decltype( experimental::submdspan( declval<underlying_span_type>(), declval<experimental::full_extent_t>(), declval<index_type>() ) )>;
    /// @brief const view of a column vector
    using const_column_type          = vector_view<decltype( experimental::submdspan( declval<const_underlying_span_type>(), declval<experimental::full_extent_t>(), declval<index_type>() ) )>;
    /// @brief mutable view of a row vector
    using row_type                   = vector_view<decltype( experimental::submdspan( declval<underlying_span_type>(), declval<index_type>(), declval<experimental::full_extent_t>() ) )>;
    /// @brief const view of a row vector
    using const_row_type             = vector_view<decltype( experimental::submdspan( declval<const_underlying_span_type>(), declval<index_type>(), declval<experimental::full_extent_t>() ) )>;
    /// @brief mutable view of a submatrix
    using submatrix_type             = matrix_view<decltype( detail::submdspan( declval<underlying_span_type>(), declval<tuple_type>(), declval<tuple_type>() ) )>;
    /// @brief const view of a submatrix
    using const_submatrix_type       = matrix_view<decltype( detail::submdspan( declval<const_underlying_span_type>(), declval<tuple_type>(), declval<tuple_type>() ) )>;
    /// @brief matrix tanspose
    using transpose_type             = fs_matrix<T,C,R,L,A>;

    //- Rebind

    /// @brief Rebind defines a type for a rebinding a fixed size matrix to the new type parameters
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
      using type = fs_matrix< ValueType,
                              R,
                              C,
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
    ~fs_matrix()                  = default;
    /// @brief Default constructor
    fs_matrix()                   = default;
    /// @brief Default move constructor
    /// @param fs_matrix to be moved
    fs_matrix( fs_matrix&& )      = default;
    /// @brief Default copy constructor
    /// @param fs_matrix to be copied
    fs_matrix( const fs_matrix& ) = default;
    /// @brief Template copy constructor
    /// @tparam matrix to be copied
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::tensor_may_be_constructible< fs_matrix > M2 >
    #else
    template < class M2, typename = enable_if_t< concepts::tensor_may_be_constructible< M2, fs_matrix > > >
    #endif
    explicit constexpr fs_matrix( const M2& rhs ) noexcept( noexcept( base_type(rhs) ) );
    /// @brief Construct from a view
    /// @tparam A two dimensional view
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< fs_matrix > MDS >
    #else
    template < class MDS, typename = enable_if_t< concepts::view_may_be_constructible_to_tensor<MDS,fs_matrix> && is_default_constructible_v<allocator_type> > >
    #endif
    explicit constexpr fs_matrix( const MDS& view ) noexcept( noexcept( base_type(view) ) );
    /// @brief Construct by applying lambda to every element in the matrix
    /// @tparam lambda lambda expression encapsulating operation to be performed on each element
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = enable_if_t< is_default_constructible_v<allocator_type> &&
                                       is_convertible_to< decltype( declval<Lambda&&>()( declval<index_type>(), declval<index_type>() ) ), element_type > > >
    #endif
    explicit constexpr fs_matrix( Lambda&& lambda ) noexcept( noexcept( declval<Lambda&&>()( declval<index_type>(), declval<index_type>() ) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires requires { { declval<Lambda&&>()( declval<index_type>(), declval<index_type>() ) } -> convertible_to<element_type>; };
    #else
      ;
    #endif
    /// @brief Default move constructor
    /// @param  fs_matrix to be moved
    /// @return self
    fs_matrix& operator = ( fs_matrix&& )      = default;
    /// @brief Default copy constructor
    /// @param  fs_matrix to be copied
    /// @return self
    fs_matrix& operator = ( const fs_matrix& ) = default;
    /// @brief Template copy constructor
    /// @tparam type of matrix to be copied
    /// @param  matrix to be copied
    /// @returns self
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::tensor_may_be_constructible< fs_matrix > M2 >
    #else
    template < class M2, typename = enable_if_t< concepts::tensor_may_be_constructible< M2, fs_matrix > > >
    #endif
    constexpr fs_matrix& operator = ( const M2& rhs ) noexcept( noexcept( declval<base_type>() = rhs ) );
    /// @brief Construct from a two dimensional view
    /// @tparam type of view to be copied
    /// @param  view to be copied
    /// @returns self
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< fs_matrix > MDS >
    #else
    template < class MDS, typename = enable_if_t< concepts::view_may_be_constructible_to_tensor<MDS,fs_matrix> > >
    #endif
    constexpr fs_matrix& operator = ( const MDS& view ) noexcept( noexcept( declval<base_type>() = view ) );

    //- Size / Capacity

    using base_type::size;
    using base_type::capacity;

    /// @brief Returns current number of columns
    /// @return number of columns
    [[nodiscard]] constexpr size_type  columns() const noexcept;
    /// @brief Returns the current number of rows
    /// @return number of rows
    [[nodiscard]] constexpr size_type  rows() const noexcept;
    /// @brief Returns the current column capacity
    /// @return column capacity
    [[nodiscard]] constexpr size_type  column_capacity() const noexcept;
    /// @brief Returns the current row capacity
    /// @return row capacity
    [[nodiscard]] constexpr size_type  row_capacity() const noexcept;

    //- Const views

    #if LINALG_USE_BRACKET_OPERATOR
    using base_type::operator[]; // Brings into scope const and mutable
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    using base_type::operator(); // Brings into scope const and mutable
    #endif
    using base_type::at;         // Brings into scope const and mutable

    /// @brief Returns a const view of the specified column
    /// @param j column
    /// @returns const view of column
    [[nodiscard]] constexpr const_column_type column( index_type j ) const;
    /// @brief Returns a const view of the specified row
    /// @param i row
    /// @returns const view of row
    [[nodiscard]] constexpr const_row_type row( index_type i ) const;
    /// @brief Returns a const view of the specified submatrix
    /// @param start (row,column) start of submatrix
    /// @param end (row,column) end of submatrix
    /// @returns const view of the specified submatrix
    [[nodiscard]] constexpr const_submatrix_type submatrix( tuple_type start,
                                                            tuple_type end ) const;

    //- Mutable views

    /// @brief Returns a mutable view of the specified column
    /// @param j column
    /// @returns mutable view of column
    [[nodiscard]] constexpr column_type column( index_type j );
    /// @brief Returns a mutable view of the specified row
    /// @param i row
    /// @returns mutable view of row
    [[nodiscard]] constexpr row_type row( index_type i );
    /// @brief Returns a mutable view of the specified submatrix
    /// @param start (row,column) start of submatrix
    /// @param end (row,column) end of submatrix
    /// @returns mutable view of the specified submatrix
    [[nodiscard]] constexpr submatrix_type submatrix( tuple_type start,
                                                      tuple_type end );

    //- Data access

    using base_type::span;
    using base_type::underlying_span;

};

//----------------------------------------------
// Implementation of fs_matrix<T,R,C,L,A>
//----------------------------------------------

//- Destructor / Constructors / Assignments

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
template < concepts::tensor_may_be_constructible< fs_matrix<T,R,C,L,A> > M2 >
constexpr fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A, typename Dummy >
template < class M2, typename >
constexpr fs_matrix<T,R,C,L,A,Dummy>::
#endif
fs_matrix( const M2& rhs )
#ifdef LINALG_ENABLE_CONCEPTS
  noexcept( noexcept( fs_matrix<T,R,C,L,A>::base_type( rhs ) ) ) :
  fs_matrix<T,R,C,L,A>::base_type( rhs )
#else
  noexcept( noexcept( fs_matrix<T,R,C,L,A,Dummy>::base_type( rhs ) ) ) :
  fs_matrix<T,R,C,L,A,Dummy>::base_type( rhs )
#endif
{
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
template < concepts::view_may_be_constructible_to_tensor< fs_matrix<T,R,C,L,A> > MDS >
constexpr fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A , typename Dummy >
template < class MDS, typename >
constexpr fs_matrix<T,R,C,L,A,Dummy>::
#endif
fs_matrix( const MDS& view )
#ifdef LINALG_ENABLE_CONCEPTS
  noexcept( noexcept( fs_matrix<T,R,C,L,A>::base_type( view ) ) ) :
  fs_matrix<T,R,C,L,A>::base_type( view )
#else
  noexcept( noexcept( fs_matrix<T,R,C,L,A,Dummy>::base_type( view ) ) ) :
  fs_matrix<T,R,C,L,A,Dummy>::base_type( view )
#endif
{
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
template < class Lambda >
constexpr fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A , typename Dummy >
template < class Lambda, typename >
constexpr fs_matrix<T,R,C,L,A,Dummy>::
#endif
fs_matrix( Lambda&& lambda ) noexcept( noexcept( declval<Lambda&&>()( declval<index_type>(), declval<index_type>() ) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires { { declval<Lambda&&>()( declval<index_type>(), declval<index_type>() ) } -> convertible_to<element_type>; } :
#else
  :
#endif
  fs_matrix<T,R,C,L,A>::base_type( lambda )
{
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
template < concepts::view_may_be_constructible_to_tensor< fs_matrix<T,R,C,L,A> > MDS >
constexpr fs_matrix<T,R,C,L,A>& fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A , typename Dummy >
template < class M2, typename >
constexpr fs_matrix<T,R,C,L,A,Dummy>& fs_matrix<T,R,C,L,A,Dummy>::
#endif
operator = ( const M2& rhs )
  noexcept( noexcept( declval<base_type>() = rhs ) )
{
  static_cast<void>( this->base_type::operator=(rhs) );
  return *this;
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
template < concepts::view_may_be_constructible_to_tensor< fs_matrix<T,R,C,L,A> > MDS >
constexpr fs_matrix<T,R,C,L,A>& fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A , typename Dummy >
template < class MDS, typename >
constexpr fs_matrix<T,R,C,L,A,Dummy>& fs_matrix<T,R,C,L,A,Dummy>::
#endif
operator = ( const MDS& view )
  noexcept( noexcept( declval<base_type>() = view ) )
{
  static_cast<void>( this->base_type::operator=(view) );
  return *this;
}

//- Size / Capacity

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A>::size_type fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A, typename Dummy >
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A,Dummy>::size_type fs_matrix<T,R,C,L,A,Dummy>::
#endif
columns() const noexcept
{
  return C;
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A>::size_type fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A, typename Dummy >
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A,Dummy>::size_type fs_matrix<T,R,C,L,A,Dummy>::
#endif
rows() const noexcept
{
  return R;
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A>::size_type fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A, typename Dummy >
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A,Dummy>::size_type fs_matrix<T,R,C,L,A,Dummy>::
#endif
column_capacity() const noexcept
{
  return this->columns();
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A>::size_type fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A, typename Dummy >
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A,Dummy>::size_type fs_matrix<T,R,C,L,A,Dummy>::
#endif
row_capacity() const noexcept
{
  return this->rows();
}

//- Const views

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A>::const_column_type fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A, typename Dummy >
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A,Dummy>::const_column_type fs_matrix<T,R,C,L,A,Dummy>::
#endif
column( typename fs_matrix<T,R,C,L,A>::index_type j ) const
{
  return const_column_type { experimental::submdspan( this->underlying_span(), experimental::full_extent, j ) };
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A>::const_row_type fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A, typename Dummy >
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A,Dummy>::const_row_type fs_matrix<T,R,C,L,A,Dummy>::
#endif
row( typename fs_matrix<T,R,C,L,A>::index_type i ) const
{
  return const_row_type { experimental::submdspan( this->underlying_span(), i, experimental::full_extent ) };
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A>::const_submatrix_type fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A, typename Dummy >
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A,Dummy>::const_submatrix_type fs_matrix<T,R,C,L,A,Dummy>::
#endif
submatrix( tuple_type start,
           tuple_type end ) const
{
  return const_submatrix_type { detail::submdspan( this->underlying_span(), start, end ) };
}

//- Mutable views

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A>::column_type fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A, typename Dummy >
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A,Dummy>::column_type fs_matrix<T,R,C,L,A,Dummy>::
#endif
column( typename fs_matrix<T,R,C,L,A>::index_type j )
{
  return column_type { experimental::submdspan( this->underlying_span(), experimental::full_extent, j ) };
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A>::row_type fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A, typename Dummy >
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A,Dummy>::row_type fs_matrix<T,R,C,L,A,Dummy>::
#endif
row( typename fs_matrix<T,R,C,L,A>::index_type i )
{
  return row_type { experimental::submdspan( this->underlying_span(), i, experimental::full_extent ) };
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class T, size_t R, size_t C, class L, class A > requires ( ( R >= 0 ) && ( C >= 0 ) )
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A>::submatrix_type fs_matrix<T,R,C,L,A>::
#else
template < class T, size_t R, size_t C, class L, class A, typename Dummy >
[[nodiscard]] constexpr typename fs_matrix<T,R,C,L,A,Dummy>::submatrix_type fs_matrix<T,R,C,L,A,Dummy>::
#endif
submatrix( tuple_type start,
           tuple_type end )
{
  return submatrix_type { detail::submdspan( this->underlying_span(), start, end ) };
}

}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_FIXED_SIZE_MATRIX_HPP
