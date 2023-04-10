//==================================================================================================
//  File:       matrix_view.hpp
//
//  Summary:    This header defines a matrix_view, which is a non-owning view into a larger
//              matrix.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_MATRIX_VIEW_HPP
#define LINEAR_ALGEBRA_MATRIX_VIEW_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{

/// @brief Matrix view.
//         Implementation satisfies the following concepts:
//         concepts::matrix
//         concepts::writable_matrix if MDS::element_type is non-const
//         concepts::readable_matrix if MDS::element_type is const
/// @tparam MDS mdspan
template < class MDS
#ifdef LINALG_ENABLE_CONCEPTS
  > requires ( detail::is_mdspan_v<MDS> &&
               ( MDS::extents_type::rank()== 2 ) &&
               MDS::is_always_unique() ) // Each element in the mdspan must have a unique mapping. (i.e. span_type and const_underlying_span_type should be the same.)
#else
  , typename = enable_if_t< ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank()== 2 ) && MDS::is_always_unique() ) > >
#endif
class matrix_view : public tensor_view<MDS>
{
  private:
    // Base tensor type
    using base_type = tensor_view<MDS>;
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
    /// @brief mutable view of a subtensor
    using subtensor_type             = typename base_type::subtensor_type;
    /// @brief const view of a subtensor
    using const_subtensor_type       = typename base_type::const_subtensor_type;
    /// @brief mutable view of a submatrix
    using submatrix_type             = matrix_view<decltype( detail::submdspan( declval<underlying_span_type>(), declval<tuple_type>(), declval<tuple_type>() ) ) >;
    /// @brief const view of a submatrix
    using const_submatrix_type       = matrix_view<decltype( detail::submdspan( declval<const_underlying_span_type>(), declval<tuple_type>(), declval<tuple_type>() ) ) >;
    /// @brief mutable view of a column vector
    using column_type                = vector_view<decltype( experimental::submdspan( declval<underlying_span_type>(), declval<experimental::full_extent_t>(), declval<index_type>() ) )>;
    /// @brief const view of a column vector
    using const_column_type          = vector_view<decltype( experimental::submdspan( declval<const_underlying_span_type>(), declval<experimental::full_extent_t>(), declval<index_type>() ) )>;
    /// @brief mutable view of a row vector
    using row_type                   = vector_view<decltype( experimental::submdspan( declval<underlying_span_type>(), declval<index_type>(), declval<experimental::full_extent_t>() ) )>;
    /// @brief const view of a row vector
    using const_row_type             = vector_view<decltype( experimental::submdspan( declval<const_underlying_span_type>(), declval<index_type>(), declval<experimental::full_extent_t>() ) )>;
    
    //- Destructor / Constructors / Assignments

    /// @brief Default destructor
    ~matrix_view()                                     = default;
    /// @brief Default constructor
    matrix_view()                                      = default;
    /// @brief Default move constructor
    /// @param matrix_view to be moved
    matrix_view( matrix_view&& rhs )                   = default;
    /// @brief Default copy constructor
    /// @param matrix_view to be copied
    matrix_view( const matrix_view& rhs )              = default;
    /// @brief Construct from view
    /// @param view to be constructed from
    explicit constexpr matrix_view( const underlying_span_type& view ) noexcept;
    /// @brief Default move assignment
    /// @param  matrix_view to be moved
    /// @return self
    matrix_view& operator = ( matrix_view&& rhs )      = default;
    /// @brief Default copy assignment
    /// @param  matrix_view to be copied
    /// @return self
    matrix_view& operator = ( const matrix_view& rhs ) = default;
    /// @brief Assign from view
    /// @param  view to be assigned
    /// @return self
    constexpr matrix_view& operator = ( const underlying_span_type& view ) noexcept;

    //- Size / Capacity

    using base_type::size;
    using base_type::capacity;

    /// @brief Returns current number of columns
    /// @return number of columns
    [[nodiscard]] constexpr size_type columns() const noexcept;
    /// @brief Returns the current number of rows
    /// @return number of rows
    [[nodiscard]] constexpr size_type rows() const noexcept;
    /// @brief Returns the current column capacity
    /// @return column capacity
    [[nodiscard]] constexpr size_type column_capacity() const noexcept;
    /// @brief Returns the current row capacity
    /// @return row capacity
    [[nodiscard]] constexpr size_type row_capacity() const noexcept;

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
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename = enable_if_t< !is_const_v<element_type> > >
    #endif
    [[nodiscard]] constexpr column_type column( index_type j )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( !is_const_v<element_type> );
    #else
      ;
    #endif
    /// @brief Returns a mutable view of the specified row
    /// @param i row
    /// @returns mutable view of row
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename = enable_if_t< !is_const_v<element_type> > >
    #endif
    [[nodiscard]] constexpr row_type row( index_type i )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( !is_const_v<element_type> );
    #else
      ;
    #endif
    /// @brief Returns a mutable view of the specified submatrix
    /// @param start (row,column) start of submatrix
    /// @param end (row,column) end of submatrix
    /// @returns mutable view of the specified submatrix
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename = enable_if_t< !is_const_v<element_type> > >
    #endif
    [[nodiscard]] constexpr submatrix_type submatrix( tuple_type start,
                                                      tuple_type end )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( !is_const_v<element_type> );
    #else
      ;
    #endif

    //- Data access

    using base_type::span;
    using base_type::underlying_span;

};

//------------------------------------------
// Implementation of matrix_view<MDS>
//------------------------------------------

//- Destructor / Constructors / Assignments

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 2 ) && MDS::is_always_unique() )
constexpr matrix_view<MDS>::
#else
template < class MDS, typename Dummy >
constexpr matrix_view<MDS,Dummy>::
#endif
matrix_view( const underlying_span_type& view ) noexcept :
#ifdef LINALG_ENABLE_CONCEPTS
  matrix_view<MDS>::base_type( view )
#else
  matrix_view<MDS,Dummy>::base_type( view )
#endif
{
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 2 ) && MDS::is_always_unique() )
constexpr matrix_view<MDS>& matrix_view<MDS>::
#else
template < class MDS, typename Dummy >
constexpr matrix_view<MDS,Dummy>& matrix_view<MDS,Dummy>::
#endif
operator = ( const underlying_span_type& view ) noexcept
{
  static_cast<void>( this->base_type::operator=( view ) );
  return *this;
}

//- Size / Capacity

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 2 ) && MDS::is_always_unique() )
[[nodiscard]] constexpr typename matrix_view<MDS>::size_type matrix_view<MDS>::
#else
template < class MDS, typename Dummy >
[[nodiscard]] constexpr typename matrix_view<MDS,Dummy>::size_type matrix_view<MDS,Dummy>::
#endif
columns() const noexcept
{
  return this->size().extent(2);
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 2 ) && MDS::is_always_unique() )
[[nodiscard]] constexpr typename matrix_view<MDS>::size_type matrix_view<MDS>::
#else
template < class MDS, typename Dummy >
[[nodiscard]] constexpr typename matrix_view<MDS,Dummy>::size_type matrix_view<MDS,Dummy>::
#endif
rows() const noexcept
{
  return this->size().extent(1);
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 2 ) && MDS::is_always_unique() )
[[nodiscard]] constexpr typename matrix_view<MDS>::size_type matrix_view<MDS>::
#else
template < class MDS, typename Dummy >
[[nodiscard]] constexpr typename matrix_view<MDS,Dummy>::size_type matrix_view<MDS,Dummy>::
#endif
column_capacity() const noexcept
{
  return this->capacity.extent(2);
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 2 ) && MDS::is_always_unique() )
[[nodiscard]] constexpr typename matrix_view<MDS>::size_type matrix_view<MDS>::
#else
template < class MDS, typename Dummy >
[[nodiscard]] constexpr typename matrix_view<MDS,Dummy>::size_type matrix_view<MDS,Dummy>::
#endif
row_capacity() const noexcept
{
  return this->capacity.extent(1);
}

//- Const views

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 2 ) && MDS::is_always_unique() )
[[nodiscard]] constexpr typename matrix_view<MDS>::const_column_type matrix_view<MDS>::
#else
template < class MDS, typename Dummy >
[[nodiscard]] constexpr typename matrix_view<MDS,Dummy>::const_column_type matrix_view<MDS,Dummy>::
#endif
column( index_type j ) const
{
  return const_column_type { experimental::submdspan( this->underlying_span(), experimental::full_extent, j ) };
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 2 ) && MDS::is_always_unique() )
[[nodiscard]] constexpr typename matrix_view<MDS>::const_row_type matrix_view<MDS>::
#else
template < class MDS, typename Dummy >
[[nodiscard]] constexpr typename matrix_view<MDS,Dummy>::const_row_type matrix_view<MDS,Dummy>::
#endif
row( index_type i ) const
{
  return const_row_type { experimental::submdspan( this->underlying_span(), i, experimental::full_extent ) };
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 2 ) && MDS::is_always_unique() )
[[nodiscard]] constexpr typename matrix_view<MDS>::const_submatrix_type matrix_view<MDS>::
#else
template < class MDS, typename Dummy >
[[nodiscard]] constexpr typename matrix_view<MDS,Dummy>::const_submatrix_type matrix_view<MDS,Dummy>::
#endif
submatrix( tuple_type start,
           tuple_type end ) const
{
  return const_submatrix_type { detail::submdspan( this->underlying_span(), start, end ) };
}

//- Mutable views

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 2 ) && MDS::is_always_unique() )
[[nodiscard]] constexpr typename matrix_view<MDS>::column_type matrix_view<MDS>::
#else
template < class MDS, typename Dummy >
template < typename >
[[nodiscard]] constexpr typename matrix_view<MDS,Dummy>::column_type matrix_view<MDS,Dummy>::
#endif
column( index_type j )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( !is_const_v<typename matrix_view<MDS>::element_type> )
#endif
{
  return column_type { experimental::submdspan( this->underlying_span(), experimental::full_extent, j ) };
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 2 ) && MDS::is_always_unique() )
[[nodiscard]] constexpr typename matrix_view<MDS>::row_type matrix_view<MDS>::
#else
template < class MDS, typename Dummy >
template < typename >
[[nodiscard]] constexpr typename matrix_view<MDS,Dummy>::row_type matrix_view<MDS,Dummy>::
#endif
row( index_type i )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( !is_const_v<typename matrix_view<MDS>::element_type> )
#endif
{
  return row_type { experimental::submdspan( this->underlying_span(), i, experimental::full_extent ) };
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 2 ) && MDS::is_always_unique() )
[[nodiscard]] constexpr typename matrix_view<MDS>::submatrix_type matrix_view<MDS>::
#else
template < class MDS, typename Dummy >
template < typename >
[[nodiscard]] constexpr typename matrix_view<MDS,Dummy>::submatrix_type matrix_view<MDS,Dummy>::
#endif
submatrix( tuple_type start,
           tuple_type end )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( !is_const_v<typename matrix_view<MDS>::element_type> )
#endif
{
  return submatrix_type { detail::submdspan( this->underlying_span(), start, end ) };
}

}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_FIXED_SIZE_MATRIX_HPP
