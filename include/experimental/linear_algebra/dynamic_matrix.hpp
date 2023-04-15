//==================================================================================================
//  File:       dynamic_matrix.hpp
//
//  Summary:    This header defines a dynamic_matrix. In this context, dynamic means the row and
//              column extents of such objects are not known at compile-time.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_DYNAMIC_MATRIX_HPP
#define LINEAR_ALGEBRA_DYNAMIC_MATRIX_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{

/// @brief Dynamic-size, dynamic-capacity matrix.
//         Implementation satisfies the following concepts:
//         concepts::dynamic_matrix
//         concepts::writable_matrix
/// @tparam T      element_type
/// @tparam Alloc  allocator_type
/// @tparam L      layout defines the ordering of elements in memory
/// @tparam Access accessor policy defines how elements are accessed
template < class T,
           class Alloc,
           class L,
           class Access >
class dr_matrix : public dr_tensor<T,2,Alloc,L,Access>
{
  private:
    // Base tensor type
    using base_type = dr_tensor<T,2,Alloc,L,Access>;
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
    /// @brief mutable view of a column vector
    using column_type                = vector_view<decltype( ::std::experimental::submdspan( ::std::declval<underlying_span_type>(), ::std::declval<::std::experimental::full_extent_t>(), ::std::declval<index_type>() ) )>;
    /// @brief const view of a column vector
    using const_column_type          = vector_view<decltype( ::std::experimental::submdspan( ::std::declval<const_underlying_span_type>(), ::std::declval<::std::experimental::full_extent_t>(), ::std::declval<index_type>() ) )>;
    /// @brief mutable view of a row vector
    using row_type                   = vector_view<decltype( ::std::experimental::submdspan( ::std::declval<underlying_span_type>(), ::std::declval<index_type>(), ::std::declval<::std::experimental::full_extent_t>() ) )>;
    /// @brief const view of a row vector
    using const_row_type             = vector_view<decltype( ::std::experimental::submdspan( ::std::declval<const_underlying_span_type>(), ::std::declval<index_type>(), ::std::declval<::std::experimental::full_extent_t>() ) )>;
    /// @brief mutable view of a submatrix
    using submatrix_type             = matrix_view<decltype( detail::submdspan( ::std::declval<underlying_span_type>(), ::std::declval<tuple_type>(), ::std::declval<tuple_type>() ) )>;
    /// @brief const view of a submatrix
    using const_submatrix_type       = matrix_view<decltype( detail::submdspan( ::std::declval<const_underlying_span_type>(), ::std::declval<tuple_type>(), ::std::declval<tuple_type>() ) )>;
    /// @brief matrix tanspose
    using transpose_type             = dr_matrix;
    
    //- Rebind

    /// @brief Rebind defines a type for a rebinding a dynamic matrix to the new type parameters
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
      using type = dr_matrix< ValueType,
                              typename ::std::allocator_traits<allocator_type>::template rebind_alloc<rebind_element_type>,
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
    ~dr_matrix()                            = default;
    /// @brief Default constructor
    constexpr dr_matrix()                   = default;
    /// @brief Default move constructor
    /// @param dr_matrix to be moved
    constexpr dr_matrix( dr_matrix&& )      = default;
    /// @brief Default copy constructor
    /// @param dr_matrix to be copied
    constexpr dr_matrix( const dr_matrix& ) = default;
    /// @brief Template copy constructor
    /// @tparam matrix to be copied
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::tensor_may_be_constructible< dr_matrix > M2 >
    #else
    template < class M2, typename = ::std::enable_if_t< concepts::tensor_may_be_constructible< M2, dr_matrix > > >
    #endif
    explicit constexpr dr_matrix( const M2& rhs ) noexcept( noexcept( base_type(rhs) ) );
    /// @brief Construct from a view
    /// @tparam A two dimensional view
    /// @param view view of matrix elements
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< dr_matrix > MDS >
    #else
    template < class MDS, typename = ::std::enable_if_t< concepts::view_may_be_constructible_to_tensor<MDS,dr_matrix> && ::std::is_default_constructible_v<allocator_type> >, typename = ::std::enable_if_t<true> >
    #endif
    explicit constexpr dr_matrix( const MDS& view ) noexcept( noexcept( base_type(view) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ::std::is_default_constructible_v<allocator_type>;
    #else
      ;
    #endif
    /// @brief Attempt to allocate sufficient resources for a size s matrix and construct
    /// @param s defines the rows and columns of the matrix
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename = enable_if_t< ::std::is_default_constructible_v<allocator_type> > >
    #endif
    explicit constexpr dr_matrix( extents_type s ) noexcept( noexcept( base_type(s) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ::std::is_default_constructible_v<allocator_type>;
    #else
      ;
    #endif
    /// @brief Attempt to allocate sufficient resources for a size s matrix with the input capacity and construct
    /// @param s defines the rows and columns of the matrix
    /// @param cap defines the capacity along each of the dimensions of the matrix
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename = enable_if_t< ::std::is_default_constructible_v<allocator_type> > >
    #endif
    constexpr dr_matrix( extents_type s, extents_type cap ) noexcept( noexcept( base_type(s,cap) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ::std::is_default_constructible_v<allocator_type>;
    #else
      ;
    #endif
    /// @brief Construct by applying lambda to every element in the matrix
    /// @tparam Lambda lambda expression with an operator()( index1, index2 ) defined
    /// @param s defines the rows and columns of the matrix
    /// @param lambda lambda expression to be performed on each element
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = enable_if_t< ::std::is_default_constructible_v<allocator_type> &&
                                       ::std::is_convertible_v< decltype( ::std::declval<Lambda&&>()( ::std::declval<index_type>(), ::std::declval<index_type>() ) ), element_type > > >
    #endif
    constexpr dr_matrix( extents_type s, Lambda&& lambda ) noexcept( noexcept( base_type(s,lambda) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ::std::is_default_constructible_v<allocator_type> &&
               requires { { ::std::declval<Lambda&&>()( ::std::declval<index_type>(), ::std::declval<index_type>() ) } -> ::std::convertible_to<element_type>; };
    #else
      ;
    #endif
    /// @brief Construct by applying lambda to every element in the matrix
    /// @tparam Lambda lambda expression with an operator()( index1, index2 ) defined
    /// @param s defines the rows and columns of the matrix
    /// @param cap defines the capacity along each of the dimensions of the matrix
    /// @param lambda lambda expression to be performed on each element
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = enable_if_t< ::std::is_default_constructible_v<allocator_type> &&
                                       ::std::is_convertible_v< decltype( ::std::declval<Lambda&&>()( ::std::declval<index_type>(), ::std::declval<index_type>() ) ), element_type > > >
    #endif
    constexpr dr_matrix( extents_type s, extents_type cap, Lambda&& lambda ) noexcept( noexcept( base_type(s,cap,lambda) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ::std::is_default_constructible_v<allocator_type> &&
               requires { { ::std::declval<Lambda&&>()( ::std::declval<index_type>(), ::std::declval<index_type>() ) } -> ::std::convertible_to<element_type>; };
    #else
      ;
    #endif
    /// @brief Construct empty dimensionless matrix with an allocator
    /// @param alloc allocator to construct with
    explicit constexpr dr_matrix( const allocator_type& alloc ) noexcept( noexcept( base_type(alloc) ) );
    /// @brief Construct from a view
    /// @tparam An N dimensional view
    /// @param view view of matrix elements
    /// @param alloc allocator to construct with
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< dr_matrix > MDS >
    #else
    template < class MDS, typename = enable_if_t< concepts::view_may_be_constructible_to_tensor<MDS,dr_matrix> > >
    #endif
    explicit constexpr dr_matrix( const MDS& view, const allocator_type& alloc ) noexcept( noexcept( base_type(view,alloc) ) );
    /// @brief Attempt to allocate sufficient resources for a size matrix and construct
    /// @param s defines the rows and columns of the matrix
    /// @param alloc allocator used to construct with
    constexpr dr_matrix( extents_type s, const allocator_type& alloc ) noexcept( noexcept( base_type(s,alloc) ) );
    /// @brief Attempt to allocate sufficient resources for a size s matrix with the input capacity and construct
    /// @param s defines the rows and columns of the matrix
    /// @param cap defines the capacity along each of the dimensions of the matrix
    /// @param alloc allocator used to construct with
    constexpr dr_matrix( extents_type s, extents_type cap, const allocator_type& alloc ) noexcept( noexcept( base_type(s,cap,alloc) ) );
    /// @brief Construct by applying lambda to every element in the matrix
    /// @tparam Lambda lambda expression with an operator()( index1, index2 ) defined
    /// @param s defines the rows and columns of the matrix
    /// @param lambda lambda expression to be performed on each element
    /// @param alloc allocator used to construct with
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = ::std::enable_if_t< is_convertible_v< decltype( ::std::declval<Lambda&&>()( ::std::declval<index_type>(), ::std::declval<index_type>() ) ), element_type > > >
    #endif
    constexpr dr_matrix( extents_type s, Lambda&& lambda, const allocator_type& alloc ) noexcept( noexcept( base_type(s,lambda,alloc) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires requires { { ::std::declval<Lambda&&>()( ::std::declval<index_type>(), ::std::declval<index_type>() ) } -> ::std::convertible_to<element_type>; };
    #else
      ;
    #endif
    /// @brief Construct by applying lambda to every element in the matrix
    /// @tparam Lambda lambda expression with an operator()( index1, index2 ) defined
    /// @param s defines the rows and columns of the matrix
    /// @param cap defines the capacity along each of the dimensions of the matrix
    /// @param lambda lambda expression to be performed on each element
    /// @param alloc allocator used to construct with
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = ::std::enable_if_t< is_convertible_v< decltype( ::std::declval<Lambda&&>()( ::std::declval<index_type>(), ::std::declval<index_type>() ) ), element_type > > >
    #endif
    constexpr dr_matrix( extents_type s, extents_type cap, Lambda&& lambda, const allocator_type& alloc ) noexcept( noexcept( base_type(s,cap,lambda,alloc) ) )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires requires { { ::std::declval<Lambda&&>()( ::std::declval<index_type>(), ::std::declval<index_type>() ) } -> ::std::convertible_to<element_type>; };
    #else
      ;
    #endif
    /// @brief Default move assignment
    /// @param  dr_matrix to be moved
    /// @return self
    constexpr dr_matrix& operator = ( dr_matrix&& )      = default;
    /// @brief Default copy assignment
    /// @param  dr_matrix to be copied
    /// @return self
    constexpr dr_matrix& operator = ( const dr_matrix& ) = default;
    /// @brief Template copy assignment
    /// @tparam type of matrix to be copied
    /// @param  matrix to be copied
    /// @returns self
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::tensor_may_be_constructible< dr_matrix > M2 >
    #else
    template < class M2, typename = ::std::enable_if_t< concepts::tensor_may_be_constructible< M2, dr_matrix > > >
    #endif
    constexpr dr_matrix& operator = ( const M2& rhs ) noexcept( noexcept( ::std::declval<base_type>() = rhs ) );
    /// @brief Construct from a two dimensional view
    /// @tparam type of view to be copied
    /// @param  view to be copied
    /// @returns self
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< dr_matrix > MDS >
    #else
    template < class MDS, typename = ::std::enable_if_t< concepts::view_may_be_constructible_to_tensor<MDS,dr_matrix> && ::std::is_default_constructible_v<allocator_type> >, typename = ::std::enable_if_t<true> >
    #endif
    constexpr dr_matrix& operator = ( const MDS& view ) noexcept( noexcept( ::std::declval<base_type>() = view ) );

    //- Size / Capacity

    using base_type::size;
    using base_type::capacity;
    using base_type::resize;
    using base_type::reserve;

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
// Implementation of dr_matrix<T,Alloc,L,Access>
//----------------------------------------------

//- Destructor / Constructors / Assignments

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_may_be_constructible< dr_matrix<T,Alloc,L,Access> > M2 >
#else
template < class M2, typename >
#endif
constexpr dr_matrix<T,Alloc,L,Access>::dr_matrix( const M2& rhs )
  noexcept( noexcept( dr_matrix<T,Alloc,L,Access>::base_type( rhs ) ) ) :
  dr_matrix<T,Alloc,L,Access>::base_type( rhs )
{
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::view_may_be_constructible_to_tensor< dr_matrix<T,Alloc,L,Access> > MDS >
#else
template < class MDS, typename, typename >
#endif
constexpr dr_matrix<T,Alloc,L,Access>::dr_matrix( const MDS& view )
  noexcept( noexcept( dr_matrix<T,Alloc,L,Access>::base_type(view) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_default_constructible_v<typename dr_matrix<T,Alloc,L,Access>::allocator_type> :
#else
  :
#endif
  dr_matrix<T,Alloc,L,Access>::base_type(view)
{
}
  
template < class T, class Alloc, class L, class Access >
#ifndef LINALG_ENABLE_CONCEPTS
template < typename >
#endif
constexpr dr_matrix<T,Alloc,L,Access>::dr_matrix( extents_type s )
  noexcept( noexcept( dr_matrix<T,Alloc,L,Access>::base_type(s) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_default_constructible_v<typename dr_matrix<T,Alloc,L,Access>::allocator_type> :
#else
  :
#endif
  dr_matrix<T,Alloc,L,Access>::base_type(s)
{
}

template < class T, class Alloc, class L, class Access >
#ifndef LINALG_ENABLE_CONCEPTS
template < typename >
#endif
constexpr dr_matrix<T,Alloc,L,Access>::dr_matrix( extents_type s, extents_type cap )
  noexcept( noexcept( dr_matrix<T,Alloc,L,Access>::base_type(s,cap) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_default_constructible_v<typename dr_matrix<T,Alloc,L,Access>::allocator_type> :
#else
  :
#endif
  dr_matrix<T,Alloc,L,Access>::base_type(s,cap)
{
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < class Lambda >
#else
template < class Lambda, typename >
#endif
constexpr dr_matrix<T,Alloc,L,Access>::dr_matrix( extents_type s, Lambda&& lambda )
  noexcept( noexcept( dr_matrix<T,Alloc,L,Access>::base_type(s,lambda) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_default_constructible_v<typename dr_matrix<T,Alloc,L,Access>::allocator_type> &&
           requires { { ::std::declval<Lambda&&>()( ::std::declval<typename dr_matrix<T,Alloc,L,Access>::index_type>(),
                                                    ::std::declval<typename dr_matrix<T,Alloc,L,Access>::index_type>() ) }
                      -> ::std::convertible_to<typename dr_matrix<T,Alloc,L,Access>::element_type>; } :
#else
  :
#endif
  dr_matrix<T,Alloc,L,Access>::base_type(s,lambda)
{
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < class Lambda >
#else
template < class Lambda, typename >
#endif
constexpr dr_matrix<T,Alloc,L,Access>::dr_matrix( extents_type s, extents_type cap, Lambda&& lambda )
  noexcept( noexcept( dr_matrix<T,Alloc,L,Access>::base_type(s,cap,lambda) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ::std::is_default_constructible_v<typename dr_matrix<T,Alloc,L,Access>::allocator_type> &&
           requires { { ::std::declval<Lambda&&>()( ::std::declval<typename dr_matrix<T,Alloc,L,Access>::index_type>(),
                                                    ::std::declval<typename dr_matrix<T,Alloc,L,Access>::index_type>() ) }
                      -> ::std::convertible_to<typename dr_matrix<T,Alloc,L,Access>::element_type>; } :
#else
  :
#endif
  dr_matrix<T,Alloc,L,Access>::base_type(s,cap,lambda)
{
}

template < class T, class Alloc, class L, class Access >
constexpr dr_matrix<T,Alloc,L,Access>::dr_matrix( const allocator_type& alloc )
  noexcept( noexcept( dr_matrix<T,Alloc,L,Access>::base_type(alloc) ) ) :
  dr_matrix<T,Alloc,L,Access>::base_type(alloc)
{
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::view_may_be_constructible_to_tensor< dr_matrix<T,Alloc,L,Access> > MDS >
#else
template < class MDS, typename >
#endif
constexpr dr_matrix<T,Alloc,L,Access>::dr_matrix( const MDS& view, const allocator_type& alloc )
  noexcept( noexcept( dr_matrix<T,Alloc,L,Access>::base_type(view,alloc) ) ) :
  dr_matrix<T,Alloc,L,Access>::base_type(view,alloc)
{
}

template < class T, class Alloc, class L, class Access >
constexpr dr_matrix<T,Alloc,L,Access>::dr_matrix( extents_type s, const allocator_type& alloc )
  noexcept( noexcept( dr_matrix<T,Alloc,L,Access>::base_type(s,alloc) ) ) :
  dr_matrix<T,Alloc,L,Access>::base_type(s,alloc)
{
}

template < class T, class Alloc, class L, class Access >
constexpr dr_matrix<T,Alloc,L,Access>::dr_matrix( extents_type s, extents_type cap, const allocator_type& alloc )
  noexcept( noexcept( dr_matrix<T,Alloc,L,Access>::base_type(s,cap,alloc) ) ) :
  dr_matrix<T,Alloc,L,Access>::base_type(s,cap,alloc)
{
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < class Lambda >
#else
template < class Lambda, typename >
#endif
constexpr dr_matrix<T,Alloc,L,Access>::dr_matrix( extents_type s, Lambda&& lambda, const allocator_type& alloc )
  noexcept( noexcept( dr_matrix<T,Alloc,L,Access>::base_type(s,lambda,alloc) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires { { ::std::declval<Lambda&&>()( ::std::declval<typename dr_matrix<T,Alloc,L,Access>::index_type>(),
                                                    ::std::declval<typename dr_matrix<T,Alloc,L,Access>::index_type>() ) }
                    -> ::std::convertible_to<typename dr_matrix<T,Alloc,L,Access>::element_type>; } :
#else
  :
#endif
  dr_matrix<T,Alloc,L,Access>::base_type(s,lambda,alloc)
{
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < class Lambda >
#else
template < class Lambda, typename >
#endif
constexpr dr_matrix<T,Alloc,L,Access>::dr_matrix( extents_type s, extents_type cap, Lambda&& lambda, const allocator_type& alloc )
  noexcept( noexcept( dr_matrix<T,Alloc,L,Access>::base_type(s,cap,lambda,alloc) ) )
#ifdef LINALG_ENABLE_CONCEPTS
  requires requires { { ::std::declval<Lambda&&>()( ::std::declval<typename dr_matrix<T,Alloc,L,Access>::index_type>(),
                                                    ::std::declval<typename dr_matrix<T,Alloc,L,Access>::index_type>() ) }
                      -> ::std::convertible_to<typename dr_matrix<T,Alloc,L,Access>::element_type>; } :
#else
  :
#endif
  dr_matrix<T,Alloc,L,Access>::base_type(s,cap,lambda,alloc)
{
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_may_be_constructible< dr_matrix<T,Alloc,L,Access> > M2 >
#else
template < class M2, typename >
#endif
constexpr dr_matrix<T,Alloc,L,Access>& dr_matrix<T,Alloc,L,Access>::operator = ( const M2& rhs )
  noexcept( noexcept( ::std::declval<typename dr_matrix<T,Alloc,L,Access>::base_type>() = rhs ) )
{
  static_cast<void>( this->base_type::operator=(rhs) );
  return *this;
}

template < class T, class Alloc, class L, class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::view_may_be_constructible_to_tensor< dr_matrix<T,Alloc,L,Access> > MDS >
#else
template < class MDS, typename, typename >
#endif
constexpr dr_matrix<T,Alloc,L,Access>& dr_matrix<T,Alloc,L,Access>::operator = ( const MDS& view )
  noexcept( noexcept( ::std::declval<typename dr_matrix<T,Alloc,L,Access>::base_type>() = view ) )
{
  static_cast<void>( this->base_type::operator=(view) );
  return *this;
}

//- Size / Capacity

template < class T, class Alloc, class L, class Access >
[[nodiscard]] constexpr typename dr_matrix<T,Alloc,L,Access>::size_type
dr_matrix<T,Alloc,L,Access>::columns() const noexcept
{
  return this->size().extent(1);
}

template < class T, class Alloc, class L, class Access >
[[nodiscard]] constexpr typename dr_matrix<T,Alloc,L,Access>::size_type
dr_matrix<T,Alloc,L,Access>::rows() const noexcept
{
  return this->size().extent(0);
}

template < class T, class Alloc, class L, class Access >
[[nodiscard]] constexpr typename dr_matrix<T,Alloc,L,Access>::size_type
dr_matrix<T,Alloc,L,Access>::column_capacity() const noexcept
{
  return this->capacity().extent(1);
}

template < class T, class Alloc, class L, class Access >
[[nodiscard]] constexpr typename dr_matrix<T,Alloc,L,Access>::size_type
dr_matrix<T,Alloc,L,Access>::row_capacity() const noexcept
{
  return this->capacity().extent(0);
}

//- Const views

template < class T, class Alloc, class L, class Access >
[[nodiscard]] constexpr typename dr_matrix<T,Alloc,L,Access>::const_column_type
dr_matrix<T,Alloc,L,Access>::column( index_type j ) const
{
  return const_column_type { ::std::experimental::submdspan( this->underlying_span(), ::std::experimental::full_extent, j ) };
}

template < class T, class Alloc, class L, class Access >
[[nodiscard]] constexpr typename dr_matrix<T,Alloc,L,Access>::const_row_type
dr_matrix<T,Alloc,L,Access>::row( index_type i ) const
{
  return const_row_type { ::std::experimental::submdspan( this->underlying_span(), i, ::std::experimental::full_extent ) };
}

template < class T, class Alloc, class L, class Access >
[[nodiscard]] constexpr typename dr_matrix<T,Alloc,L,Access>::const_submatrix_type
dr_matrix<T,Alloc,L,Access>::submatrix( tuple_type start,
                                        tuple_type end ) const
{
  return const_submatrix_type { detail::submdspan( this->underlying_span(), start, end ) };
}

//- Mutable views

template < class T, class Alloc, class L, class Access >
[[nodiscard]] constexpr typename dr_matrix<T,Alloc,L,Access>::column_type
dr_matrix<T,Alloc,L,Access>::column( index_type j )
{
  return column_type { ::std::experimental::submdspan( this->underlying_span(), ::std::experimental::full_extent, j ) };
}

template < class T, class Alloc, class L, class Access >
[[nodiscard]] constexpr typename dr_matrix<T,Alloc,L,Access>::row_type
dr_matrix<T,Alloc,L,Access>::row( index_type i )
{
  return row_type { ::std::experimental::submdspan( this->underlying_span(), i, ::std::experimental::full_extent ) };
}

template < class T, class Alloc, class L, class Access >
[[nodiscard]] constexpr typename dr_matrix<T,Alloc,L,Access>::submatrix_type
dr_matrix<T,Alloc,L,Access>::submatrix( tuple_type start,
                                        tuple_type end )
{
  return submatrix_type { detail::submdspan( this->underlying_span(), start, end ) };
}

}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_FIXED_SIZE_MATRIX_HPP
