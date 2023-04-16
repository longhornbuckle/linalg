//==================================================================================================
//  File:       vector_view.hpp
//
//  Summary:    This header defines a vector_view, which is a non-owning view into a larger
//              vector.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_VECTOR_VIEW_HPP
#define LINEAR_ALGEBRA_VECTOR_VIEW_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{

/// @brief Vector view.
//         Implementation satisfies the following concepts:
//         concepts::vector
//         concepts::vector_view if MDS::element_type is non-const
//         concepts::vector_view if MDS::element_type is const
/// @tparam MDS mdspan
template < class MDS
#ifdef LINALG_ENABLE_CONCEPTS
  > requires ( detail::is_mdspan_v<MDS> &&
               ( MDS::extents_type::rank()== 1 ) &&
               MDS::is_always_unique() ) // Each element in the mdspan must have a unique mapping. (i.e. span_type and const_underlying_span_type should be the same.)
#else
  , typename >
#endif
class vector_view : public tensor_view<MDS>
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
    /// @brief Type contained by the vector
    using element_type               = typename base_type::element_type;
    /// @brief Type returned by const index access
    using value_type                 = typename base_type::value_type;
    /// @brief Type used for indexing
    using index_type                 = typename base_type::index_type;
    /// @brief Type used for size along any dimension
    using size_type                  = ::std::size_t;
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
    /// @brief mutable view of a subvector
    using subvector_type             = vector_view<decltype( ::std::experimental::submdspan( ::std::declval<underlying_span_type>(), ::std::declval< ::std::tuple<index_type,index_type> >() ) ) >;
    /// @brief const view of a subvector
    using const_subvector_type       = vector_view<decltype( ::std::experimental::submdspan( ::std::declval<const_underlying_span_type>(), ::std::declval< ::std::tuple<index_type,index_type> >() ) ) >;
    
    //- Destructor / Constructors / Assignments

    /// @brief Default destructor
    ~vector_view()                                     = default;
    /// @brief Default constructor
    vector_view()                                      = default;
    /// @brief Default move constructor
    /// @param vector_view to be moved
    vector_view( vector_view&& rhs )                   = default;
    /// @brief Default copy constructor
    /// @param vector_view to be copied
    vector_view( const vector_view& rhs )              = default;
    /// @brief Construct from view
    /// @param view to be constructed from
    explicit constexpr vector_view( const underlying_span_type& view ) noexcept;
    /// @brief Default move assignment
    /// @param  vector_view to be moved
    /// @return self
    vector_view& operator = ( vector_view&& rhs )      = default;
    /// @brief Default copy assignment
    /// @param  vector_view to be copied
    /// @return self
    vector_view& operator = ( const vector_view& rhs ) = default;
    /// @brief Assign from view
    /// @param  view to be assigned
    /// @return self
    constexpr vector_view& operator = ( const underlying_span_type& view ) noexcept;

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
    using base_type::subvector;  // Brings into scope const and mutable
    using base_type::submatrix;  // Brings into scope const and mutable

    /// @brief Returns the value at index
    /// @param index the location of the desired element
    /// @returns value at index
    [[nodiscard]] constexpr value_type at( index_type index ) const;
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
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename Elem = element_type, typename = ::std::enable_if_t< !::std::is_const_v<Elem> > >
    #endif
    [[nodiscard]] constexpr subvector_type subvector( index_type start,
                                                      index_type end )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( !::std::is_const_v<element_type> );
    #else
      ;
    #endif

    //- Data access

    using base_type::span;
    using base_type::underlying_span;
};

//------------------------------------------
// Implementation of vector_view<MDS>
//------------------------------------------

//- Destructor / Constructors / Assignments

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank()== 1 ) && MDS::is_always_unique() )
constexpr vector_view<MDS>::
#else
template < class MDS, typename Dummy >
constexpr vector_view<MDS,Dummy>::
#endif
vector_view( const underlying_span_type& view ) noexcept :
#ifdef LINALG_ENABLE_CONCEPTS
  vector_view<MDS>::base_type( view )
#else
  vector_view<MDS,Dummy>::base_type( view )
#endif
{
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank()== 1 ) && MDS::is_always_unique() )
constexpr vector_view<MDS>& vector_view<MDS>::
#else
template < class MDS, typename Dummy >
constexpr vector_view<MDS,Dummy>& vector_view<MDS,Dummy>::
#endif
operator = ( const underlying_span_type& view ) noexcept
{
  static_cast<void>( this->base_type::operator=( view ) );
  return *this;
}

//- Const views

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank()== 1 ) && MDS::is_always_unique() )
[[nodiscard]] constexpr typename vector_view<MDS>::const_subvector_type vector_view<MDS>::
#else
template < class MDS, typename Dummy >
[[nodiscard]] constexpr typename vector_view<MDS,Dummy>::const_subvector_type vector_view<MDS,Dummy>::
#endif
subvector( index_type start,
           index_type end ) const
{
  return const_subvector_type { ::std::experimental::submdspan( this->underlying_span(), ::std::tuple( start, end ) ) };
}

//- Mutable views

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank()== 1 ) && MDS::is_always_unique() )
[[nodiscard]] constexpr typename vector_view<MDS>::subvector_type vector_view<MDS>::
#else
template < class MDS, typename Dummy >
template < typename Elem, typename >
[[nodiscard]] constexpr typename vector_view<MDS,Dummy>::subvector_type vector_view<MDS,Dummy>::
#endif
subvector( index_type start,
           index_type end )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( !::std::is_const_v<typename vector_view<MDS>::element_type> )
#endif
{
  return subvector_type { ::std::experimental::submdspan( this->underlying_span(), ::std::tuple( start, end ) ) };
}

}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_VECTOR_VIEW_HPP
