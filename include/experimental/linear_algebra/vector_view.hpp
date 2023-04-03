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
template < class MDS > requires ( detail::is_mdspan_v<MDS> &&
                                  ( MDS::extents_type::rank()== 1 ) &&
                                  MDS::is_always_unique() ) // Each element in the mdspan must have a unique mapping. (i.e. span_type and const_underlying_span_type should be the same.)
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
    using size_type                  = size_t;
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
    /// @brief mutable view of a subvector
    using subvector_type             = vector_view<decltype( experimental::submdspan( declval<underlying_span_type>(), declval< tuple<index_type,index_type> >() ) ) >;
    /// @brief const view of a subvector
    using const_subvector_type       = vector_view<decltype( experimental::submdspan( declval<const_underlying_span_type>(), declval< tuple<index_type,index_type> >() ) ) >;
    
    //- Destructor / Constructors / Assignments

    /// @brief Default destructor
    ~vector_view()                                           = default;
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

    /// @brief Returns the current number of elements
    /// @return number of elements
    [[nodiscard]] constexpr size_type size() const noexcept;
    /// @brief Returns the current capacity
    /// @return capacity
    [[nodiscard]] constexpr size_type capacity() const noexcept;

    //- Const views

    /// @brief Returns the value at index
    /// @param index the location of the desired element
    /// @returns value at index
    [[nodiscard]] constexpr value_type operator[]( index_type index ) const noexcept;
    /// @brief Returns the value at index
    /// @param index the location of the desired element
    /// @returns value at index
    [[nodiscard]] constexpr value_type at( index_type index ) const;
    /// @brief Returns a const view of the specified subvector
    /// @param start (row,column) start of subvector
    /// @param end (row,column) end of subvector
    /// @returns const view of the specified subvector
    [[nodiscard]] constexpr const_subvector_type subvector( tuple_type start,
                                                            tuple_type end ) const;

    //- Mutable views

    /// @brief Returns the value at index
    /// @param index the location of the desired element
    /// @returns value at index
    [[nodiscard]] constexpr reference_type operator[]( index_type index ) noexcept;
    /// @brief Returns the value at index
    /// @param index the location of the desired element
    /// @returns value at index
    [[nodiscard]] constexpr reference_type at( index_type index );
    /// @brief Returns a mutable view of the specified subvector
    /// @param start (row,column) start of subvector
    /// @param end (row,column) end of subvector
    /// @returns mutable view of the specified subvector
    [[nodiscard]] constexpr subvector_type subvector( tuple_type start,
                                                      tuple_type end ) requires ( !is_const_v<element_type> );

    //- Data access

    using base_type::span;
    using base_type::underlying_span;
};

//------------------------------------------
// Implementation of vector_view<MDS>
//------------------------------------------

//- Destructor / Constructors / Assignments

template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 1 ) &&  MDS::is_always_unique() )
constexpr vector_view<MDS>::vector_view( const underlying_span_type& view ) noexcept :
  vector_view<MDS>::base_type( view )
{
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 1 ) &&  MDS::is_always_unique() )
constexpr vector_view<MDS>& vector_view<MDS>::operator = ( const underlying_span_type& view ) noexcept
{
  static_cast<void>( this->base_type::operator=( view ) );
  return *this;
}

//- Size / Capacity

template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 1 ) &&  MDS::is_always_unique() )
[[nodiscard]] constexpr typename vector_view<MDS>::size_type
vector_view<MDS>::size() const noexcept
{
  return detail::template extents_helper<size_type,extents_type::rank()>::size( this->base_type::size() );
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 1 ) &&  MDS::is_always_unique() )
[[nodiscard]] constexpr typename vector_view<MDS>::size_type
vector_view<MDS>::capacity() const noexcept
{
  return detail::template extents_helper<size_type,extents_type::rank()>::size( this->base_type::capacity() );
}

//- Const views

template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 1 ) &&  MDS::is_always_unique() )
[[nodiscard]] constexpr typename vector_view<MDS>::value_type
vector_view<MDS>::operator[]( index_type index ) const noexcept
{
  return this->underlying_span()[index];
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 1 ) &&  MDS::is_always_unique() )
[[nodiscard]] constexpr typename vector_view<MDS>::value_type
vector_view<MDS>::at( index_type index ) const
{
  return this->underlying_span()[index];
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 1 ) &&  MDS::is_always_unique() )
[[nodiscard]] constexpr typename vector_view<MDS>::const_subvector_type
vector_view<MDS>::subvector( tuple_type start,
                                    tuple_type end ) const
{
  return const_subtensor_type { experimental::submdspan( this->underlying_span(), tuple( start, end ) ) };
}

//- Mutable views

template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 1 ) &&  MDS::is_always_unique() )
[[nodiscard]] constexpr typename vector_view<MDS>::reference_type
vector_view<MDS>::operator[]( index_type index ) noexcept
{
  return this->underlying_span()[index];
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 1 ) &&  MDS::is_always_unique() )
[[nodiscard]] constexpr typename vector_view<MDS>::reference_type
vector_view<MDS>::at( index_type index )
{
  return this->underlying_span()[index];
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && ( MDS::extents_type::rank() == 1 ) &&  MDS::is_always_unique() )
[[nodiscard]] constexpr typename vector_view<MDS>::subvector_type
vector_view<MDS>::subvector( tuple_type start,
                             tuple_type end ) requires ( !is_const_v<typename vector_view<MDS>::element_type> )
{
  return subtensor_type { experimental::submdspan( this->underlying_span(), tuple( start, end ) ) };
}

}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_VECTOR_VIEW_HPP
