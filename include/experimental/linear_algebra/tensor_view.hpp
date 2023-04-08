//==================================================================================================
//  File:       tensor_view.hpp
//
//  Summary:    This header defines a tensor_view, which is a non-owning view into a larger
//              tensor.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_VIEW_HPP
#define LINEAR_ALGEBRA_TENSOR_VIEW_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{

/// @brief Tensor view.
//         Implementation satisfies the following concepts:
//         concepts::tensor
//         concepts::writable_tensor if MDS::element_type is non-const
//         concepts::readable_tensor if MDS::element_type is const
/// @tparam MDS mdspan
template < class MDS > requires ( detail::is_mdspan_v<MDS> &&
                                  MDS::is_always_unique() ) // Each element in the mdspan must have a unique mapping. (i.e. span_type and const_underlying_span_type should be the same.)
class tensor_view
{
  public:
    //- Types

    /// @brief Type used to define memory layout
    using layout_type                = typename MDS::layout_type;
    /// @brief Type used to define access into memory
    using accessor_type              = typename MDS::accessor_type;
    /// @brief Type contained by the tensor view
    using element_type               = typename MDS::element_type;
    /// @brief Type returned by const index access
    using value_type                 = typename MDS::value_type;
    /// @brief Type used to express size of tensor
    using extents_type               = typename MDS::extents_type;
    /// @brief Type used for indexing
    using index_type                 = typename extents_type::size_type;
    /// @brief Type used for size along any dimension
    using size_type                  = size_t;
    /// @brief Type used to represent a node in the tensor
    using tuple_type                 = typename detail::template extents_helper<size_type,extents_type::rank()>::tuple_type;
    /// @brief Type used to const view memory
    using const_underlying_span_type = experimental::mdspan<const remove_const_t<element_type>,extents_type,layout_type,accessor_type>;
    /// @brief Type used to view memory
    using underlying_span_type       = experimental::mdspan<element_type,extents_type,layout_type,accessor_type>;
    /// @brief Type used to portray tensor as an N dimensional view
    using span_type                  = const_underlying_span_type;
    /// @brief Type returned by mutable index access
    using reference_type             = typename MDS::reference;
    /// @brief mutable view of a subtensor
    using subtensor_type             = tensor_view<decltype( detail::submdspan( declval<underlying_span_type>(), declval<tuple_type>(), declval<tuple_type>() ) ) >;
    /// @brief const view of a subtensor
    using const_subtensor_type       = tensor_view<decltype( detail::submdspan( declval<const_underlying_span_type>(), declval<tuple_type>(), declval<tuple_type>() ) ) >;
    
    //- Destructor / Constructors / Assignments

    /// @brief Default destructor
    ~tensor_view()                                     = default;
    /// @brief Default constructor
    tensor_view()                                      = default;
    /// @brief Default move constructor
    /// @param tensor_view to be moved
    tensor_view( tensor_view&& rhs )                   = default;
    /// @brief Default copy constructor
    /// @param tensor_view to be copied
    tensor_view( const tensor_view& rhs )              = default;
    /// @brief Construct from view
    /// @param view to be constructed from
    explicit constexpr tensor_view( const underlying_span_type& view ) noexcept;
    /// @brief Default move assignment
    /// @param  tensor_view to be moved
    /// @return self
    tensor_view& operator = ( tensor_view&& rhs )      = default;
    /// @brief Default copy assignment
    /// @param  tensor_view to be copied
    /// @return self
    tensor_view& operator = ( const tensor_view& rhs ) = default;
    /// @brief Assign from view
    /// @param  view to be assigned
    /// @return self
    constexpr tensor_view& operator = ( const underlying_span_type& rhs ) noexcept;

    //- Size / Capacity

    /// @brief Returns the current number of (rows,columns,depth,etc.)
    /// @return number of (rows,columns,depth,etc.)
    [[nodiscard]] constexpr extents_type size() const noexcept;
    /// @brief Returns the current capacity of (rows,columns,depth,etc.)
    /// @return capacity of (rows,columns,depth,etc.)
    [[nodiscard]] constexpr extents_type capacity() const noexcept;

    //- Const views

    /// @brief Returns the value at (indices...)
    /// @param indices set indices representing a node in the tensor
    /// @returns value at row i, column j, depth k, etc.
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... IndexType >
    [[nodiscard]] constexpr value_type operator[]( IndexType ... indices ) const noexcept
      requires ( sizeof...(IndexType) == extents_type::rank() ) && ( is_convertible_v<IndexType,index_type> && ... );
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... IndexType >
    [[nodiscard]] constexpr value_type operator()( IndexType ... indices ) const noexcept
      requires ( sizeof...(IndexType) == extents_type::rank() ) && ( is_convertible_v<IndexType,index_type> && ... );
    #endif
    /// @brief Returns the value at (indices...)
    /// @param indices set indices representing a node in the tensor
    /// @returns value at row i, column j, depth k, etc.
    template < class ... IndexType >
    [[nodiscard]] constexpr value_type at( IndexType ... indices ) const
      requires ( sizeof...(IndexType) == extents_type::rank() ) && ( is_convertible_v<IndexType,index_type> && ... );
    /// @brief Returns a const view of the specified subtensor
    /// @param start tuple start of subtensor
    /// @param end tuple end of subtensor
    /// @returns const view of the specified subtensor
    [[nodiscard]] constexpr const_subtensor_type subtensor( tuple_type start,
                                                            tuple_type end ) const;

    //- Mutable views

    /// @brief Returns a mutable value at (indices...)
    /// @param indices set indices representing a node in the tensor
    /// @returns mutable value at row i, column j, depth k, etc.
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... IndexType >
    [[nodiscard]] constexpr reference_type operator[]( IndexType ... indices ) noexcept
      requires ( sizeof...(IndexType) == extents_type::rank() ) && ( is_convertible_v<IndexType,index_type> && ... ) &&
               ( !is_const_v<element_type> );
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... IndexType >
    [[nodiscard]] constexpr reference_type operator()( IndexType ... indices ) noexcept
      requires ( sizeof...(IndexType) == extents_type::rank() ) && ( is_convertible_v<IndexType,index_type> && ... ) &&
               ( !is_const_v<element_type> );
    #endif
    /// @brief Returns a mutable value at (indices...)
    /// @param indices set indices representing a node in the tensor
    /// @returns mutable value at row i, column j, depth k, etc.
    template < class ... IndexType >
    [[nodiscard]] constexpr reference_type at( IndexType ... indices )
      requires ( sizeof...(IndexType) == extents_type::rank() ) && ( is_convertible_v<IndexType,index_type> && ... ) &&
               ( !is_const_v<element_type> );
    /// @brief Returns a mutable view of the specified subtensor
    /// @param start tuple start of subtensor
    /// @param end tuple end of subtensor
    /// @returns mutable view of the specified subtensor
    [[nodiscard]] constexpr subtensor_type subtensor( tuple_type start,
                                                      tuple_type end ) requires ( !is_const_v<element_type> );

    //- Data access
    
    /// @brief returns a const view
    /// @returns const view
    [[nodiscard]] constexpr span_type                  span() const noexcept;
    /// @brief the view
    /// @returns returns the view
    [[nodiscard]] constexpr underlying_span_type       underlying_span() noexcept requires ( !is_const_v<element_type> );
    /// @brief returns a const view
    /// @returns const view
    [[nodiscard]] constexpr const_underlying_span_type underlying_span() const noexcept;

  private:
    
    //- Data

    /// @brief non-owning view of the elements in the tensor
    underlying_span_type view_;
};

//------------------------------------------
// Implementation of tensor_view<MDS>
//------------------------------------------

//- Destructor / Constructors / Assignments

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
constexpr tensor_view<MDS>::tensor_view( const underlying_span_type& view ) noexcept :
  view_(view)
{
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
constexpr tensor_view<MDS>& tensor_view<MDS>::operator = ( const underlying_span_type& rhs ) noexcept
{
  static_cast<void>( this->view_ = rhs );
  return *this;
}

//- Size / Capacity

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
[[nodiscard]] constexpr typename tensor_view<MDS>::extents_type
tensor_view<MDS>::size() const noexcept
{
  return this->view_.extents();
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
[[nodiscard]] constexpr typename tensor_view<MDS>::extents_type
tensor_view<MDS>::capacity() const noexcept
{
  return this->view_.extents();
}

//- Const views

#if LINALG_USE_BRACKET_OPERATOR
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... IndexType >
[[nodiscard]] constexpr typename tensor_view<MDS>::value_type
tensor_view<MDS>::operator[]( IndexType ... indices ) const noexcept
  requires ( sizeof...(IndexType) == tensor_view<MDS>::extents_type::rank() ) && ( is_convertible_v<IndexType,typename tensor_view<MDS>::index_type> && ... )
{
  return this->underlying_span()[ indices ... ];
}
#endif

#if LINALG_USE_PAREN_OPERATOR
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... IndexType >
[[nodiscard]] constexpr typename tensor_view<MDS>::value_type
tensor_view<MDS>::operator()( IndexType ... indices ) const noexcept
  requires ( sizeof...(IndexType) == tensor_view<MDS>::extents_type::rank() ) && ( is_convertible_v<IndexType,typename tensor_view<MDS>::index_type> && ... )
{
  return this->underlying_span()( indices ... );
}
#endif

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... IndexType >
[[nodiscard]] constexpr tensor_view<MDS>::value_type
tensor_view<MDS>::at( IndexType ... indices ) const
  requires ( sizeof...(IndexType) == tensor_view<MDS>::extents_type::rank() ) && ( is_convertible_v<IndexType,typename tensor_view<MDS>::index_type> && ... )
{
  return detail::access( this->underlying_span(), indices ... );
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
[[nodiscard]] constexpr typename tensor_view<MDS>::const_subtensor_type
tensor_view<MDS>::subtensor( tuple_type start,
                                    tuple_type end ) const
{
  return const_subtensor_type( detail::submdspan( this->underlying_span(), start, end ) );
}

//- Mutable views

#if LINALG_USE_BRACKET_OPERATOR
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... IndexType >
[[nodiscard]] constexpr typename tensor_view<MDS>::reference_type
tensor_view<MDS>::operator[]( IndexType ... indices ) noexcept
  requires ( sizeof...(IndexType) == tensor_view<MDS>::extents_type::rank() ) && ( is_convertible_v<IndexType,typename tensor_view<MDS>::index_type> && ... ) &&
           ( !is_const_v<typename tensor_view<MDS>::element_type> )
{
  return forward<typename tensor_view<MDS>::reference_type>( this->underlying_span()[ indices ... ] );
}
#endif

#if LINALG_USE_PAREN_OPERATOR
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... IndexType >
[[nodiscard]] constexpr typename tensor_view<MDS>::reference_type
tensor_view<MDS>::operator()( IndexType ... indices ) noexcept
  requires ( sizeof...(IndexType) == tensor_view<MDS>::extents_type::rank() ) && ( is_convertible_v<IndexType,typename tensor_view<MDS>::index_type> && ... ) &&
           ( !is_const_v<typename tensor_view<MDS>::element_type> )
{
  return forward<typename tensor_view<MDS>::reference_type>( this->underlying_span()( indices ... ) );
}
#endif

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... IndexType >
[[nodiscard]] constexpr typename tensor_view<MDS>::reference_type
tensor_view<MDS>::at( IndexType ... indices )
  requires ( sizeof...(IndexType) == tensor_view<MDS>::extents_type::rank() ) && ( is_convertible_v<IndexType,typename tensor_view<MDS>::index_type> && ... ) &&
           ( !is_const_v<typename tensor_view<MDS>::element_type> )
{
  return forward<typename tensor_view<MDS>::reference_type>( detail::access( this->underlying_span(), indices ... ) );
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
[[nodiscard]] constexpr typename tensor_view<MDS>::subtensor_type
tensor_view<MDS>::subtensor( tuple_type start,
                                    tuple_type end ) requires ( !is_const_v<typename tensor_view<MDS>::element_type> )
{
  return subtensor_type( detail::submdspan( this->underlying_span(), start, end ) );
}

//- Data access

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
[[nodiscard]] constexpr typename tensor_view<MDS>::span_type
tensor_view<MDS>::span() const noexcept
{
  return this->const_underlying_span();
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
[[nodiscard]] constexpr typename tensor_view<MDS>::underlying_span_type
tensor_view<MDS>::underlying_span() noexcept requires ( !is_const_v<typename tensor_view<MDS>::element_type> )
{
  return this->view_;
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
[[nodiscard]] constexpr typename tensor_view<MDS>::const_underlying_span_type
tensor_view<MDS>::underlying_span() const noexcept
{
  return this->view_;
}

}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_FIXED_SIZE_MATRIX_ENGINE_HPP
