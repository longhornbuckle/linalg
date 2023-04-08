//==================================================================================================
//  File:       tensor_view.hpp
//
//  Summary:    This header defines the tensor view implementation
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_TENSOR_VIEW_IMPL_HPP
#define LINEAR_ALGEBRA_TENSOR_VIEW_IMPL_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{

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
[[nodiscard]] constexpr typename tensor_view<MDS>::value_type
tensor_view<MDS>::at( IndexType ... indices ) const
  requires ( sizeof...(IndexType) == tensor_view<MDS>::extents_type::rank() ) && ( is_convertible_v<IndexType,typename tensor_view<MDS>::index_type> && ... )
{
  return detail::access( this->underlying_span(), indices ... );
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS>::subvector( SliceArgs ... args ) const
  requires ( decltype( experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 1 )
{
  using subspan_type = decltype( experimental::submdspan( this->underlying_span(), args ... ) );
  return vector_view<subspan_type>( experimental::submdspan( this->underlying_span(), args ... ) );
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS>::submatrix( SliceArgs ... args ) const
  requires ( decltype( experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 2 )
{
  using subspan_type = decltype( experimental::submdspan( this->underlying_span(), args ... ) );
  return matrix_view<subspan_type>( experimental::submdspan( this->underlying_span(), args ... ) );
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS>::subtensor( SliceArgs ... args ) const
{
  using subspan_type = decltype( experimental::submdspan( this->underlying_span(), args ... ) );
  return tensor_view<subspan_type>( experimental::submdspan( this->underlying_span(), args ... ) );
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
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS>::subvector( SliceArgs ... args )
  requires ( decltype( experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 1 )
{
  using subspan_type = decltype( experimental::submdspan( this->underlying_span(), args ... ) );
  return vector_view<subspan_type>( experimental::submdspan( this->underlying_span(), args ... ) );
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS>::submatrix( SliceArgs ... args )
  requires ( decltype( experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 2 )
{
  using subspan_type = decltype( experimental::submdspan( this->underlying_span(), args ... ) );
  return matrix_view<subspan_type>( experimental::submdspan( this->underlying_span(), args ... ) );
}

template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS>::subtensor( SliceArgs ... args )
{
  using subspan_type = decltype( experimental::submdspan( this->underlying_span(), args ... ) );
  return tensor_view<subspan_type>( experimental::submdspan( this->underlying_span(), args ... ) );
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
