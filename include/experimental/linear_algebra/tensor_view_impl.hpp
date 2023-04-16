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

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
constexpr tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
constexpr tensor_view<MDS,Dummy>::
#endif
tensor_view( const underlying_span_type& view ) noexcept :
  view_(view)
{
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
constexpr tensor_view<MDS>& tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
constexpr tensor_view<MDS,Dummy>& tensor_view<MDS,Dummy>::
#endif
operator = ( const underlying_span_type& rhs ) noexcept
{
  static_cast<void>( this->view_ = rhs );
  return *this;
}

//- Size / Capacity

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
[[nodiscard]] constexpr typename tensor_view<MDS>::extents_type tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
[[nodiscard]] constexpr typename tensor_view<MDS,Dummy>::extents_type tensor_view<MDS,Dummy>::
#endif
size() const noexcept
{
  return this->view_.extents();
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
[[nodiscard]] constexpr typename tensor_view<MDS>::extents_type tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
[[nodiscard]] constexpr typename tensor_view<MDS,Dummy>::extents_type tensor_view<MDS,Dummy>::
#endif
capacity() const noexcept
{
  return this->view_.extents();
}

//- Const views

#if LINALG_USE_BRACKET_OPERATOR
#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... IndexType >
[[nodiscard]] constexpr typename tensor_view<MDS>::value_type tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
template < class ... IndexType >
[[nodiscard]] constexpr typename tensor_view<MDS,Dummy>::value_type tensor_view<MDS,Dummy>::
#endif
operator[]( IndexType ... indices ) const noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(IndexType) == tensor_view<MDS>::extents_type::rank() ) && ( ::std::is_convertible_v<IndexType,typename tensor_view<MDS>::index_type> && ... )
#endif
{
  return this->underlying_span()[ indices ... ];
}
#endif

#if LINALG_USE_PAREN_OPERATOR
#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... IndexType >
[[nodiscard]] constexpr typename tensor_view<MDS>::value_type tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
template < class ... IndexType >
[[nodiscard]] constexpr typename tensor_view<MDS,Dummy>::value_type tensor_view<MDS,Dummy>::
#endif
operator()( IndexType ... indices ) const noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(IndexType) == tensor_view<MDS>::extents_type::rank() ) && ( ::std::is_convertible_v<IndexType,typename tensor_view<MDS>::index_type> && ... )
#endif
{
  return this->underlying_span()( indices ... );
}
#endif

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS,Dummy>::
#endif
subvector( SliceArgs ... args ) const
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 1 )
#endif
{
  using subspan_type = decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
  return vector_view<subspan_type>( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS,Dummy>::
#endif
submatrix( SliceArgs ... args ) const
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 2 )
#endif
{
  using subspan_type = decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
  return matrix_view<subspan_type>( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS,Dummy>::
#endif
subtensor( SliceArgs ... args ) const
{
  using subspan_type = decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
  return tensor_view<subspan_type>( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
}

//- Mutable views

#if LINALG_USE_BRACKET_OPERATOR
#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... IndexType >
[[nodiscard]] constexpr typename tensor_view<MDS>::reference tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
template < class ... IndexType >
[[nodiscard]] constexpr typename tensor_view<MDS,Dummy>::reference tensor_view<MDS,Dummy>::
#endif
operator[]( IndexType ... indices ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(IndexType) == tensor_view<MDS>::extents_type::rank() ) && ( ::std::is_convertible_v<IndexType,typename tensor_view<MDS>::index_type> && ... ) &&
           ( !::std::is_const_v<typename tensor_view<MDS>::element_type> )
#endif
{
  return ::std::forward<reference>( this->underlying_span()[ indices ... ] );
}
#endif

#if LINALG_USE_PAREN_OPERATOR
#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... IndexType >
[[nodiscard]] constexpr typename tensor_view<MDS>::reference tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
template < class ... IndexType >
[[nodiscard]] constexpr typename tensor_view<MDS,Dummy>::reference tensor_view<MDS,Dummy>::
#endif
operator()( IndexType ... indices ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(IndexType) == tensor_view<MDS>::extents_type::rank() ) && ( ::std::is_convertible_v<IndexType,typename tensor_view<MDS>::index_type> && ... ) &&
           ( !::std::is_const_v<typename tensor_view<MDS>::element_type> )
#endif
{
  return ::std::forward<reference>( this->underlying_span()( indices ... ) );
}
#endif

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS,Dummy>::
#endif
subvector( SliceArgs ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 1 )
#endif
{
  using subspan_type = decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
  return vector_view<subspan_type>( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS,Dummy>::
#endif
submatrix( SliceArgs ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 2 )
#endif
{
  using subspan_type = decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
  return matrix_view<subspan_type>( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
template < class ... SliceArgs >
[[nodiscard]] constexpr auto tensor_view<MDS,Dummy>::
#endif
subtensor( SliceArgs ... args )
{
  using subspan_type = decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
  return tensor_view<subspan_type>( ::std::experimental::submdspan( this->underlying_span(), args ... ) );
}

//- Data access

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
[[nodiscard]] constexpr typename tensor_view<MDS>::span_type tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
[[nodiscard]] constexpr typename tensor_view<MDS,Dummy>::span_type tensor_view<MDS,Dummy>::
#endif
span() const noexcept
{
  return this->underlying_span();
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
[[nodiscard]] constexpr typename tensor_view<MDS>::underlying_span_type tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
template < typename Elem, typename >
[[nodiscard]] constexpr typename tensor_view<MDS,Dummy>::underlying_span_type tensor_view<MDS,Dummy>::
#endif
underlying_span() noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( !::std::is_const_v<typename tensor_view<MDS>::element_type> )
#endif
{
  return this->view_;
}

#ifdef LINALG_ENABLE_CONCEPTS
template < class MDS > requires ( detail::is_mdspan_v<MDS> && MDS::is_always_unique() )
[[nodiscard]] constexpr typename tensor_view<MDS>::const_underlying_span_type tensor_view<MDS>::
#else
template < class MDS, typename Dummy >
[[nodiscard]] constexpr typename tensor_view<MDS,Dummy>::const_underlying_span_type tensor_view<MDS,Dummy>::
#endif
underlying_span() const noexcept
{
  return this->view_;
}

}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_FIXED_SIZE_MATRIX_ENGINE_HPP
