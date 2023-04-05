//==================================================================================================
//  File:       mdspan_improvements.hpp
//
//  Summary:    Defines improvements to mdspan
//==================================================================================================
//
#ifndef MDSPAN_IMPROVEMENTS_HPP
#define MDSPAN_IMPROVEMENTS_HPP

//- Disable some unnecessary compiler warnings coming from mdspan.
//
#if defined __clang__
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#elif defined __GNUG__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4100)
#endif

#include <experimental/mdspan>
using std::experimental::dynamic_extent;

namespace std
{
namespace experimental
{

template < class IndexType, size_t Extent >
class extents<IndexType,Extent>
{
public:
  // typedefs for integral types used
  using index_type = IndexType;
  using size_type = make_unsigned_t<index_type>;
  using rank_type = size_t;

  static_assert(std::is_integral<index_type>::value && !std::is_same<index_type, bool>::value,
                "extents::index_type must be a signed or unsigned integer type");
private:
  constexpr static rank_type m_rank = 1;
  constexpr static rank_type m_rank_dynamic = ( Extent == dynamic_extent );

  // internal storage type using maybe_static_array
  using vals_t =
      detail::maybe_static_array<IndexType, size_t, dynamic_extent, Extent >;
  _MDSPAN_NO_UNIQUE_ADDRESS vals_t m_vals;

public:
  // [mdspan.extents.obs], observers of multidimensional index space
  MDSPAN_INLINE_FUNCTION
  constexpr static rank_type rank() noexcept { return m_rank; }
  MDSPAN_INLINE_FUNCTION
  constexpr static rank_type rank_dynamic() noexcept { return m_rank_dynamic; }

  MDSPAN_INLINE_FUNCTION
  constexpr index_type extent(rank_type r) const noexcept { return m_vals.value(r); }
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t static_extent(rank_type r) noexcept {
    return vals_t::static_value(r);
  }

  // [mdspan.extents.cons], constructors
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr extents() noexcept = default;

  // Construction from just dynamic or all values.
  // Precondition check is deferred to maybe_static_array constructor
  MDSPAN_TEMPLATE_REQUIRES(
      class... OtherIndexTypes,
      /* requires */ (
          _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_convertible, OtherIndexTypes,
                                         index_type) /* && ... */) &&
          _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_nothrow_constructible, index_type,
                                         OtherIndexTypes) /* && ... */) &&
          (sizeof...(OtherIndexTypes) == m_rank ||
           sizeof...(OtherIndexTypes) == m_rank_dynamic)))
  MDSPAN_INLINE_FUNCTION
  constexpr extents(OtherIndexTypes... dynvals) noexcept
      : m_vals(static_cast<index_type>(dynvals)...) {}

  MDSPAN_TEMPLATE_REQUIRES(
      class OtherIndexType, size_t N,
      /* requires */
      (
          _MDSPAN_TRAIT(is_convertible, OtherIndexType, index_type) &&
          _MDSPAN_TRAIT(is_nothrow_constructible, index_type,
              OtherIndexType) &&
          (N == m_rank || N == m_rank_dynamic)))
  MDSPAN_INLINE_FUNCTION
  MDSPAN_CONDITIONAL_EXPLICIT(N != m_rank_dynamic)
  constexpr extents(const array<OtherIndexType, N> &exts) noexcept
      : m_vals(std::move(exts)) {}

#ifdef __cpp_lib_span
  MDSPAN_TEMPLATE_REQUIRES(
      class OtherIndexType, size_t N,
      /* requires */
      (_MDSPAN_TRAIT(is_convertible, OtherIndexType, index_type) &&
       _MDSPAN_TRAIT(is_nothrow_constructible, index_type, OtherIndexType) &&
       (N == m_rank || N == m_rank_dynamic)))
  MDSPAN_INLINE_FUNCTION
  MDSPAN_CONDITIONAL_EXPLICIT(N != m_rank_dynamic)
  constexpr extents(const span<OtherIndexType, N> &exts) noexcept
      : m_vals(std::move(exts)) {}
#endif

private:
  // Function to construct extents storage from other extents.
  // With C++ 17 the first two variants could be collapsed using if constexpr
  // in which case you don't need all the requires clauses.
  // in C++ 14 mode that doesn't work due to infinite recursion
  MDSPAN_TEMPLATE_REQUIRES(
      size_t DynCount, size_t R, class OtherExtents, class... DynamicValues,
      /* requires */ ((R < m_rank) && (static_extent(R) == dynamic_extent)))
  MDSPAN_INLINE_FUNCTION
  vals_t __construct_vals_from_extents(std::integral_constant<size_t, DynCount>,
                                       std::integral_constant<size_t, R>,
                                       const OtherExtents &exts,
                                       DynamicValues... dynamic_values) noexcept {
    return __construct_vals_from_extents(
        std::integral_constant<size_t, DynCount + 1>(),
        std::integral_constant<size_t, R + 1>(), exts, dynamic_values...,
        exts.extent(R));
  }

  MDSPAN_TEMPLATE_REQUIRES(
      size_t DynCount, size_t R, class OtherExtents, class... DynamicValues,
      /* requires */ ((R < m_rank) && (static_extent(R) != dynamic_extent)))
  MDSPAN_INLINE_FUNCTION
  vals_t __construct_vals_from_extents(std::integral_constant<size_t, DynCount>,
                                       std::integral_constant<size_t, R>,
                                       const OtherExtents &exts,
                                       DynamicValues... dynamic_values) noexcept {
    return __construct_vals_from_extents(
        std::integral_constant<size_t, DynCount>(),
        std::integral_constant<size_t, R + 1>(), exts, dynamic_values...);
  }

  MDSPAN_TEMPLATE_REQUIRES(
      size_t DynCount, size_t R, class OtherExtents, class... DynamicValues,
      /* requires */ ((R == m_rank) && (DynCount == m_rank_dynamic)))
  MDSPAN_INLINE_FUNCTION
  vals_t __construct_vals_from_extents(std::integral_constant<size_t, DynCount>,
                                       std::integral_constant<size_t, R>,
                                       const OtherExtents &,
                                       DynamicValues... dynamic_values) noexcept {
    return vals_t{static_cast<index_type>(dynamic_values)...};
  }

public:

  // Converting constructor from other extents specializations
  MDSPAN_TEMPLATE_REQUIRES(
      class OtherIndexType, size_t... OtherExtents,
      /* requires */
      (
          /* multi-stage check to protect from invalid pack expansion when sizes
             don't match? */
          decltype(detail::__check_compatible_extents(
              std::integral_constant<bool, 1 ==
                                               sizeof...(OtherExtents)>{},
              std::integer_sequence<size_t, Extent>{},
              std::integer_sequence<size_t, OtherExtents...>{}))::value))
  MDSPAN_INLINE_FUNCTION
  MDSPAN_CONDITIONAL_EXPLICIT((((Extent != dynamic_extent) &&
                                (OtherExtents == dynamic_extent)) ||
                               ...) ||
                              (std::numeric_limits<index_type>::max() <
                               std::numeric_limits<OtherIndexType>::max()))
  constexpr extents(const extents<OtherIndexType, OtherExtents...> &other) noexcept
      : m_vals(__construct_vals_from_extents(
            std::integral_constant<size_t, 0>(),
            std::integral_constant<size_t, 0>(), other)) {}

  // Comparison operator
  template <class OtherIndexType, size_t... OtherExtents>
  MDSPAN_INLINE_FUNCTION friend constexpr bool
  operator==(const extents &lhs,
             const extents<OtherIndexType, OtherExtents...> &rhs) noexcept {
    bool value = true;
    for (size_type r = 0; r < m_rank; r++)
      value &= rhs.extent(r) == lhs.extent(r);
    return value;
  }

#if !(MDSPAN_HAS_CXX_20)
  template <class OtherIndexType, size_t... OtherExtents>
  MDSPAN_INLINE_FUNCTION friend constexpr bool
  operator!=(extents const &lhs,
             extents<OtherIndexType, OtherExtents...> const &rhs) noexcept {
    return !(lhs == rhs);
  }
#endif

  // SPECIALIZATIONS FOR SIZEOF...(EXTENTS) == 1

  // Implicit conversion to underlying index type
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr
  operator ThisIndexType() noexcept { return __extent<0>(); }

  // Comparison operators
  template < class OtherIndexType >
  MDSPAN_INLINE_FUNCTION
  constexpr bool operator==( OtherIndexType const& rhs ) noexcept requires ( !detail::__is_extents_v<OtherIndexType> ) {
    return this->extent(0) == rhs;
  }
};

} //- experimental
} //- std namespace


//- Restore the compiler's diagnostic state.
//
#if defined __clang__
    #pragma clang diagnostic pop
#elif defined __GNUG__
    #pragma GCC diagnostic pop
#elif defined _MSC_VER
    #pragma warning(pop)
#endif

#endif  //- MDSPAN_IMPROVEMENTS_HPP
