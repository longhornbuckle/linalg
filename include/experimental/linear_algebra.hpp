//==================================================================================================
//  File:       linear_algebra.hpp
//
//  Summary:    This is a driver header for including all of the linear algebra facilities
//              defined by the library.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include <algorithm>
#include <array>
#include <concepts>
#include <complex>
#include <cstddef>
#include <execution>
#include <exception>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>

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
#include <experimental/mdspan_improvements.hpp>
using std::experimental::dynamic_extent;

//- Restore the compiler's diagnostic state.
//
#if defined __clang__
    #pragma clang diagnostic pop
#elif defined __GNUG__
    #pragma GCC diagnostic pop
#elif defined _MSC_VER
    #pragma warning(pop)
#endif

//- Implementation headers.
//
#include "linear_algebra/config.hpp"
#include "linear_algebra/macros.hpp"
#include "linear_algebra/private_support.hpp"
#include "linear_algebra/tensor_concepts.hpp"
#include "linear_algebra/vector_concepts.hpp"
#include "linear_algebra/matrix_concepts.hpp"
#include "linear_algebra/tensor_view.hpp"
#include "linear_algebra/vector_view.hpp"
#include "linear_algebra/matrix_view.hpp"
#include "linear_algebra/fixed_size_tensor.hpp"
#include "linear_algebra/dynamic_tensor.hpp"
#include "linear_algebra/fixed_size_matrix.hpp"
#include "linear_algebra/dynamic_matrix.hpp"
#include "linear_algebra/fixed_size_vector.hpp"
#include "linear_algebra/dynamic_vector.hpp"
#include "linear_algebra/instant_evaluated_operations.hpp"
namespace std::math::operations { using namespace std::math::instant_evaluated_operations; }
#include "linear_algebra/arithmetic_operators.hpp"

#endif  //- LINEAR_ALGEBRA_HPP
