#include <gtest/gtest.h>
#include <experimental/linear_algebra.hpp>

namespace
{
  TEST( DR_TENSOR, DEFAULT_CONSTRUCTOR_AND_DESTRUCTOR )
  {
    // Default construction
    std::math::dr_tensor<double,2> dyn_tensor;
    // Destructor will be called when unit test ends and the dr tensor exits scope
  }

  TEST( DR_TENSOR, MUTABLE_AND_CONST_INDEX_ACCESS )
  {
    // Construct
    std::math::dr_tensor<double,3> dyn_tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( dyn_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( dyn_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( dyn_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( dyn_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( dyn_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( dyn_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( dyn_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( dyn_tensor, 1, 1, 1 ) = 8.0;
    // Get a const reference
    const std::math::dr_tensor<double,3>& const_dyn_tensor( dyn_tensor );
    // Access elements from const fs tensor
    auto val1 = std::math::detail::access( const_dyn_tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( const_dyn_tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( const_dyn_tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( const_dyn_tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( const_dyn_tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( const_dyn_tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( const_dyn_tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( const_dyn_tensor, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( DR_TENSOR, COPY_CONSTRUCTOR )
  {
    // Construct
    std::math::dr_tensor<double,3> dyn_tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( dyn_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( dyn_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( dyn_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( dyn_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( dyn_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( dyn_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( dyn_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( dyn_tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    std::math::dr_tensor<double,3> dyn_tensor_copy{ dyn_tensor };
    // Get a const reference to copy
    const std::math::dr_tensor<double,3>& const_dyn_tensor( dyn_tensor_copy );
    // Access elements from const dyn tensor
    auto val1 = std::math::detail::access( const_dyn_tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( const_dyn_tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( const_dyn_tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( const_dyn_tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( const_dyn_tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( const_dyn_tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( const_dyn_tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( const_dyn_tensor, 1, 1, 1 );
    // Check the dyn tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( DR_TENSOR, MOVE_CONSTRUCTOR )
  {
    // Construct
    std::math::dr_tensor<double,3> dyn_tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( dyn_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( dyn_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( dyn_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( dyn_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( dyn_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( dyn_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( dyn_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( dyn_tensor, 1, 1, 1 ) = 8.0;
    // Move construct
    std::math::dr_tensor<double,3> dyn_tensor_move{ std::move( dyn_tensor ) };
    // Get a const reference to moved tensor
    const std::math::dr_tensor<double,3>& const_dyn_tensor( dyn_tensor_move );
    // Access elements from const fs tensor
    auto val1 = std::math::detail::access( const_dyn_tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( const_dyn_tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( const_dyn_tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( const_dyn_tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( const_dyn_tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( const_dyn_tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( const_dyn_tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( const_dyn_tensor, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( DR_TENSOR, CONSTRUCT_FROM_VIEW )
  {
    // Construct
    std::math::dr_tensor<double,3> dyn_tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( dyn_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( dyn_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( dyn_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( dyn_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( dyn_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( dyn_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( dyn_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( dyn_tensor, 1, 1, 1 ) = 8.0;
    // Construct from view
    std::math::dr_tensor<double,3> dyn_tensor_view{ dyn_tensor.span() };
    // Get a const reference to constructed tensor
    const std::math::dr_tensor<double,3>& const_dyn_tensor( dyn_tensor_view );
    // Access elements from const dyn tensor
    auto val1 = std::math::detail::access( const_dyn_tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( const_dyn_tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( const_dyn_tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( const_dyn_tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( const_dyn_tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( const_dyn_tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( const_dyn_tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( const_dyn_tensor, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( DR_TENSOR, TEMPLATE_COPY_CONSTRUCTOR )
  {
    using float_left_tensor_type   = std::math::fs_tensor<float,std::experimental::layout_right,std::experimental::default_accessor<float>,2,2,2>;
    using double_right_tensor_type = std::math::dr_tensor<double,3>;
    // Default construct
    float_left_tensor_type fs_tensor;
    // Populate via mutable index access
    std::math::detail::access( fs_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( fs_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( fs_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( fs_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( fs_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( fs_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( fs_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Construct from float tensor
    double_right_tensor_type dyn_tensor_copy{ fs_tensor };
    // Access elements from const fs tensor
    auto val1 = std::math::detail::access( dyn_tensor_copy, 0, 0, 0 );
    auto val2 = std::math::detail::access( dyn_tensor_copy, 0, 0, 1 );
    auto val3 = std::math::detail::access( dyn_tensor_copy, 0, 1, 0 );
    auto val4 = std::math::detail::access( dyn_tensor_copy, 0, 1, 1 );
    auto val5 = std::math::detail::access( dyn_tensor_copy, 1, 0, 0 );
    auto val6 = std::math::detail::access( dyn_tensor_copy, 1, 0, 1 );
    auto val7 = std::math::detail::access( dyn_tensor_copy, 1, 1, 0 );
    auto val8 = std::math::detail::access( dyn_tensor_copy, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( DR_TENSOR, CONSTRUCT_FROM_LAMBDA_EXPRESSION )
  {
    using fs_tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,2,2,2>;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    std::math::detail::access( fs_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( fs_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( fs_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( fs_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( fs_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( fs_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( fs_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Get underling view
    auto view = fs_tensor.span();
    // Create a lambda expression from view
    auto lambda = [&view]< class ... SizeType >( SizeType ... indices ) { return std::math::detail::access( view, indices ... ); };
    // Construct from lambda
    std::math::dr_tensor<double,3> dyn_tensor_copy( std::experimental::extents<size_t,2,2,2>(), lambda );
    // Access elements from const fs tensor tensor
    auto val1 = std::math::detail::access( dyn_tensor_copy, 0, 0, 0 );
    auto val2 = std::math::detail::access( dyn_tensor_copy, 0, 0, 1 );
    auto val3 = std::math::detail::access( dyn_tensor_copy, 0, 1, 0 );
    auto val4 = std::math::detail::access( dyn_tensor_copy, 0, 1, 1 );
    auto val5 = std::math::detail::access( dyn_tensor_copy, 1, 0, 0 );
    auto val6 = std::math::detail::access( dyn_tensor_copy, 1, 0, 1 );
    auto val7 = std::math::detail::access( dyn_tensor_copy, 1, 1, 0 );
    auto val8 = std::math::detail::access( dyn_tensor_copy, 1, 1, 1 );
    // Check the fs tensor tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( DR_TENSOR, ASSIGNMENT_OPERATOR )
  {
    // Construct
    std::math::dr_tensor<double,3> dyn_tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( dyn_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( dyn_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( dyn_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( dyn_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( dyn_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( dyn_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( dyn_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( dyn_tensor, 1, 1, 1 ) = 8.0;
    // Default construct and assign
    std::math::dr_tensor<double,3> dyn_tensor_copy;
    dyn_tensor_copy = dyn_tensor;
    // Access elements from dyn tensor
    auto val1 = std::math::detail::access( dyn_tensor_copy, 0, 0, 0 );
    auto val2 = std::math::detail::access( dyn_tensor_copy, 0, 0, 1 );
    auto val3 = std::math::detail::access( dyn_tensor_copy, 0, 1, 0 );
    auto val4 = std::math::detail::access( dyn_tensor_copy, 0, 1, 1 );
    auto val5 = std::math::detail::access( dyn_tensor_copy, 1, 0, 0 );
    auto val6 = std::math::detail::access( dyn_tensor_copy, 1, 0, 1 );
    auto val7 = std::math::detail::access( dyn_tensor_copy, 1, 1, 0 );
    auto val8 = std::math::detail::access( dyn_tensor_copy, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( DR_TENSOR, TEMPLATE_ASSIGNMENT_OPERATOR )
  {
    using float_left_tensor_type = std::math::fs_tensor<float,std::experimental::layout_right,std::experimental::default_accessor<float>,2,2,2>;
    // Default construct
    float_left_tensor_type fs_tensor;
    // Populate via mutable index access
    std::math::detail::access( fs_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( fs_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( fs_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( fs_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( fs_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( fs_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( fs_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Default construct and then assign
    std::math::dr_tensor<double,3> dyn_tensor_copy;
    dyn_tensor_copy = fs_tensor;
    // Access elements from const fs tensor tensor
    auto val1 = std::math::detail::access( dyn_tensor_copy, 0, 0, 0 );
    auto val2 = std::math::detail::access( dyn_tensor_copy, 0, 0, 1 );
    auto val3 = std::math::detail::access( dyn_tensor_copy, 0, 1, 0 );
    auto val4 = std::math::detail::access( dyn_tensor_copy, 0, 1, 1 );
    auto val5 = std::math::detail::access( dyn_tensor_copy, 1, 0, 0 );
    auto val6 = std::math::detail::access( dyn_tensor_copy, 1, 0, 1 );
    auto val7 = std::math::detail::access( dyn_tensor_copy, 1, 1, 0 );
    auto val8 = std::math::detail::access( dyn_tensor_copy, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( DR_TENSOR, ASSIGN_FROM_VIEW )
  {
    using fs_tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,2,2,2>;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    std::math::detail::access( fs_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( fs_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( fs_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( fs_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( fs_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( fs_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( fs_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Construct and assign from view
    std::math::dr_tensor<double,3> dyn_tensor_view;
    dyn_tensor_view = fs_tensor.span();
    // Access elements from const fs tensor
    auto val1 = std::math::detail::access( dyn_tensor_view, 0, 0, 0 );
    auto val2 = std::math::detail::access( dyn_tensor_view, 0, 0, 1 );
    auto val3 = std::math::detail::access( dyn_tensor_view, 0, 1, 0 );
    auto val4 = std::math::detail::access( dyn_tensor_view, 0, 1, 1 );
    auto val5 = std::math::detail::access( dyn_tensor_view, 1, 0, 0 );
    auto val6 = std::math::detail::access( dyn_tensor_view, 1, 0, 1 );
    auto val7 = std::math::detail::access( dyn_tensor_view, 1, 1, 0 );
    auto val8 = std::math::detail::access( dyn_tensor_view, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( DR_TENSOR, SIZE_AND_CAPACITY )
  {
    // Construct
    std::math::dr_tensor<double,4> dyn_tensor{ std::experimental::extents<size_t,2,5,1,7>(), std::experimental::extents<size_t,3,5,2,10>() };
    EXPECT_TRUE( ( dyn_tensor.size().extent(0) == 2 ) );
    EXPECT_TRUE( ( dyn_tensor.size().extent(1) == 5 ) );
    EXPECT_TRUE( ( dyn_tensor.size().extent(2) == 1 ) );
    EXPECT_TRUE( ( dyn_tensor.size().extent(3) == 7 ) );
    EXPECT_TRUE( ( dyn_tensor.capacity().extent(0) == 3 ) );
    EXPECT_TRUE( ( dyn_tensor.capacity().extent(1) == 5 ) );
    EXPECT_TRUE( ( dyn_tensor.capacity().extent(2) == 2 ) );
    EXPECT_TRUE( ( dyn_tensor.capacity().extent(3) == 10 ) );
  }

  TEST( DR_TENSOR, RESIZE )
  {
    // Construct
    std::math::dr_tensor<double,3> dyn_tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( dyn_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( dyn_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( dyn_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( dyn_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( dyn_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( dyn_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( dyn_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( dyn_tensor, 1, 1, 1 ) = 8.0;
    // Resize
    dyn_tensor.resize( std::experimental::extents<size_t,3,3,3>() );
    std::math::detail::access( dyn_tensor, 0, 0, 2 ) = 9.0;
    std::math::detail::access( dyn_tensor, 0, 1, 2 ) = 10.0;
    std::math::detail::access( dyn_tensor, 0, 2, 0 ) = 11.0;
    std::math::detail::access( dyn_tensor, 0, 2, 1 ) = 12.0;
    std::math::detail::access( dyn_tensor, 0, 2, 2 ) = 13.0;
    std::math::detail::access( dyn_tensor, 1, 0, 2 ) = 14.0;
    std::math::detail::access( dyn_tensor, 1, 1, 2 ) = 15.0;
    std::math::detail::access( dyn_tensor, 1, 2, 0 ) = 16.0;
    std::math::detail::access( dyn_tensor, 1, 2, 1 ) = 17.0;
    std::math::detail::access( dyn_tensor, 1, 2, 2 ) = 18.0;
    std::math::detail::access( dyn_tensor, 2, 0, 0 ) = 19.0;
    std::math::detail::access( dyn_tensor, 2, 0, 1 ) = 20.0;
    std::math::detail::access( dyn_tensor, 2, 0, 2 ) = 21.0;
    std::math::detail::access( dyn_tensor, 2, 1, 0 ) = 22.0;
    std::math::detail::access( dyn_tensor, 2, 1, 1 ) = 23.0;
    std::math::detail::access( dyn_tensor, 2, 1, 2 ) = 24.0;
    std::math::detail::access( dyn_tensor, 2, 2, 0 ) = 25.0;
    std::math::detail::access( dyn_tensor, 2, 2, 1 ) = 26.0;
    std::math::detail::access( dyn_tensor, 2, 2, 2 ) = 27.0;
    // Get values
    auto val1  = std::math::detail::access( dyn_tensor, 0, 0, 0 );
    auto val2  = std::math::detail::access( dyn_tensor, 0, 0, 1 );
    auto val3  = std::math::detail::access( dyn_tensor, 0, 1, 0 );
    auto val4  = std::math::detail::access( dyn_tensor, 0, 1, 1 );
    auto val5  = std::math::detail::access( dyn_tensor, 1, 0, 0 );
    auto val6  = std::math::detail::access( dyn_tensor, 1, 0, 1 );
    auto val7  = std::math::detail::access( dyn_tensor, 1, 1, 0 );
    auto val8  = std::math::detail::access( dyn_tensor, 1, 1, 1 );
    auto val9  = std::math::detail::access( dyn_tensor, 0, 0, 2 );
    auto val10 = std::math::detail::access( dyn_tensor, 0, 1, 2 );
    auto val11 = std::math::detail::access( dyn_tensor, 0, 2, 0 );
    auto val12 = std::math::detail::access( dyn_tensor, 0, 2, 1 );
    auto val13 = std::math::detail::access( dyn_tensor, 0, 2, 2 );
    auto val14 = std::math::detail::access( dyn_tensor, 1, 0, 2 );
    auto val15 = std::math::detail::access( dyn_tensor, 1, 1, 2 );
    auto val16 = std::math::detail::access( dyn_tensor, 1, 2, 0 );
    auto val17 = std::math::detail::access( dyn_tensor, 1, 2, 1 );
    auto val18 = std::math::detail::access( dyn_tensor, 1, 2, 2 );
    auto val19 = std::math::detail::access( dyn_tensor, 2, 0, 0 );
    auto val20 = std::math::detail::access( dyn_tensor, 2, 0, 1 );
    auto val21 = std::math::detail::access( dyn_tensor, 2, 0, 2 );
    auto val22 = std::math::detail::access( dyn_tensor, 2, 1, 0 );
    auto val23 = std::math::detail::access( dyn_tensor, 2, 1, 1 );
    auto val24 = std::math::detail::access( dyn_tensor, 2, 1, 2 );
    auto val25 = std::math::detail::access( dyn_tensor, 2, 2, 0 );
    auto val26 = std::math::detail::access( dyn_tensor, 2, 2, 1 );
    auto val27 = std::math::detail::access( dyn_tensor, 2, 2, 2 );
    // Check the values are correct
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
    EXPECT_EQ( val9, 9.0 );
    EXPECT_EQ( val10, 10.0 );
    EXPECT_EQ( val11, 11.0 );
    EXPECT_EQ( val12, 12.0 );
    EXPECT_EQ( val13, 13.0 );
    EXPECT_EQ( val14, 14.0 );
    EXPECT_EQ( val15, 15.0 );
    EXPECT_EQ( val16, 16.0 );
    EXPECT_EQ( val17, 17.0 );
    EXPECT_EQ( val18, 18.0 );
    EXPECT_EQ( val19, 19.0 );
    EXPECT_EQ( val20, 20.0 );
    EXPECT_EQ( val21, 21.0 );
    EXPECT_EQ( val22, 22.0 );
    EXPECT_EQ( val23, 23.0 );
    EXPECT_EQ( val24, 24.0 );
    EXPECT_EQ( val25, 25.0 );
    EXPECT_EQ( val26, 26.0 );
    EXPECT_EQ( val27, 27.0 );
  }

  TEST( DR_TENSOR, RESERVE )
  {
    // Construct
    std::math::dr_tensor<double,3> dyn_tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,2,2,2>() };
    // Populate via mutable index access
    std::math::detail::access( dyn_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( dyn_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( dyn_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( dyn_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( dyn_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( dyn_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( dyn_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( dyn_tensor, 1, 1, 1 ) = 8.0;
    // Resize
    dyn_tensor.reserve( std::experimental::extents<size_t,4,4,4>() );
    // Get values
    auto val1  = std::math::detail::access( dyn_tensor, 0, 0, 0 );
    auto val2  = std::math::detail::access( dyn_tensor, 0, 0, 1 );
    auto val3  = std::math::detail::access( dyn_tensor, 0, 1, 0 );
    auto val4  = std::math::detail::access( dyn_tensor, 0, 1, 1 );
    auto val5  = std::math::detail::access( dyn_tensor, 1, 0, 0 );
    auto val6  = std::math::detail::access( dyn_tensor, 1, 0, 1 );
    auto val7  = std::math::detail::access( dyn_tensor, 1, 1, 0 );
    auto val8  = std::math::detail::access( dyn_tensor, 1, 1, 1 );
    // Check the values are correct
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( DR_TENSOR, CONST_SUBVECTOR )
  {
    // Construct
    std::math::dr_tensor<double,3> dyn_tensor{ std::experimental::extents<size_t,5,5,5>(), std::experimental::extents<size_t,10,10,10>() };
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::math::detail::access( dyn_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const std::math::dr_tensor<double,3>& const_dyn_tensor( dyn_tensor );
    auto subvector = const_dyn_tensor.subvector( 0, std::experimental::full_extent, 1 );
    
    EXPECT_EQ( ( std::math::detail::access( subvector, 0 ) ), ( std::math::detail::access( dyn_tensor, 0, 0, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subvector, 1 ) ), ( std::math::detail::access( dyn_tensor, 0, 1, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subvector, 2 ) ), ( std::math::detail::access( dyn_tensor, 0, 2, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subvector, 3 ) ), ( std::math::detail::access( dyn_tensor, 0, 3, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subvector, 4 ) ), ( std::math::detail::access( dyn_tensor, 0, 4, 1 ) ) );
  }

  TEST( DR_TENSOR, CONST_SUBMATRIX )
  {
    // Construct
    std::math::dr_tensor<double,3> dyn_tensor{ std::experimental::extents<size_t,5,5,5>(), std::experimental::extents<size_t,10,10,10>() };
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::math::detail::access( dyn_tensor, i,j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const std::math::dr_tensor<double,3>& const_dyn_tensor( dyn_tensor );
    auto submatrix = const_dyn_tensor.submatrix( 0, std::experimental::full_extent, std::tuple(0,1) );
    
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 0 ) ), ( std::math::detail::access( dyn_tensor, 0, 0, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 0 ) ), ( std::math::detail::access( dyn_tensor, 0, 1, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 0 ) ), ( std::math::detail::access( dyn_tensor, 0, 2, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 3, 0 ) ), ( std::math::detail::access( dyn_tensor, 0, 3, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 4, 0 ) ), ( std::math::detail::access( dyn_tensor, 0, 4, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 1 ) ), ( std::math::detail::access( dyn_tensor, 0, 0, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 1 ) ), ( std::math::detail::access( dyn_tensor, 0, 1, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 1 ) ), ( std::math::detail::access( dyn_tensor, 0, 2, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 3, 1 ) ), ( std::math::detail::access( dyn_tensor, 0, 3, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 4, 1 ) ), ( std::math::detail::access( dyn_tensor, 0, 4, 1 ) ) );
  }

  TEST( DR_TENSOR, CONST_SUBTENSOR )
  {
    // Construct
    std::math::dr_tensor<double,3> dyn_tensor{ std::experimental::extents<size_t,5,5,5>(), std::experimental::extents<size_t,10,10,10>() };
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::math::detail::access( dyn_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const std::math::dr_tensor<double,3>& const_dyn_tensor( dyn_tensor );
    auto subtensor = const_dyn_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,3) );
    
    EXPECT_EQ( ( std::math::detail::access( subtensor, 0, 0, 0 ) ), ( std::math::detail::access( dyn_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 1, 0, 0 ) ), ( std::math::detail::access( dyn_tensor, 3, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 2, 0, 0 ) ), ( std::math::detail::access( dyn_tensor, 4, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 0, 1, 0 ) ), ( std::math::detail::access( dyn_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 1, 1, 0 ) ), ( std::math::detail::access( dyn_tensor, 3, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 2, 1, 0 ) ), ( std::math::detail::access( dyn_tensor, 4, 3, 2 ) ) );
  }

  TEST( DR_TENSOR, SUBVECTOR )
  {
    // Construct
    std::math::dr_tensor<double,3> dyn_tensor{ std::experimental::extents<size_t,5,5,5>(), std::experimental::extents<size_t,10,10,10>() };
    // Set values in tensor
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::math::detail::access( dyn_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Get subvector
    auto subvector = dyn_tensor.subvector( 1, std::experimental::full_extent, 0 );
    // Modify view
    for ( auto i : { 1, 2, 3 } )
    {
      std::math::detail::access( subvector, i ) = val;
      val = 2 * val;
    }
    // Assert original tensor has been modified as well
    EXPECT_EQ( ( std::math::detail::access( subvector, 1 ) ), ( std::math::detail::access( dyn_tensor, 1, 1, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subvector, 2 ) ), ( std::math::detail::access( dyn_tensor, 1, 2, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subvector, 3 ) ), ( std::math::detail::access( dyn_tensor, 1, 3, 0 ) ) );
  }

  TEST( DR_TENSOR, SUBMATRIX )
  {
    // Construct
    std::math::dr_tensor<double,3> dyn_tensor{ std::experimental::extents<size_t,5,5,5>(), std::experimental::extents<size_t,10,10,10>() };
    // Set values in tensor
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::math::detail::access( dyn_tensor, i,j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Get submatrix
    auto submatrix = dyn_tensor.submatrix( 1, std::experimental::full_extent, std::tuple( 1, 4 ) );
    // Modify view
    for ( auto i : { 1, 2, 3 } )
    {
      for ( auto j : { 0, 1 } )
      {
        std::math::detail::access( submatrix, i, j ) = val;
        val = 2 * val;
      }
    }
    // Assert original tensor has been modified as well
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 0 ) ), ( std::math::detail::access( dyn_tensor, 1, 1, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 0 ) ), ( std::math::detail::access( dyn_tensor, 1, 2, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 3, 0 ) ), ( std::math::detail::access( dyn_tensor, 1, 3, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 1 ) ), ( std::math::detail::access( dyn_tensor, 1, 1, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 1 ) ), ( std::math::detail::access( dyn_tensor, 1, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 3, 1 ) ), ( std::math::detail::access( dyn_tensor, 1, 3, 2 ) ) );
  }

  TEST( DR_TENSOR, SUBTENSOR )
  {
    // Construct
    std::math::dr_tensor<double,3> dyn_tensor{ std::experimental::extents<size_t,5,5,5>(), std::experimental::extents<size_t,10,10,10>() };
    // Set values in tensor
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::math::detail::access( dyn_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Get subtensor
    auto subtensor = dyn_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,3) );
    // Modify view
    for ( auto i : { 0, 1, 2 } )
    {
      for ( auto j : { 0, 1 } )
      {
        for ( auto k : { 0 } )
        {
          std::math::detail::access( subtensor, i,j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Assert original tensor has been modified as well
    EXPECT_EQ( ( std::math::detail::access( subtensor, 0, 0, 0 ) ), ( std::math::detail::access( dyn_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 1, 0, 0 ) ), ( std::math::detail::access( dyn_tensor, 3, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 2, 0, 0 ) ), ( std::math::detail::access( dyn_tensor, 4, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 0, 1, 0 ) ), ( std::math::detail::access( dyn_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 1, 1, 0 ) ), ( std::math::detail::access( dyn_tensor, 3, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 2, 1, 0 ) ), ( std::math::detail::access( dyn_tensor, 4, 3, 2 ) ) );
  }

  TEST( DR_TENSOR, NEGATION )
  {
    using tensor_type = std::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Negate the tensor
    tensor_type negate_tensor { -tensor };
    // Access elements from const tensor
    auto val1 = std::math::detail::access( negate_tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( negate_tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( negate_tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( negate_tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( negate_tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( negate_tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( negate_tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( negate_tensor, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, -1.0 );
    EXPECT_EQ( val2, -2.0 );
    EXPECT_EQ( val3, -3.0 );
    EXPECT_EQ( val4, -4.0 );
    EXPECT_EQ( val5, -5.0 );
    EXPECT_EQ( val6, -6.0 );
    EXPECT_EQ( val7, -7.0 );
    EXPECT_EQ( val8, -8.0 );
  }

  TEST( DR_TENSOR, ADD )
  {
    using tensor_type = std::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Add the two tensors together
    tensor_type tensor_sum { tensor + tensor_copy };
    // Access elements from const tensor
    auto val1 = std::math::detail::access( tensor_sum, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor_sum, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor_sum, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor_sum, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor_sum, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor_sum, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor_sum, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor_sum, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( DR_TENSOR, ADD_ASSIGN )
  {
    using tensor_type = std::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Add the two tensors together
    static_cast<void>( tensor += tensor_copy );
    // Access elements from const tensor
    auto val1 = std::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( DR_TENSOR, SUBTRACT )
  {
    using tensor_type = std::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Subtract the two tensors
    tensor_type tensor_diff { tensor - tensor_copy };
    // Access elements from const tensor
    auto val1 = std::math::detail::access( tensor_diff, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor_diff, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor_diff, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor_diff, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor_diff, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor_diff, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor_diff, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor_diff, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
    EXPECT_EQ( val5, 0 );
    EXPECT_EQ( val6, 0 );
    EXPECT_EQ( val7, 0 );
    EXPECT_EQ( val8, 0 );
  }

  TEST( DR_TENSOR, SUBTRACT_ASSIGN )
  {
    using tensor_type = std::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Subtract the two tensors
    static_cast<void>( tensor -= tensor_copy );
    // Access elements from const tensor
    auto val1 = std::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
    EXPECT_EQ( val5, 0 );
    EXPECT_EQ( val6, 0 );
    EXPECT_EQ( val7, 0 );
    EXPECT_EQ( val8, 0 );
  }

  TEST( DR_TENSOR, SCALAR_PREMULTIPLY )
  {
    using tensor_type = std::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Pre multiply
    tensor_type tensor_prod { 2 * tensor };
    // Access elements from const tensor
    auto val1 = std::math::detail::access( tensor_prod, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor_prod, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor_prod, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor_prod, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor_prod, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor_prod, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor_prod, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor_prod, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( DR_TENSOR, SCALAR_POSTMULTIPLY )
  {
    using tensor_type = std::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Post multiply
    tensor_type tensor_prod { tensor * 2 };
    // Access elements from const tensor
    auto val1 = std::math::detail::access( tensor_prod, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor_prod, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor_prod, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor_prod, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor_prod, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor_prod, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor_prod, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor_prod, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( DR_TENSOR, SCALAR_MULTIPLY_ASSIGN )
  {
    using tensor_type = std::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Post multiply
    static_cast<void>( tensor *= 2 );
    // Access elements from tensor
    auto val1 = std::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( DR_TENSOR, SCALAR_DIVIDE )
  {
    using tensor_type = std::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Divide
    tensor_type tensor_divide { tensor / 2 };
    // Access elements from const tensor
    auto val1 = std::math::detail::access( tensor_divide, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor_divide, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor_divide, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor_divide, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor_divide, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor_divide, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor_divide, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor_divide, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
    EXPECT_EQ( val5, 2.5 );
    EXPECT_EQ( val6, 3.0 );
    EXPECT_EQ( val7, 3.5 );
    EXPECT_EQ( val8, 4.0 );
  }

  TEST( DR_TENSOR, SCALAR_DIVIDE_ASSIGN )
  {
    using tensor_type = std::math::dr_tensor<double,3>;
    // Construct
    tensor_type tensor{ std::experimental::extents<size_t,2,2,2>(), std::experimental::extents<size_t,3,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Divide
    static_cast<void>( tensor /= 2 );
    // Access elements from tensor
    auto val1 = std::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
    EXPECT_EQ( val5, 2.5 );
    EXPECT_EQ( val6, 3.0 );
    EXPECT_EQ( val7, 3.5 );
    EXPECT_EQ( val8, 4.0 );
  }

  TEST( FS_TENSOR, DEFAULT_CONSTRUCTOR_AND_DESTRUCTOR )
  {
    // Default construction
    std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,2,2,2> fs_tensor;
    // Destructor will be called when unit test ends and the fs tensor exits scope
  }

  TEST( FS_TENSOR, MUTABLE_AND_CONST_INDEX_ACCESS )
  {
    using fs_tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,2,2,2>;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    std::math::detail::access( fs_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( fs_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( fs_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( fs_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( fs_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( fs_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( fs_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Get a const reference
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    // Access elements from const fs tensor
    auto val1 = std::math::detail::access( const_fs_tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( const_fs_tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( const_fs_tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( const_fs_tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( const_fs_tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( const_fs_tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( const_fs_tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( const_fs_tensor, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }
  
  TEST( FS_TENSOR, COPY_CONSTRUCTOR )
  {
    using fs_tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,2,2,2>;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    std::math::detail::access( fs_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( fs_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( fs_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( fs_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( fs_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( fs_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( fs_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    fs_tensor_type fs_tensor_copy{ fs_tensor };
    // Get a const reference to copy
    const fs_tensor_type& const_fs_tensor( fs_tensor_copy );
    // Access elements from const fs tensor
    auto val1 = std::math::detail::access( const_fs_tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( const_fs_tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( const_fs_tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( const_fs_tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( const_fs_tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( const_fs_tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( const_fs_tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( const_fs_tensor, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }
  
  TEST( FS_TENSOR, MOVE_CONSTRUCTOR )
  {
    using fs_tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,2,2,2>;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    std::math::detail::access( fs_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( fs_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( fs_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( fs_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( fs_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( fs_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( fs_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Move construct
    fs_tensor_type fs_tensor_move{ std::move( fs_tensor ) };
    // Get a const reference to moved tensor
    const fs_tensor_type& const_fs_tensor( fs_tensor_move );
    // Access elements from const fs tensor
    auto val1 = std::math::detail::access( const_fs_tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( const_fs_tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( const_fs_tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( const_fs_tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( const_fs_tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( const_fs_tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( const_fs_tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( const_fs_tensor, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }
  
  TEST( FS_TENSOR, CONSTRUCT_FROM_VIEW )
  {
    using fs_tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,2,2,2>;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    std::math::detail::access( fs_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( fs_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( fs_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( fs_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( fs_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( fs_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( fs_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Construct from view
    fs_tensor_type fs_tensor_view{ fs_tensor.span() };
    // Get a const reference to constructed tensor
    const fs_tensor_type& const_fs_tensor( fs_tensor_view );
    // Access elements from const fs tensor
    auto val1 = std::math::detail::access( const_fs_tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( const_fs_tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( const_fs_tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( const_fs_tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( const_fs_tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( const_fs_tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( const_fs_tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( const_fs_tensor, 1, 1, 1 );
    // Check the fs tensor tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }
  
  TEST( FS_TENSOR, TEMPLATE_COPY_CONSTRUCTOR )
  {
    using float_left_tensor_type  = std::math::fs_tensor<float,std::experimental::layout_right,std::experimental::default_accessor<float>,2,2,2>;
    using double_right_tensor_type = std::math::fs_tensor<double,std::experimental::layout_left,std::experimental::default_accessor<double>,2,2,2>;
    // Default construct
    float_left_tensor_type fs_tensor;
    // Populate via mutable index access
    std::math::detail::access( fs_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( fs_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( fs_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( fs_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( fs_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( fs_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( fs_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Construct from float tensor
    double_right_tensor_type fs_tensor_copy{ fs_tensor };
    // Access elements from const fs tensor tensor
    auto val1 = std::math::detail::access( fs_tensor_copy, 0, 0, 0 );
    auto val2 = std::math::detail::access( fs_tensor_copy, 0, 0, 1 );
    auto val3 = std::math::detail::access( fs_tensor_copy, 0, 1, 0 );
    auto val4 = std::math::detail::access( fs_tensor_copy, 0, 1, 1 );
    auto val5 = std::math::detail::access( fs_tensor_copy, 1, 0, 0 );
    auto val6 = std::math::detail::access( fs_tensor_copy, 1, 0, 1 );
    auto val7 = std::math::detail::access( fs_tensor_copy, 1, 1, 0 );
    auto val8 = std::math::detail::access( fs_tensor_copy, 1, 1, 1 );
    // Check the fs tensor tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }
  
  TEST( FS_TENSOR, CONSTRUCT_FROM_LAMBDA_EXPRESSION )
  {
    using fs_tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,2,2,2>;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    std::math::detail::access( fs_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( fs_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( fs_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( fs_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( fs_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( fs_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( fs_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Get underling view
    auto view = fs_tensor.span();
    // Create a lambda expression from view
    auto lambda = [&view]< class ... SizeType >( SizeType ... indices ) { return std::math::detail::access( view, indices ... ); };
    // Construct from lambda
    fs_tensor_type fs_tensor_copy( lambda );
    // Access elements from const fs tensor
    auto val1 = std::math::detail::access( fs_tensor_copy, 0, 0, 0 );
    auto val2 = std::math::detail::access( fs_tensor_copy, 0, 0, 1 );
    auto val3 = std::math::detail::access( fs_tensor_copy, 0, 1, 0 );
    auto val4 = std::math::detail::access( fs_tensor_copy, 0, 1, 1 );
    auto val5 = std::math::detail::access( fs_tensor_copy, 1, 0, 0 );
    auto val6 = std::math::detail::access( fs_tensor_copy, 1, 0, 1 );
    auto val7 = std::math::detail::access( fs_tensor_copy, 1, 1, 0 );
    auto val8 = std::math::detail::access( fs_tensor_copy, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }
  
  TEST( FS_TENSOR, ASSIGNMENT_OPERATOR )
  {
    using fs_tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,2,2,2>;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    std::math::detail::access( fs_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( fs_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( fs_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( fs_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( fs_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( fs_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( fs_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Construct from lambda
    fs_tensor_type fs_tensor_copy;
    fs_tensor_copy = fs_tensor;
    // Access elements from const fs tensor
    auto val1 = std::math::detail::access( fs_tensor_copy, 0, 0, 0 );
    auto val2 = std::math::detail::access( fs_tensor_copy, 0, 0, 1 );
    auto val3 = std::math::detail::access( fs_tensor_copy, 0, 1, 0 );
    auto val4 = std::math::detail::access( fs_tensor_copy, 0, 1, 1 );
    auto val5 = std::math::detail::access( fs_tensor_copy, 1, 0, 0 );
    auto val6 = std::math::detail::access( fs_tensor_copy, 1, 0, 1 );
    auto val7 = std::math::detail::access( fs_tensor_copy, 1, 1, 0 );
    auto val8 = std::math::detail::access( fs_tensor_copy, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }
  
  TEST( FS_TENSOR, TEMPLATE_ASSIGNMENT_OPERATOR )
  {
    using float_left_tensor_type   = std::math::fs_tensor<float,std::experimental::layout_right,std::experimental::default_accessor<float>,2,2,2>;
    using double_right_tensor_type = std::math::fs_tensor<double,std::experimental::layout_left,std::experimental::default_accessor<double>,2,2,2>;
    // Default construct
    float_left_tensor_type fs_tensor;
    // Populate via mutable index access
    std::math::detail::access( fs_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( fs_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( fs_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( fs_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( fs_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( fs_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( fs_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Default construct and then assign
    double_right_tensor_type fs_tensor_copy;
    fs_tensor_copy = fs_tensor;
    // Access elements from const fs tensor
    auto val1 = std::math::detail::access( fs_tensor_copy, 0, 0, 0 );
    auto val2 = std::math::detail::access( fs_tensor_copy, 0, 0, 1 );
    auto val3 = std::math::detail::access( fs_tensor_copy, 0, 1, 0 );
    auto val4 = std::math::detail::access( fs_tensor_copy, 0, 1, 1 );
    auto val5 = std::math::detail::access( fs_tensor_copy, 1, 0, 0 );
    auto val6 = std::math::detail::access( fs_tensor_copy, 1, 0, 1 );
    auto val7 = std::math::detail::access( fs_tensor_copy, 1, 1, 0 );
    auto val8 = std::math::detail::access( fs_tensor_copy, 1, 1, 1 );
    // Check the fs tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( FS_TENSOR, ASSIGN_FROM_VIEW )
  {
    using fs_tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,2,2,2>;
    // Default construct
    fs_tensor_type fs_tensor;
    // Populate via mutable index access
    std::math::detail::access( fs_tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( fs_tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( fs_tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( fs_tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( fs_tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( fs_tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( fs_tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( fs_tensor, 1, 1, 1 ) = 8.0;
    // Default construct and assign from view
    fs_tensor_type fs_tensor_view;
    fs_tensor_view = fs_tensor.span();
    // Get a const reference to constructed tensor
    const fs_tensor_type& const_fs_tensor( fs_tensor_view );
    // Access elements from const fs tensor tensor
    auto val1 = std::math::detail::access( const_fs_tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( const_fs_tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( const_fs_tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( const_fs_tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( const_fs_tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( const_fs_tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( const_fs_tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( const_fs_tensor, 1, 1, 1 );
    // Check the fs tensor tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 7.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( FS_TENSOR, SIZE_AND_CAPACITY )
  {
    using fs_tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,2,5,1,7>;
    // Default construct
    fs_tensor_type fs_tensor;
    EXPECT_TRUE( ( fs_tensor.size().extent(0) == 2 ) );
    EXPECT_TRUE( ( fs_tensor.size().extent(1) == 5 ) );
    EXPECT_TRUE( ( fs_tensor.size().extent(2) == 1 ) );
    EXPECT_TRUE( ( fs_tensor.size().extent(3) == 7 ) );
    EXPECT_TRUE( ( fs_tensor.capacity().extent(0) == 2 ) );
    EXPECT_TRUE( ( fs_tensor.capacity().extent(1) == 5 ) );
    EXPECT_TRUE( ( fs_tensor.capacity().extent(2) == 1 ) );
    EXPECT_TRUE( ( fs_tensor.capacity().extent(3) == 7 ) );
  }

  TEST( FS_TENSOR, CONST_SUBVECTOR )
  {
    // Construct
    std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5> fs_tensor{ };
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const auto& const_fs_tensor( fs_tensor );
    auto subvector = const_fs_tensor.subvector( 0, std::experimental::full_extent, 1 );
    
    EXPECT_EQ( ( std::math::detail::access( subvector, 0 ) ), ( std::math::detail::access( fs_tensor, 0, 0, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subvector, 1 ) ), ( std::math::detail::access( fs_tensor, 0, 1, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subvector, 2 ) ), ( std::math::detail::access( fs_tensor, 0, 2, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subvector, 3 ) ), ( std::math::detail::access( fs_tensor, 0, 3, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subvector, 4 ) ), ( std::math::detail::access( fs_tensor, 0, 4, 1 ) ) );
  }

  TEST( FS_TENSOR, CONST_SUBMATRIX )
  {
    // Construct
    std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5> fs_tensor{ };
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const auto& const_fs_tensor( fs_tensor );
    auto submatrix = const_fs_tensor.submatrix( 0, std::experimental::full_extent, std::tuple(0,1) );
    
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 0 ) ), ( std::math::detail::access( fs_tensor, 0, 0, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 0 ) ), ( std::math::detail::access( fs_tensor, 0, 1, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 0 ) ), ( std::math::detail::access( fs_tensor, 0, 2, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 3, 0 ) ), ( std::math::detail::access( fs_tensor, 0, 3, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 4, 0 ) ), ( std::math::detail::access( fs_tensor, 0, 4, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 1 ) ), ( std::math::detail::access( fs_tensor, 0, 0, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 1 ) ), ( std::math::detail::access( fs_tensor, 0, 1, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 1 ) ), ( std::math::detail::access( fs_tensor, 0, 2, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 3, 1 ) ), ( std::math::detail::access( fs_tensor ,0, 3, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 4, 1 ) ), ( std::math::detail::access( fs_tensor, 0, 4, 1 ) ) );
  }

  TEST( FS_TENSOR, CONST_SUBTENSOR )
  {
    using fs_tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    auto subtensor = const_fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,3) );
    
    EXPECT_EQ( ( std::math::detail::access( subtensor, 0, 0, 0 ) ), ( std::math::detail::access( fs_tensor, 2,2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 1, 0, 0 ) ), ( std::math::detail::access( fs_tensor, 3,2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 2, 0, 0 ) ), ( std::math::detail::access( fs_tensor, 4,2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 0, 1, 0 ) ), ( std::math::detail::access( fs_tensor, 2,3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 1, 1, 0 ) ), ( std::math::detail::access( fs_tensor, 3,3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 2, 1, 0 ) ), ( std::math::detail::access( fs_tensor, 4,3, 2 ) ) );
  }

  TEST( FS_TENSOR, SUBVECTOR )
  {
    // Construct
    std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5> fs_tensor{ };
    // Set values in tensor
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Get subvector
    auto subvector = fs_tensor.subvector( 1, std::experimental::full_extent, 0 );
    // Modify view
    for ( auto i : { 1, 2, 3 } )
    {
      std::math::detail::access( subvector, i ) = val;
      val = 2 * val;
    }
    // Assert original tensor has been modified as well
    EXPECT_EQ( ( std::math::detail::access( subvector, 1 ) ), ( std::math::detail::access( fs_tensor, 1, 1, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subvector, 2 ) ), ( std::math::detail::access( fs_tensor, 1, 2, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subvector, 3 ) ), ( std::math::detail::access( fs_tensor ,1, 3, 0 ) ) );
  }

  TEST( FS_TENSOR, SUBMATRIX )
  {
    // Construct
    std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5> fs_tensor{ };
    // Set values in tensor
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Get submatrix
    auto submatrix = fs_tensor.submatrix( 1, std::experimental::full_extent, std::tuple( 1, 4 ) );
    // Modify view
    for ( auto i : { 1, 2, 3 } )
    {
      for ( auto j : { 0, 1 } )
      {
        std::math::detail::access( submatrix, i, j ) = val;
        val = 2 * val;
      }
    }
    // Assert original tensor has been modified as well
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 0 ) ), ( std::math::detail::access( fs_tensor, 1, 1, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 0 ) ), ( std::math::detail::access( fs_tensor, 1, 2, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 3, 0 ) ), ( std::math::detail::access( fs_tensor, 1, 3, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 1 ) ), ( std::math::detail::access( fs_tensor, 1, 1, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 1 ) ), ( std::math::detail::access( fs_tensor, 1, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 3, 1 ) ), ( std::math::detail::access( fs_tensor, 1, 3, 2 ) ) );
  }

  TEST( FS_TENSOR, SUBTENSOR )
  {
    using fs_tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,5,5,5>;
    // Default construct
    fs_tensor_type fs_tensor;
    // Set values in tensor tensor
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          std::math::detail::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Get subtensor
    auto subtensor = fs_tensor.subtensor( std::tuple(2,5), std::tuple(2,4), std::tuple(2,3) );
    // Modify view
    for ( auto i : { 0, 1, 2 } )
    {
      for ( auto j : { 0, 1 } )
      {
        for ( auto k : { 0 } )
        {
          std::math::detail::access( subtensor, i,j, k ) = val;
          val = 2 * val;
        }
      }
    }
    // Assert original tensor has been modified as well
    EXPECT_EQ( ( std::math::detail::access( subtensor, 0, 0, 0 ) ), ( std::math::detail::access( fs_tensor, 2, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 1, 0, 0 ) ), ( std::math::detail::access( fs_tensor, 3, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 2, 0, 0 ) ), ( std::math::detail::access( fs_tensor, 4, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 0, 1, 0 ) ), ( std::math::detail::access( fs_tensor, 2, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 1, 1, 0 ) ), ( std::math::detail::access( fs_tensor, 3, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( subtensor, 2, 1, 0 ) ), ( std::math::detail::access( fs_tensor, 4, 3, 2 ) ) );
  }

  TEST( FS_TENSOR, NEGATION )
  {
    using tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Negate the tensor
    tensor_type negate_tensor { -tensor };
    // Access elements from const tensor
    auto val1 = std::math::detail::access( negate_tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( negate_tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( negate_tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( negate_tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( negate_tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( negate_tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( negate_tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( negate_tensor, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, -1.0 );
    EXPECT_EQ( val2, -2.0 );
    EXPECT_EQ( val3, -3.0 );
    EXPECT_EQ( val4, -4.0 );
    EXPECT_EQ( val5, -5.0 );
    EXPECT_EQ( val6, -6.0 );
    EXPECT_EQ( val7, -7.0 );
    EXPECT_EQ( val8, -8.0 );
  }

  TEST( FS_TENSOR, ADD )
  {
    using tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Add the two tensors together
    tensor_type tensor_sum { tensor + tensor_copy };
    // Access elements from const tensor
    auto val1 = std::math::detail::access( tensor_sum, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor_sum, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor_sum, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor_sum, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor_sum, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor_sum, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor_sum, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor_sum, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( FS_TENSOR, ADD_ASSIGN )
  {
    using tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Add the two tensors together
    static_cast<void>( tensor += tensor_copy );
    // Access elements from tensor
    auto val1 = std::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( FS_TENSOR, SUBTRACT )
  {
    using tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Subtract the two tensors
    tensor_type tensor_diff { tensor - tensor_copy };
    // Access elements from const tensor
    auto val1 = std::math::detail::access( tensor_diff, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor_diff, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor_diff, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor_diff, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor_diff, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor_diff, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor_diff, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor_diff, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
    EXPECT_EQ( val5, 0 );
    EXPECT_EQ( val6, 0 );
    EXPECT_EQ( val7, 0 );
    EXPECT_EQ( val8, 0 );
  }

  TEST( FS_TENSOR, SUBTRACT_ASSIGN )
  {
    using tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Copy construct
    tensor_type tensor_copy{ tensor };
    // Subtract the two tensors
    static_cast<void>( tensor -= tensor_copy );
    // Access elements from tensor
    auto val1 = std::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
    EXPECT_EQ( val5, 0 );
    EXPECT_EQ( val6, 0 );
    EXPECT_EQ( val7, 0 );
    EXPECT_EQ( val8, 0 );
  }

  TEST( FS_TENSOR, SCALAR_PREMULTIPLY )
  {
    using tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Pre multiply
    tensor_type tensor_prod { 2 * tensor };
    // Access elements from const tensor
    auto val1 = std::math::detail::access( tensor_prod, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor_prod, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor_prod, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor_prod, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor_prod, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor_prod, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor_prod, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor_prod, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( FS_TENSOR, SCALAR_POSTMULTIPLY )
  {
    using tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Post multiply
    tensor_type tensor_prod { tensor * 2 };
    // Access elements from const tensor
    auto val1 = std::math::detail::access( tensor_prod, 0,0, 0 );
    auto val2 = std::math::detail::access( tensor_prod, 0,0, 1 );
    auto val3 = std::math::detail::access( tensor_prod, 0,1, 0 );
    auto val4 = std::math::detail::access( tensor_prod, 0,1, 1 );
    auto val5 = std::math::detail::access( tensor_prod, 1,0, 0 );
    auto val6 = std::math::detail::access( tensor_prod, 1,0, 1 );
    auto val7 = std::math::detail::access( tensor_prod, 1,1, 0 );
    auto val8 = std::math::detail::access( tensor_prod, 1,1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( FS_TENSOR, SCALAR_MULTIPLY_ASSIGN )
  {
    using tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Post multiply
    static_cast<void>( tensor *= 2 );
    // Access elements from const tensor
    auto val1 = std::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
    EXPECT_EQ( val5, 10.0 );
    EXPECT_EQ( val6, 12.0 );
    EXPECT_EQ( val7, 14.0 );
    EXPECT_EQ( val8, 16.0 );
  }

  TEST( FS_TENSOR, SCALAR_DIVIDE )
  {
    using tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Divide
    tensor_type tensor_divide { tensor / 2 };
    // Access elements from const tensor
    auto val1 = std::math::detail::access( tensor_divide, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor_divide, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor_divide, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor_divide, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor_divide, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor_divide, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor_divide, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor_divide, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
    EXPECT_EQ( val5, 2.5 );
    EXPECT_EQ( val6, 3.0 );
    EXPECT_EQ( val7, 3.5 );
    EXPECT_EQ( val8, 4.0 );
  }

  TEST( FS_TENSOR, SCALAR_DIVIDE_ASSIGN )
  {
    using tensor_type = std::math::fs_tensor<double,std::experimental::layout_right,std::experimental::default_accessor<double>,3,3,3>;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    std::math::detail::access( tensor, 0, 0, 0 ) = 1.0;
    std::math::detail::access( tensor, 0, 0, 1 ) = 2.0;
    std::math::detail::access( tensor, 0, 1, 0 ) = 3.0;
    std::math::detail::access( tensor, 0, 1, 1 ) = 4.0;
    std::math::detail::access( tensor, 1, 0, 0 ) = 5.0;
    std::math::detail::access( tensor, 1, 0, 1 ) = 6.0;
    std::math::detail::access( tensor, 1, 1, 0 ) = 7.0;
    std::math::detail::access( tensor, 1, 1, 1 ) = 8.0;
    // Divide
    static_cast<void>( tensor /= 2 );
    // Access elements from const tensor
    auto val1 = std::math::detail::access( tensor, 0, 0, 0 );
    auto val2 = std::math::detail::access( tensor, 0, 0, 1 );
    auto val3 = std::math::detail::access( tensor, 0, 1, 0 );
    auto val4 = std::math::detail::access( tensor, 0, 1, 1 );
    auto val5 = std::math::detail::access( tensor, 1, 0, 0 );
    auto val6 = std::math::detail::access( tensor, 1, 0, 1 );
    auto val7 = std::math::detail::access( tensor, 1, 1, 0 );
    auto val8 = std::math::detail::access( tensor, 1, 1, 1 );
    // Check the tensor was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
    EXPECT_EQ( val5, 2.5 );
    EXPECT_EQ( val6, 3.0 );
    EXPECT_EQ( val7, 3.5 );
    EXPECT_EQ( val8, 4.0 );
  }

}