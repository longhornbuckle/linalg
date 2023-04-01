#include <gtest/gtest.h>
#include <experimental/linear_algebra.hpp>

namespace
{
  TEST( DR_VECTOR, DEFAULT_CONSTRUCTOR_AND_DESTRUCTOR )
  {
    // Default construction
    std::math::dr_vector<double> dyn_vector;
    // Destructor will be called when unit test ends and the dr vector exits scope
  }

  TEST( DR_VECTOR, MUTABLE_AND_CONST_INDEX_ACCESS )
  {
    // Construct
    std::math::dr_vector<double> dyn_vector{ std::experimental::extents<size_t,4>(), std::experimental::extents<size_t,10>() };
    // Populate via mutable index access
    dyn_vector[0] = 1.0;
    dyn_vector[1] = 2.0;
    dyn_vector[2] = 3.0;
    dyn_vector[3] = 4.0;
    // Get a const reference
    const std::math::dr_vector<double>& const_dyn_vector( dyn_vector );
    // Access elements from const dyn vector
    auto val1 = const_dyn_vector[0];
    auto val2 = const_dyn_vector[1];
    auto val3 = const_dyn_vector[2];
    auto val4 = const_dyn_vector[3];
    // Check the dyn vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_VECTOR, COPY_CONSTRUCTOR )
  {
    // Construct
    std::math::dr_vector<double> dyn_vector{ std::experimental::extents<size_t,4>(), std::experimental::extents<size_t,6>() };
    // Populate via mutable index access
    dyn_vector[0] = 1.0;
    dyn_vector[1] = 2.0;
    dyn_vector[2] = 3.0;
    dyn_vector[3] = 4.0;
    // Copy construct
    std::math::dr_vector<double> dyn_vector_copy{ dyn_vector };
    // Get a const reference to copy
    const std::math::dr_vector<double>& const_dyn_vector( dyn_vector_copy );
    // Access elements from const dyn vector
    auto val1 = const_dyn_vector[0];
    auto val2 = const_dyn_vector[1];
    auto val3 = const_dyn_vector[2];
    auto val4 = const_dyn_vector[3];
    // Check the dyn vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_VECTOR, MOVE_CONSTRUCTOR )
  {
    // Construct
    std::math::dr_vector<double> dyn_vector{ std::experimental::extents<size_t,4>(), std::experimental::extents<size_t,6>() };
    // Populate via mutable index access
    dyn_vector[0] = 1.0;
    dyn_vector[1] = 2.0;
    dyn_vector[2] = 3.0;
    dyn_vector[3] = 4.0;
    // Move construct
    std::math::dr_vector<double> dyn_vector_move{ std::move( dyn_vector ) };
    // Get a const reference to moved vector
    const std::math::dr_vector<double>& const_dyn_vector( dyn_vector_move );
    // Access elements from const dyn vector
    auto val1 = const_dyn_vector[0];
    auto val2 = const_dyn_vector[1];
    auto val3 = const_dyn_vector[2];
    auto val4 = const_dyn_vector[3];
    // Check the dyn vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_VECTOR, CONSTRUCT_FROM_VIEW )
  {
    // Construct
    std::math::dr_vector<double> dyn_vector{ std::experimental::extents<size_t,4>(), std::experimental::extents<size_t,6>() };
    // Populate via mutable index access
    dyn_vector[0] = 1.0;
    dyn_vector[1] = 2.0;
    dyn_vector[2] = 3.0;
    dyn_vector[3] = 4.0;
    // Construct from view
    std::math::dr_vector<double> dyn_vector_view{ dyn_vector.span() };
    // Get a const reference to constructed vector
    const std::math::dr_vector<double>& const_dyn_vector( dyn_vector_view );
    // Access elements from const dyn vector
    auto val1 = const_dyn_vector[0];
    auto val2 = const_dyn_vector[1];
    auto val3 = const_dyn_vector[2];
    auto val4 = const_dyn_vector[3];
    // Check the dyn vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_VECTOR, TEMPLATE_COPY_CONSTRUCTOR )
  {
    using float_left_vector_type   = std::math::fs_vector<float,4,std::experimental::layout_left,std::experimental::default_accessor<float> >;
    using double_right_vector_type = std::math::dr_vector<double>;
    // Default construct
    float_left_vector_type fs_vector;
    // Populate via mutable index access
    fs_vector[0] = 1.0;
    fs_vector[1] = 2.0;
    fs_vector[2] = 3.0;
    fs_vector[3] = 4.0;
    // Construct from float vector
    double_right_vector_type dyn_vector_copy{ fs_vector };
    // Access elements from dyn vector
    auto val1 = dyn_vector_copy[0];
    auto val2 = dyn_vector_copy[1];
    auto val3 = dyn_vector_copy[2];
    auto val4 = dyn_vector_copy[3];
    // Check the dyn vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_VECTOR, CONSTRUCT_FROM_LAMBDA_EXPRESSION )
  {
    using fs_vector_type = std::math::fs_vector<double,4,std::experimental::layout_right,std::experimental::default_accessor<double> >;
    // Default construct
    fs_vector_type fs_vector;
    // Populate via mutable index access
    fs_vector[0] = 1.0;
    fs_vector[1] = 2.0;
    fs_vector[2] = 3.0;
    fs_vector[3] = 4.0;
    // Get underling view
    auto view = fs_vector.span();
    // Create a lambda expression from view
    auto lambda = [&view]< class ... SizeType >( SizeType ... indices ) { return view[indices...]; };
    // Construct from lambda
    std::math::dr_vector<double> dyn_vector_copy( std::experimental::extents<size_t,4>(), lambda );
    // Access elements from const dyn vector
    auto val1 = dyn_vector_copy[0];
    auto val2 = dyn_vector_copy[1];
    auto val3 = dyn_vector_copy[2];
    auto val4 = dyn_vector_copy[3];
    // Check the dyn vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_VECTOR, ASSIGNMENT_OPERATOR )
  {
    // Construct
    std::math::dr_vector<double> dyn_vector{ std::experimental::extents<size_t,4>(), std::experimental::extents<size_t,5>() };
    // Populate via mutable index access
    dyn_vector[0] = 1.0;
    dyn_vector[1] = 2.0;
    dyn_vector[2] = 3.0;
    dyn_vector[3] = 4.0;
    // Default construct and assign
    std::math::dr_vector<double> dyn_vector_copy;
    dyn_vector_copy = dyn_vector;
    // Access elements from dyn vector
    auto val1 = dyn_vector_copy[0];
    auto val2 = dyn_vector_copy[1];
    auto val3 = dyn_vector_copy[2];
    auto val4 = dyn_vector_copy[3];
    // Check the dyn vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_VECTOR, TEMPLATE_ASSIGNMENT_OPERATOR )
  {
    using float_left_vector_type = std::math::fs_vector<float,4,std::experimental::layout_left,std::experimental::default_accessor<float> >;
    // Default construct
    float_left_vector_type fs_vector { };
    // Populate via mutable index access
    fs_vector[0] = 1.0;
    fs_vector[1] = 2.0;
    fs_vector[2] = 3.0;
    fs_vector[3] = 4.0;
    // Default construct and then assign
    std::math::dr_vector<double> dyn_vector_copy;
    dyn_vector_copy = fs_vector;
    // Access elements from const dyn vector
    auto val1 = dyn_vector_copy[0];
    auto val2 = dyn_vector_copy[1];
    auto val3 = dyn_vector_copy[2];
    auto val4 = dyn_vector_copy[3];
    // Check the dyn vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_VECTOR, ASSIGN_FROM_VIEW )
  {
    using fs_vector_type = std::math::fs_vector<double,4,std::experimental::layout_left,std::experimental::default_accessor<double> >;
    // Default construct
    fs_vector_type fs_vector;
    // Populate via mutable index access
    fs_vector[0] = 1.0;
    fs_vector[1] = 2.0;
    fs_vector[2] = 3.0;
    fs_vector[3] = 4.0;
    // Construct and assign from view
    std::math::dr_vector<double> dyn_vector_view;
    dyn_vector_view = fs_vector.span();
    // Access elements from const dyn vector
    auto val1 = dyn_vector_view[0];
    auto val2 = dyn_vector_view[1];
    auto val3 = dyn_vector_view[2];
    auto val4 = dyn_vector_view[3];
    // Check the dyn vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_VECTOR, SIZE_AND_CAPACITY )
  {
    // Construct
    std::math::dr_vector<double> dyn_vector{ std::experimental::extents<size_t,2>(), std::experimental::extents<size_t,3>() };
    EXPECT_TRUE( ( dyn_vector.size() == 2 ) );
    EXPECT_TRUE( ( dyn_vector.capacity() == 3 ) );
  }

  TEST( DR_VECTOR, RESIZE )
  {
    // Construct
    std::math::dr_vector<double> dyn_vector{ std::experimental::extents<size_t,4>(), std::experimental::extents<size_t,9>() };
    // Populate via mutable index access
    dyn_vector[0] = 1.0;
    dyn_vector[1] = 2.0;
    dyn_vector[2] = 3.0;
    dyn_vector[3] = 4.0;
    // Resize
    dyn_vector.resize( 9 );
    dyn_vector[4] = 5.0;
    dyn_vector[5] = 6.0;
    dyn_vector[6] = 7.0;
    dyn_vector[7] = 8.0;
    dyn_vector[8] = 9.0;
    // Get values
    auto val1 = dyn_vector[0];
    auto val2 = dyn_vector[1];
    auto val3 = dyn_vector[2];
    auto val4 = dyn_vector[3];
    auto val5 = dyn_vector[4];
    auto val6 = dyn_vector[5];
    auto val7 = dyn_vector[6];
    auto val8 = dyn_vector[7];
    auto val9 = dyn_vector[8];
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
  }

  TEST( DR_VECTOR, RESERVE )
  {
    // Construct
    std::math::dr_vector<double> dyn_vector{ std::experimental::extents<size_t,4>(), std::experimental::extents<size_t,4>() };
    // Populate via mutable index access
    dyn_vector[0] = 1.0;
    dyn_vector[1] = 2.0;
    dyn_vector[2] = 3.0;
    dyn_vector[3] = 4.0;
    // Resize
    dyn_vector.reserve( 16 );
    // Get values
    auto val1  = dyn_vector[0];
    auto val2  = dyn_vector[1];
    auto val3  = dyn_vector[2];
    auto val4  = dyn_vector[3];
    // Check the values are correct
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_VECTOR, CONST_SUBVECTOR )
  {
    // Construct
    std::math::dr_vector<double> dyn_vector{ std::experimental::extents<size_t,5>(), std::experimental::extents<size_t,10>() };
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      dyn_vector[i] = val;
      val = 2 * val;
    }
    const std::math::dr_vector<double>& const_dyn_vector( dyn_vector );
    auto subvector = const_dyn_vector.subvector( 2, 5 );
    // Assert subvector maps to original vector
    EXPECT_EQ( ( subvector[0] ), ( dyn_vector[2] ) );
    EXPECT_EQ( ( subvector[1] ), ( dyn_vector[3] ) );
    EXPECT_EQ( ( subvector[2] ), ( dyn_vector[4] ) );
  }

  TEST( DR_VECTOR, SUBVECTOR )
  {
    // Construct
    std::math::dr_vector<double> dyn_vector{ std::experimental::extents<size_t,5>(), std::experimental::extents<size_t,10>() };
    // Set values in tensor engine
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      dyn_vector[i] = val;
      val = 2 * val;
    }
    // Get subvector
    auto subvector = dyn_vector.subvector( 2, 5 );
    // Modify view
    for ( auto i : { 0, 1, 2 } )
    {
      subvector[i] = val;
      val = 2 * val;
    }
    // Assert original vector has been modified as well
    EXPECT_EQ( ( subvector[0] ), ( dyn_vector[2] ) );
    EXPECT_EQ( ( subvector[1] ), ( dyn_vector[3] ) );
    EXPECT_EQ( ( subvector[2] ), ( dyn_vector[4] ) );
  }

  TEST( DR_VECTOR, NEGATION )
  {
    using vector_type = std::math::dr_vector<double>;
    // Construct
    vector_type vector{ 4 };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Negate the vector
    vector_type negate_vector { -vector };
    // Access elements from const vector
    auto val1 = negate_vector[0];
    auto val2 = negate_vector[1];
    auto val3 = negate_vector[2];
    auto val4 = negate_vector[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, -1.0 );
    EXPECT_EQ( val2, -2.0 );
    EXPECT_EQ( val3, -3.0 );
    EXPECT_EQ( val4, -4.0 );
  }

  TEST( DR_VECTOR, TRANSPOSE )
  {
    using vector_type = std::math::fs_vector<double,3>;
    // Construct
    vector_type vector { };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    // Transpose the vector
    auto transpose_vector { trans(vector) };
    // Access elements from transpose vector
    auto val1 = transpose_vector[0];
    auto val2 = transpose_vector[1];
    auto val3 = transpose_vector[2];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
  }

  TEST( DR_VECTOR, CONJUGATE )
  {
    using vector_type = std::math::dr_vector<std::complex<double> >;
    // Construct
    vector_type vector { 3 };
    // Populate via mutable index access
    vector[0] = std::complex<double>( 1.0, 1.0 );
    vector[1] = std::complex<double>( 2.0, 2.0 );
    vector[2] = std::complex<double>( 3.0, 3.0 );
    // Conjugate the vector
    auto conjugate_vector { conj(vector) };
    // Access elements from conjugate vector
    auto val1 = conjugate_vector[0];
    auto val2 = conjugate_vector[1];
    auto val3 = conjugate_vector[2];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, std::complex<double>( 1.0, -1.0 ) );
    EXPECT_EQ( val2, std::complex<double>( 2.0, -2.0 ) );
    EXPECT_EQ( val3, std::complex<double>( 3.0, -3.0 ) );
  }

  TEST( DR_VECTOR, ADD )
  {
    using vector_type = std::math::dr_vector<double>;
    // Construct
    vector_type vector{ 4 };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Copy construct
    vector_type vector_copy{ vector };
    // Add the two vectors together
    vector_type vector_sum { vector + vector_copy };
    // Access elements from const vector
    auto val1 = vector_sum[0];
    auto val2 = vector_sum[1];
    auto val3 = vector_sum[2];
    auto val4 = vector_sum[3];
    // Check the vector copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( DR_VECTOR, ADD_ASSIGN )
  {
    using vector_type = std::math::dr_vector<double>;
    // Construct
    vector_type vector{ 4 };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Copy construct
    vector_type vector_copy{ vector };
    // Add the two vectors together
    static_cast<void>( vector += vector_copy );
    // Access elements from vector
    auto val1 = vector[0];
    auto val2 = vector[1];
    auto val3 = vector[2];
    auto val4 = vector[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( DR_VECTOR, SUBTRACT )
  {
    using vector_type = std::math::dr_vector<double>;
    // Construct
    vector_type vector{ 4 };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Copy construct
    vector_type vector_copy{ vector };
    // Subtract the two vectors
    vector_type vector_diff { vector - vector_copy };
    // Access elements from const vector
    auto val1 = vector_diff[0];
    auto val2 = vector_diff[1];
    auto val3 = vector_diff[2];
    auto val4 = vector_diff[3];
    // Check the vector copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
  }

  TEST( DR_VECTOR, SUBTRACT_ASSIGN )
  {
    using vector_type = std::math::dr_vector<double>;
    // Construct
    vector_type vector{ 4 };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Copy construct
    vector_type vector_copy{ vector };
    // Subtract the two vectors
    static_cast<void>( vector -= vector_copy );
    // Access elements from vector
    auto val1 = vector[0];
    auto val2 = vector[1];
    auto val3 = vector[2];
    auto val4 = vector[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
  }

  TEST( DR_VECTOR, SCALAR_PREMULTIPLY )
  {
    using vector_type = std::math::dr_vector<double>;
    // Construct
    vector_type vector{ 4 };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Pre multiply
    vector_type vector_prod { 2 * vector };
    // Access elements from const vector
    auto val1 = vector_prod[0];
    auto val2 = vector_prod[1];
    auto val3 = vector_prod[2];
    auto val4 = vector_prod[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( DR_VECTOR, SCALAR_POSTMULTIPLY )
  {
    using vector_type = std::math::dr_vector<double>;
    // Construct
    vector_type vector{ 4 };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Post multiply
    vector_type vector_prod { vector * 2 };
    // Access elements from const vector
    auto val1 = vector_prod[0];
    auto val2 = vector_prod[1];
    auto val3 = vector_prod[2];
    auto val4 = vector_prod[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( DR_VECTOR, SCALAR_MULTIPLY_ASSIGN )
  {
    using vector_type = std::math::dr_vector<double>;
    // Construct
    vector_type vector{ 4 };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Post multiply
    static_cast<void>( vector *= 2 );
    // Access elements from vector
    auto val1 = vector[0];
    auto val2 = vector[1];
    auto val3 = vector[2];
    auto val4 = vector[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( DR_VECTOR, SCALAR_DIVIDE )
  {
    using vector_type = std::math::dr_vector<double>;
    // Construct
    vector_type vector{ 4 };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Divide
    vector_type vector_divide { vector / 2 };
    // Access elements from const vector
    auto val1 = vector_divide[0];
    auto val2 = vector_divide[1];
    auto val3 = vector_divide[2];
    auto val4 = vector_divide[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
  }

  TEST( DR_VECTOR, SCALAR_DIVIDE_ASSIGN )
  {
    using vector_type = std::math::dr_vector<double>;
    // Construct
    vector_type vector{ 4 };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Divide
    static_cast<void>( vector /= 2 );
    // Access elements from vector
    auto val1 = vector[0];
    auto val2 = vector[1];
    auto val3 = vector[2];
    auto val4 = vector[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
  }

  TEST( DR_VECTOR, INNER_PRODUCT )
  {
    using vector_type = std::math::dr_vector<double>;
    // Construct
    vector_type vector{ 6 };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    vector[4] = 5.0;
    vector[5] = 6.0;
    // Compute inner product
    auto inner_product_val = inner_prod( vector, vector );
    // Check the inner product was properly computed
    EXPECT_EQ( inner_product_val, 91 );
  }

  TEST( DR_VECTOR, OUTER_PRODUCT )
  {
    using vector_type = std::math::dr_vector<double>;
    // Construct
    vector_type vector{ 3 };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    // Compute outer product
    auto outer_product = outer_prod( vector, vector );
    // Access elements from outer product
    auto val1 = outer_product[0,0];
    auto val2 = outer_product[0,1];
    auto val3 = outer_product[0,2];
    auto val4 = outer_product[1,0];
    auto val5 = outer_product[1,1];
    auto val6 = outer_product[1,2];
    auto val7 = outer_product[2,0];
    auto val8 = outer_product[2,1];
    auto val9 = outer_product[2,2];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 2.0 );
    EXPECT_EQ( val5, 4.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 3.0 );
    EXPECT_EQ( val8, 6.0 );
    EXPECT_EQ( val9, 9.0 );
  }

  TEST( DR_VECTOR, MATRIX_POSTMULTIPLY )
  {
    using matrix_type = std::math::dr_matrix<double>;
    using vector_type = std::math::dr_vector<double>;
    // Construct matrix
    matrix_type matrix{ std::experimental::extents<size_t,2,3>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    matrix[0,0] = 1.0;
    matrix[0,1] = 2.0;
    matrix[0,2] = 3.0;
    matrix[1,0] = 4.0;
    matrix[1,1] = 5.0;
    matrix[1,2] = 6.0;
    // Construct vector
    vector_type vector{ std::experimental::extents<size_t,2>(), std::experimental::extents<size_t,4>() };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    // Multiply vector with matrix
    auto vector_multiply { vector * matrix };
    // Access elements from vector
    auto val1 = vector_multiply[0];
    auto val2 = vector_multiply[1];
    auto val3 = vector_multiply[2];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 9.0 );
    EXPECT_EQ( val2, 12.0 );
    EXPECT_EQ( val3, 15.0 );
  }

  TEST( DR_VECTOR, MATRIX_PREMULTIPLY )
  {
    using matrix_type = std::math::dr_matrix<double>;
    using vector_type = std::math::dr_vector<double>;
    // Construct matrix
    matrix_type matrix{ std::experimental::extents<size_t,2,3>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    matrix[0,0] = 1.0;
    matrix[0,1] = 2.0;
    matrix[0,2] = 3.0;
    matrix[1,0] = 4.0;
    matrix[1,1] = 5.0;
    matrix[1,2] = 6.0;
    // Construct vector
    vector_type vector{ std::experimental::extents<size_t,3>(), std::experimental::extents<size_t,4>() };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    // Multiply matrix with vector
    auto vector_multiply { matrix * vector };
    // Access elements from vector
    auto val1 = vector_multiply[0];
    auto val2 = vector_multiply[1];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 14.0 );
    EXPECT_EQ( val2, 32.0 );
  }

  TEST( DR_VECTOR, MATRIX_MULTIPLY_ASSIGN )
  {
    using matrix_type = std::math::dr_matrix<double>;
    using vector_type = std::math::dr_vector<double>;
    // Construct matrix
    matrix_type matrix{ std::experimental::extents<size_t,2,3>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    matrix[0,0] = 1.0;
    matrix[0,1] = 2.0;
    matrix[0,2] = 3.0;
    matrix[1,0] = 4.0;
    matrix[1,1] = 5.0;
    matrix[1,2] = 6.0;
    // Construct vector
    vector_type vector{ std::experimental::extents<size_t,2>(), std::experimental::extents<size_t,4>() };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    // Multiply matrix with vector
    static_cast<void>( vector *= matrix );
    // Access elements from vector
    auto val1 = vector[0];
    auto val2 = vector[1];
    auto val3 = vector[2];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 9.0 );
    EXPECT_EQ( val2, 12.0 );
    EXPECT_EQ( val3, 15.0 );
  }

  TEST( FS_VECTOR, DEFAULT_CONSTRUCTOR_AND_DESTRUCTOR )
  {
    // Default construction
    std::math::fs_vector<double,2> fs_vector;
    // Destructor will be called when unit test ends and the fs vector exits scope
  }

  TEST( FS_VECTOR, MUTABLE_AND_CONST_INDEX_ACCESS )
  {
    using fs_vector_type = std::math::fs_vector<double,4>;
    // Default construct
    fs_vector_type fs_vector;
    // Populate via mutable index access
    fs_vector[0] = 1.0;
    fs_vector[1] = 2.0;
    fs_vector[2] = 3.0;
    fs_vector[3] = 4.0;
    // Get a const reference
    const fs_vector_type& const_fs_vector( fs_vector );
    // Access elements from const fs vector
    auto val1 = const_fs_vector[0];
    auto val2 = const_fs_vector[1];
    auto val3 = const_fs_vector[2];
    auto val4 = const_fs_vector[3];
    // Check the fs vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_VECTOR, COPY_CONSTRUCTOR )
  {
    using fs_vector_type = std::math::fs_vector<double,4>;
    // Default construct
    fs_vector_type fs_vector;
    // Populate via mutable index access
    fs_vector[0] = 1.0;
    fs_vector[1] = 2.0;
    fs_vector[2] = 3.0;
    fs_vector[3] = 4.0;
    // Copy construct
    fs_vector_type fs_vector_copy{ fs_vector };
    // Get a const reference to copy
    const fs_vector_type& const_fs_vector( fs_vector_copy );
    // Access elements from const fs vector
    auto val1 = const_fs_vector[0];
    auto val2 = const_fs_vector[1];
    auto val3 = const_fs_vector[2];
    auto val4 = const_fs_vector[3];
    // Check the fs vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_VECTOR, MOVE_CONSTRUCTOR )
  {
    using fs_vector_type = std::math::fs_vector<double,4>;
    // Default construct
    fs_vector_type fs_vector;
    // Populate via mutable index access
    fs_vector[0] = 1.0;
    fs_vector[1] = 2.0;
    fs_vector[2] = 3.0;
    fs_vector[3] = 4.0;
    // Move construct
    fs_vector_type fs_vector_move{ std::move( fs_vector ) };
    // Get a const reference to moved vector
    const fs_vector_type& const_fs_vector( fs_vector_move );
    // Access elements from const fs vector
    auto val1 = const_fs_vector[0];
    auto val2 = const_fs_vector[1];
    auto val3 = const_fs_vector[2];
    auto val4 = const_fs_vector[3];
    // Check the fs vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_VECTOR, CONSTRUCT_FROM_VIEW )
  {
    using fs_vector_type = std::math::fs_vector<double,4>;
    // Default construct
    fs_vector_type fs_vector;
    // Populate via mutable index access
    fs_vector[0] = 1.0;
    fs_vector[1] = 2.0;
    fs_vector[2] = 3.0;
    fs_vector[3] = 4.0;
    // Construct from view
    fs_vector_type fs_vector_view{ fs_vector.span() };
    // Get a const reference to constructed vector
    const fs_vector_type& const_fs_vector( fs_vector_view );
    // Access elements from const fs vector
    auto val1 = const_fs_vector[0];
    auto val2 = const_fs_vector[1];
    auto val3 = const_fs_vector[2];
    auto val4 = const_fs_vector[3];
    // Check the fs vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_VECTOR, TEMPLATE_COPY_CONSTRUCTOR )
  {
    using float_left_vector_type   = std::math::fs_vector<float,4>;
    using double_right_vector_type = std::math::fs_vector<double,4>;
    // Default construct
    float_left_vector_type fs_vector;
    // Populate via mutable index access
    fs_vector[0] = 1.0;
    fs_vector[1] = 2.0;
    fs_vector[2] = 3.0;
    fs_vector[3] = 4.0;
    // Construct from float vector
    double_right_vector_type fs_vector_copy{ fs_vector };
    // Access elements from const fs vector
    auto val1 = fs_vector_copy[0];
    auto val2 = fs_vector_copy[1];
    auto val3 = fs_vector_copy[2];
    auto val4 = fs_vector_copy[3];
    // Check the fs vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_VECTOR, CONSTRUCT_FROM_LAMBDA_EXPRESSION )
  {
    using fs_vector_type = std::math::fs_vector<double,4>;
    // Default construct
    fs_vector_type fs_vector;
    // Populate via mutable index access
    fs_vector[0] = 1.0;
    fs_vector[1] = 2.0;
    fs_vector[2] = 3.0;
    fs_vector[3] = 4.0;
    // Get underling view
    auto view = fs_vector.span();
    // Create a lambda expression from view
    auto lambda = [&view]< class ... SizeType >( SizeType ... indices ) { return view[indices...]; };
    // Construct from lambda
    fs_vector_type fs_vector_copy( lambda );
    // Access elements from const fs vector
    auto val1 = fs_vector_copy[0];
    auto val2 = fs_vector_copy[1];
    auto val3 = fs_vector_copy[2];
    auto val4 = fs_vector_copy[3];
    // Check the fs vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_VECTOR, ASSIGNMENT_OPERATOR )
  {
    using fs_vector_type = std::math::fs_vector<double,4>;
    // Default construct
    fs_vector_type fs_vector;
    // Populate via mutable index access
    fs_vector[0] = 1.0;
    fs_vector[1] = 2.0;
    fs_vector[2] = 3.0;
    fs_vector[3] = 4.0;
    // Construct from lambda
    fs_vector_type fs_vector_copy;
    fs_vector_copy = fs_vector;
    // Access elements from const fs vector
    auto val1 = fs_vector_copy[0];
    auto val2 = fs_vector_copy[1];
    auto val3 = fs_vector_copy[2];
    auto val4 = fs_vector_copy[3];
    // Check the fs vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_VECTOR, TEMPLATE_ASSIGNMENT_OPERATOR )
  {
    using float_left_vector_type   = std::math::fs_vector<float,4>;
    using double_right_vector_type = std::math::fs_vector<double,4>;
    // Default construct
    float_left_vector_type fs_vector;
    // Populate via mutable index access
    fs_vector[0] = 1.0;
    fs_vector[1] = 2.0;
    fs_vector[2] = 3.0;
    fs_vector[3] = 4.0;
    // Default construct and then assign
    double_right_vector_type fs_vector_copy;
    fs_vector_copy = fs_vector;
    // Access elements from const fs vector
    auto val1 = fs_vector_copy[0];
    auto val2 = fs_vector_copy[1];
    auto val3 = fs_vector_copy[2];
    auto val4 = fs_vector_copy[3];
    // Check the fs vector vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_VECTOR, ASSIGN_FROM_VIEW )
  {
    using fs_vector_type = std::math::fs_vector<double,4>;
    // Default construct
    fs_vector_type fs_vector;
    // Populate via mutable index access
    fs_vector[0] = 1.0;
    fs_vector[1] = 2.0;
    fs_vector[2] = 3.0;
    fs_vector[3] = 4.0;
    // Default construct and assign from view
    fs_vector_type fs_vector_view;
    fs_vector_view = fs_vector.span();
    // Get a const reference to constructed vector
    const fs_vector_type& const_fs_vector( fs_vector_view );
    // Access elements from const fs vector
    auto val1 = const_fs_vector[0];
    auto val2 = const_fs_vector[1];
    auto val3 = const_fs_vector[2];
    auto val4 = const_fs_vector[3];
    // Check the fs vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_VECTOR, SIZE_AND_CAPACITY )
  {
    using fs_vector_type = std::math::fs_vector<double,5>;
    // Default construct
    fs_vector_type fs_vector;
    EXPECT_TRUE( ( fs_vector.size() == 5 ) );
    EXPECT_TRUE( ( fs_vector.capacity() == 5 ) );
  }

  TEST( FS_VECTOR, CONST_SUBVECTOR )
  {
    using fs_vector_type = std::math::fs_vector<double,5>;
    // Default construct
    fs_vector_type fs_vector;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      fs_vector[i] = val;
      val = 2 * val;
    }
    const fs_vector_type& const_fs_vector( fs_vector );
    auto subvector = const_fs_vector.subvector( 2, 5 );
    
    EXPECT_EQ( ( subvector[0] ), ( fs_vector[2] ) );
    EXPECT_EQ( ( subvector[1] ), ( fs_vector[3] ) );
    EXPECT_EQ( ( subvector[2] ), ( fs_vector[4] ) );
  }

  TEST( FS_VECTOR, SUBVECTOR )
  {
    using fs_vector_type = std::math::fs_vector<double,5>;
    // Default construct
    fs_vector_type fs_vector;
    // Set values in vector
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      fs_vector[i] = val;
      val = 2 * val;
    }
    // Get subvector
    auto subvector = fs_vector.subvector( 2, 5 );
    // Modify view
    for ( auto i : { 0, 1, 2 } )
    {
      subvector[i] = val;
      val = 2 * val;
    }
    // Assert original vector has been modified as well
    EXPECT_EQ( ( subvector[0] ), ( fs_vector[2] ) );
    EXPECT_EQ( ( subvector[1] ), ( fs_vector[3] ) );
    EXPECT_EQ( ( subvector[2] ), ( fs_vector[4] ) );
  }

  TEST( FS_VECTOR, NEGATION )
  {
    using vector_type = std::math::fs_vector<double,4>;
    // Construct
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Negate the vector
    vector_type negate_vector { -vector };
    // Access elements from const vector
    auto val1 = negate_vector[0];
    auto val2 = negate_vector[1];
    auto val3 = negate_vector[2];
    auto val4 = negate_vector[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, -1.0 );
    EXPECT_EQ( val2, -2.0 );
    EXPECT_EQ( val3, -3.0 );
    EXPECT_EQ( val4, -4.0 );
  }

  TEST( FS_VECTOR, TRANSPOSE )
  {
    using vector_type = std::math::fs_vector<double,3>;
    // Construct
    vector_type vector;
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    // Transpose the vector
    auto transpose_vector { trans(vector) };
    // Access elements from transpose vector
    auto val1 = transpose_vector[0];
    auto val2 = transpose_vector[1];
    auto val3 = transpose_vector[2];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
  }

  TEST( FS_VECTOR, CONJUGATE )
  {
    using vector_type = std::math::fs_vector<std::complex<double>,3>;
    // Construct
    vector_type vector { };
    // Populate via mutable index access
    vector[0] = std::complex<double>( 1.0, 1.0 );
    vector[1] = std::complex<double>( 2.0, 2.0 );
    vector[2] = std::complex<double>( 3.0, 3.0 );
    // Conjugate the vector
    auto conjugate_vector { conj(vector) };
    // Access elements from conjugate vector
    auto val1 = conjugate_vector[0];
    auto val2 = conjugate_vector[1];
    auto val3 = conjugate_vector[2];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, std::complex<double>( 1.0, -1.0 ) );
    EXPECT_EQ( val2, std::complex<double>( 2.0, -2.0 ) );
    EXPECT_EQ( val3, std::complex<double>( 3.0, -3.0 ) );
  }

  TEST( FS_VECTOR, ADD )
  {
    using vector_type = std::math::fs_vector<double,4>;
    // Construct
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Copy construct
    vector_type vector_copy{ vector };
    // Add the two vectors together
    vector_type vector_sum { vector + vector_copy };
    // Access elements from const vector
    auto val1 = vector_sum[0];
    auto val2 = vector_sum[1];
    auto val3 = vector_sum[2];
    auto val4 = vector_sum[3];
    // Check the vector copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( FS_VECTOR, ADD_ASSIGN )
  {
    using vector_type = std::math::fs_vector<double,4>;
    // Construct
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Copy construct
    vector_type vector_copy{ vector };
    // Add the two vectors together
    static_cast<void>( vector += vector_copy );
    // Access elements from vector
    auto val1 = vector[0];
    auto val2 = vector[1];
    auto val3 = vector[2];
    auto val4 = vector[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( FS_VECTOR, SUBTRACT )
  {
    using vector_type = std::math::fs_vector<double,4>;
    // Construct
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Copy construct
    vector_type vector_copy{ vector };
    // Subtract the two vectors
    vector_type vector_diff { vector - vector_copy };
    // Access elements from const vector
    auto val1 = vector_diff[0];
    auto val2 = vector_diff[1];
    auto val3 = vector_diff[2];
    auto val4 = vector_diff[3];
    // Check the vector copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
  }

  TEST( FS_VECTOR, SUBTRACT_ASSIGN )
  {
    using vector_type = std::math::fs_vector<double,4>;
    // Construct
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Copy construct
    vector_type vector_copy{ vector };
    // Subtract the two vectors
    static_cast<void>( vector -= vector_copy );
    // Access elements from vector
    auto val1 = vector[0];
    auto val2 = vector[1];
    auto val3 = vector[2];
    auto val4 = vector[3];
    // Check the vector copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
  }

  TEST( FS_VECTOR, SCALAR_PREMULTIPLY )
  {
    using vector_type = std::math::fs_vector<double,4>;
    // Construct
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Pre multiply
    vector_type vector_prod { 2 * vector };
    // Access elements from const vector
    auto val1 = vector_prod[0];
    auto val2 = vector_prod[1];
    auto val3 = vector_prod[2];
    auto val4 = vector_prod[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( FS_VECTOR, SCALAR_POSTMULTIPLY )
  {
    using vector_type = std::math::fs_vector<double,4>;
    // Construct
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Post multiply
    vector_type vector_prod { vector * 2 };
    // Access elements from const vector
    auto val1 = vector_prod[0];
    auto val2 = vector_prod[1];
    auto val3 = vector_prod[2];
    auto val4 = vector_prod[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( FS_VECTOR, SCALAR_MULTIPLY_ASSIGN )
  {
    using vector_type = std::math::fs_vector<double,4>;
    // Construct
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Post multiply
    static_cast<void>( vector *= 2 );
    // Access elements from vector
    auto val1 = vector[0];
    auto val2 = vector[1];
    auto val3 = vector[2];
    auto val4 = vector[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( FS_VECTOR, SCALAR_DIVIDE )
  {
    using vector_type = std::math::fs_vector<double,4>;
    // Construct
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Divide
    vector_type vector_divide { vector / 2 };
    // Access elements from const vector
    auto val1 = vector_divide[0];
    auto val2 = vector_divide[1];
    auto val3 = vector_divide[2];
    auto val4 = vector_divide[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
  }

  TEST( FS_VECTOR, SCALAR_DIVIDE_ASSIGN )
  {
    using vector_type = std::math::fs_vector<double,4>;
    // Construct
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    // Divide
    static_cast<void>( vector /= 2 );
    // Access elements from vector
    auto val1 = vector[0];
    auto val2 = vector[1];
    auto val3 = vector[2];
    auto val4 = vector[3];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
  }

  TEST( FS_VECTOR, INNER_PRODUCT )
  {
    using vector_type = std::math::fs_vector<double,6>;
    // Construct
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    vector[3] = 4.0;
    vector[4] = 5.0;
    vector[5] = 6.0;
    // Compute inner product
    auto inner_product_val = inner_prod( vector, vector );
    // Check the inner product was properly computed
    EXPECT_EQ( inner_product_val, 91 );
  }

  TEST( FS_VECTOR, OUTER_PRODUCT )
  {
    using vector_type = std::math::fs_vector<double,3>;
    // Construct
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    // Compute outer product
    auto outer_product = outer_prod( vector, vector );
    // Access elements from outer product
    auto val1 = outer_product[0,0];
    auto val2 = outer_product[0,1];
    auto val3 = outer_product[0,2];
    auto val4 = outer_product[1,0];
    auto val5 = outer_product[1,1];
    auto val6 = outer_product[1,2];
    auto val7 = outer_product[2,0];
    auto val8 = outer_product[2,1];
    auto val9 = outer_product[2,2];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 2.0 );
    EXPECT_EQ( val5, 4.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 3.0 );
    EXPECT_EQ( val8, 6.0 );
    EXPECT_EQ( val9, 9.0 );
  }

  TEST( FS_VECTOR, MATRIX_POSTMULTIPLY )
  {
    using matrix_type = std::math::fs_matrix<double,2,3>;
    using vector_type = std::math::fs_vector<double,2>;
    // Construct matrix
    matrix_type matrix{ };
    // Populate via mutable index access
    matrix[0,0] = 1.0;
    matrix[0,1] = 2.0;
    matrix[0,2] = 3.0;
    matrix[1,0] = 4.0;
    matrix[1,1] = 5.0;
    matrix[1,2] = 6.0;
    // Construct vector
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    // Multiply vector with matrix
    auto vector_multiply { vector * matrix };
    // Access elements from vector
    auto val1 = vector_multiply[0];
    auto val2 = vector_multiply[1];
    auto val3 = vector_multiply[2];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 9.0 );
    EXPECT_EQ( val2, 12.0 );
    EXPECT_EQ( val3, 15.0 );
  }

  TEST( FS_VECTOR, MATRIX_PREMULTIPLY )
  {
    using matrix_type = std::math::fs_matrix<double,2,3>;
    using vector_type = std::math::fs_vector<double,3>;
    // Construct matrix
    matrix_type matrix{ };
    // Populate via mutable index access
    matrix[0,0] = 1.0;
    matrix[0,1] = 2.0;
    matrix[0,2] = 3.0;
    matrix[1,0] = 4.0;
    matrix[1,1] = 5.0;
    matrix[1,2] = 6.0;
    // Construct vector
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    // Multiply matrix with vector
    auto vector_multiply { matrix * vector };
    // Access elements from vector
    auto val1 = vector_multiply[0];
    auto val2 = vector_multiply[1];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 14.0 );
    EXPECT_EQ( val2, 32.0 );
  }

  TEST( FS_VECTOR, MATRIX_MULTIPLY_ASSIGN )
  {
    using matrix_type = std::math::fs_matrix<double,3,3>;
    using vector_type = std::math::fs_vector<double,3>;
    // Construct matrix
    matrix_type matrix{ };
    // Populate via mutable index access
    matrix[0,0] = 1.0;
    matrix[0,1] = 2.0;
    matrix[0,2] = 3.0;
    matrix[1,0] = 4.0;
    matrix[1,1] = 5.0;
    matrix[1,2] = 6.0;
    matrix[2,0] = 7.0;
    matrix[2,1] = 8.0;
    matrix[2,2] = 9.0;
    // Construct vector
    vector_type vector{ };
    // Populate via mutable index access
    vector[0] = 1.0;
    vector[1] = 2.0;
    vector[2] = 3.0;
    // Multiply matrix with vector
    static_cast<void>( vector *= matrix );
    // Access elements from vector
    auto val1 = vector[0];
    auto val2 = vector[1];
    auto val3 = vector[2];
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 30.0 );
    EXPECT_EQ( val2, 36.0 );
    EXPECT_EQ( val3, 42.0 );
  }

}