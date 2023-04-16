#include <gtest/gtest.h>
#include <experimental/linear_algebra.hpp>

namespace
{
  TEST( DR_MATRIX, DEFAULT_CONSTRUCTOR_AND_DESTRUCTOR )
  {
    // Default construction
    std::math::dr_matrix<double> dyn_matrix;
    // Destructor will be called when unit test ends and the dr matrix exits scope
  }

  TEST( DR_MATRIX, MUTABLE_AND_CONST_INDEX_ACCESS )
  {
    // Construct
    std::math::dr_matrix<double> dyn_matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( dyn_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( dyn_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( dyn_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( dyn_matrix, 1, 1 ) = 4.0;
    // Get a const reference
    const std::math::dr_matrix<double>& const_dyn_matrix( dyn_matrix );
    // Access elements from const dyn matrix
    auto val1 = std::math::detail::access( const_dyn_matrix, 0, 0 );
    auto val2 = std::math::detail::access( const_dyn_matrix, 0, 1 );
    auto val3 = std::math::detail::access( const_dyn_matrix, 1, 0 );
    auto val4 = std::math::detail::access( const_dyn_matrix, 1, 1 );
    // Check the dr matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_MATRIX, COPY_CONSTRUCTOR )
  {
    // Construct
    std::math::dr_matrix<double> dyn_matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( dyn_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( dyn_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( dyn_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( dyn_matrix, 1, 1 ) = 4.0;
    // Copy construct
    std::math::dr_matrix<double> dyn_matrix_copy{ dyn_matrix };
    // Get a const reference to copy
    const std::math::dr_matrix<double>& const_dyn_matrix( dyn_matrix_copy );
    // Access elements from const dyn matrix
    auto val1 = std::math::detail::access( const_dyn_matrix, 0, 0 );
    auto val2 = std::math::detail::access( const_dyn_matrix, 0, 1 );
    auto val3 = std::math::detail::access( const_dyn_matrix, 1, 0 );
    auto val4 = std::math::detail::access( const_dyn_matrix, 1, 1 );
    // Check the dyn matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_MATRIX, MOVE_CONSTRUCTOR )
  {
    // Construct
    std::math::dr_matrix<double> dyn_matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( dyn_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( dyn_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( dyn_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( dyn_matrix, 1, 1 ) = 4.0;
    // Move construct
    std::math::dr_matrix<double> dyn_matrix_move{ std::move( dyn_matrix ) };
    // Get a const reference to moved matrix
    const std::math::dr_matrix<double>& const_dyn_matrix( dyn_matrix_move );
    // Access elements from const dyn matrix
    auto val1 = std::math::detail::access( const_dyn_matrix, 0, 0 );
    auto val2 = std::math::detail::access( const_dyn_matrix, 0, 1 );
    auto val3 = std::math::detail::access( const_dyn_matrix, 1, 0 );
    auto val4 = std::math::detail::access( const_dyn_matrix, 1, 1 );
    // Check the dyn matrix engine was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_MATRIX, CONSTRUCT_FROM_VIEW )
  {
    // Construct
    std::math::dr_matrix<double> dyn_matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( dyn_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( dyn_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( dyn_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( dyn_matrix, 1, 1 ) = 4.0;
    // Construct from view
    std::math::dr_matrix<double> dyn_matrix_view{ dyn_matrix.span() };
    // Get a const reference to constructed matrix
    const std::math::dr_matrix<double>& const_dyn_matrix( dyn_matrix_view );
    // Access elements from const dyn matrix
    auto val1 = std::math::detail::access( const_dyn_matrix, 0, 0 );
    auto val2 = std::math::detail::access( const_dyn_matrix, 0, 1 );
    auto val3 = std::math::detail::access( const_dyn_matrix, 1, 0 );
    auto val4 = std::math::detail::access( const_dyn_matrix, 1, 1 );
    // Check the dyn matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_MATRIX, TEMPLATE_COPY_CONSTRUCTOR )
  {
    using float_left_matrix_type   = std::math::fs_matrix<float,2,2,std::experimental::layout_left,std::experimental::default_accessor<float> >;
    using double_right_matrix_type = std::math::dr_matrix<double>;
    // Default construct
    float_left_matrix_type fs_matrix;
    // Populate via mutable index access
    std::math::detail::access( fs_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( fs_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( fs_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( fs_matrix, 1, 1 ) = 4.0;
    // Construct from float matrix
    double_right_matrix_type dyn_matrix_copy{ fs_matrix };
    // Access elements from const dr matrix
    auto val1 = std::math::detail::access( dyn_matrix_copy, 0, 0 );
    auto val2 = std::math::detail::access( dyn_matrix_copy, 0, 1 );
    auto val3 = std::math::detail::access( dyn_matrix_copy, 1, 0 );
    auto val4 = std::math::detail::access( dyn_matrix_copy, 1, 1 );
    // Check the dr matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_MATRIX, CONSTRUCT_FROM_LAMBDA_EXPRESSION )
  {
    using fs_matrix_type = std::math::fs_matrix<double,2,2,std::experimental::layout_right,std::experimental::default_accessor<double> >;
    // Default construct
    fs_matrix_type fs_matrix;
    // Populate via mutable index access
    std::math::detail::access( fs_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( fs_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( fs_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( fs_matrix, 1, 1 ) = 4.0;
    // Get underling view
    auto view = fs_matrix.span();
    // Create a lambda expression from view
    auto lambda = [&view]( auto index1, auto index2 ) { return std::math::detail::access( view, index1, index2 ); };
    // Construct from lambda
    std::math::dr_matrix<double> dyn_matrix_copy( std::experimental::extents<size_t,2,2>(), lambda );
    // Access elements from const dr matrix
    auto val1 = std::math::detail::access( dyn_matrix_copy, 0, 0 );
    auto val2 = std::math::detail::access( dyn_matrix_copy, 0, 1 );
    auto val3 = std::math::detail::access( dyn_matrix_copy, 1, 0 );
    auto val4 = std::math::detail::access( dyn_matrix_copy, 1, 1 );
    // Check the dr matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_MATRIX, ASSIGNMENT_OPERATOR )
  {
    // Construct
    std::math::dr_matrix<double> dyn_matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( dyn_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( dyn_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( dyn_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( dyn_matrix, 1, 1 ) = 4.0;
    // Default construct and assign
    std::math::dr_matrix<double> dyn_matrix_copy;
    dyn_matrix_copy = dyn_matrix;
    // Access elements from dr matrix
    auto val1 = std::math::detail::access( dyn_matrix_copy, 0, 0 );
    auto val2 = std::math::detail::access( dyn_matrix_copy, 0, 1 );
    auto val3 = std::math::detail::access( dyn_matrix_copy, 1, 0 );
    auto val4 = std::math::detail::access( dyn_matrix_copy, 1, 1 );
    // Check the dr matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_MATRIX, TEMPLATE_ASSIGNMENT_OPERATOR )
  {
    using float_left_matrix_type   = std::math::fs_matrix<float,2,2,std::experimental::layout_left,std::experimental::default_accessor<float> >;
    // Default construct
    float_left_matrix_type fs_matrix;
    // Populate via mutable index access
    std::math::detail::access( fs_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( fs_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( fs_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( fs_matrix, 1, 1 ) = 4.0;
    // Default construct and then assign
    std::math::dr_matrix<double> dyn_matrix_copy;
    dyn_matrix_copy = fs_matrix;
    // Access elements from const dr matrix
    auto val1 = std::math::detail::access( dyn_matrix_copy, 0, 0 );
    auto val2 = std::math::detail::access( dyn_matrix_copy, 0, 1 );
    auto val3 = std::math::detail::access( dyn_matrix_copy, 1, 0 );
    auto val4 = std::math::detail::access( dyn_matrix_copy, 1, 1 );
    // Check the dr matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_MATRIX, ASSIGN_FROM_VIEW )
  {
    using fs_matrix_type = std::math::fs_matrix<double,2,2,std::experimental::layout_left,std::experimental::default_accessor<double> >;
    // Default construct
    fs_matrix_type fs_matrix;
    // Populate via mutable index access
    std::math::detail::access( fs_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( fs_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( fs_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( fs_matrix, 1, 1 ) = 4.0;
    // Construct and assign from view
    std::math::dr_matrix<double> dyn_matrix_view;
    dyn_matrix_view = fs_matrix.span();
    // Access elements from const dyr matrix
    auto val1 = std::math::detail::access( dyn_matrix_view, 0, 0 );
    auto val2 = std::math::detail::access( dyn_matrix_view, 0, 1 );
    auto val3 = std::math::detail::access( dyn_matrix_view, 1, 0 );
    auto val4 = std::math::detail::access( dyn_matrix_view, 1, 1 );
    // Check the dr matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_MATRIX, SIZE_AND_CAPACITY )
  {
    // Construct
    std::math::dr_matrix<double> dyn_matrix{ std::experimental::extents<size_t,2,5>(), std::experimental::extents<size_t,3,5>() };
    EXPECT_TRUE( ( dyn_matrix.size().extent(0) == 2 ) );
    EXPECT_TRUE( ( dyn_matrix.size().extent(1) == 5 ) );
    EXPECT_TRUE( ( dyn_matrix.rows() == 2 ) );
    EXPECT_TRUE( ( dyn_matrix.columns() == 5 ) );
    EXPECT_TRUE( ( dyn_matrix.capacity().extent(0) == 3 ) );
    EXPECT_TRUE( ( dyn_matrix.capacity().extent(1) == 5 ) );
    EXPECT_TRUE( ( dyn_matrix.row_capacity() == 3 ) );
    EXPECT_TRUE( ( dyn_matrix.column_capacity() == 5 ) );
  }

  TEST( DR_MATRIX, RESIZE )
  {
    // Construct
    std::math::dr_matrix<double> dyn_matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( dyn_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( dyn_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( dyn_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( dyn_matrix, 1, 1 ) = 4.0;
    // Resize
    dyn_matrix.resize( std::experimental::extents<size_t,3,3>() );
    std::math::detail::access( dyn_matrix, 0, 2 ) = 5.0;
    std::math::detail::access( dyn_matrix, 1, 2 ) = 6.0;
    std::math::detail::access( dyn_matrix, 2, 0 ) = 7.0;
    std::math::detail::access( dyn_matrix, 2, 1 ) = 8.0;
    std::math::detail::access( dyn_matrix, 2, 2 ) = 9.0;
    // Get values
    auto val1 = std::math::detail::access( dyn_matrix, 0, 0 );
    auto val2 = std::math::detail::access( dyn_matrix, 0, 1 );
    auto val3 = std::math::detail::access( dyn_matrix, 1, 0 );
    auto val4 = std::math::detail::access( dyn_matrix, 1, 1 );
    auto val5 = std::math::detail::access( dyn_matrix, 0, 2 );
    auto val6 = std::math::detail::access( dyn_matrix, 1, 2 );
    auto val7 = std::math::detail::access( dyn_matrix, 2, 0 );
    auto val8 = std::math::detail::access( dyn_matrix, 2, 1 );
    auto val9 = std::math::detail::access( dyn_matrix, 2, 2 );
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

  TEST( DR_MATRIX, RESERVE )
  {
    // Construct
    std::math::dr_matrix<double> dyn_matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,2,2>() };
    // Populate via mutable index access
    std::math::detail::access( dyn_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( dyn_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( dyn_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( dyn_matrix, 1, 1 ) = 4.0;
    // Resize
    dyn_matrix.reserve( std::experimental::extents<size_t,4,4>() );
    // Get values
    auto val1  = std::math::detail::access( dyn_matrix, 0, 0 );
    auto val2  = std::math::detail::access( dyn_matrix, 0, 1 );
    auto val3  = std::math::detail::access( dyn_matrix, 1, 0 );
    auto val4  = std::math::detail::access( dyn_matrix, 1, 1 );
    // Check the values are correct
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( DR_MATRIX, CONST_SUBMATRIX )
  {
    // Construct
    std::math::dr_matrix<double> dyn_matrix{ std::experimental::extents<size_t,5,5>(), std::experimental::extents<size_t,10,10>() };
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( dyn_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const std::math::dr_matrix<double>& const_dyn_matrix( dyn_matrix );
    auto submatrix = const_dyn_matrix.submatrix( std::tuple(2,2), std::tuple(5,4) );
    // Assert submatrix maps to original matrix
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 0 ) ), ( std::math::detail::access( dyn_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 0 ) ), ( std::math::detail::access( dyn_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 0 ) ), ( std::math::detail::access( dyn_matrix, 4, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 1 ) ), ( std::math::detail::access( dyn_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 1 ) ), ( std::math::detail::access( dyn_matrix, 3, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 1 ) ), ( std::math::detail::access( dyn_matrix, 4, 3 ) ) );
  }

  TEST( DR_MATRIX, CONST_ROW_VECTOR )
  {
    // Construct
    std::math::dr_matrix<double> dyn_matrix{ std::experimental::extents<size_t,5,5>(), std::experimental::extents<size_t,10,10>() };
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( dyn_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const std::math::dr_matrix<double>& const_dyn_matrix( dyn_matrix );
    auto row_vector = const_dyn_matrix.row( 2 );
    // Assert submatrix maps to original matrix
    EXPECT_EQ( ( std::math::detail::access( row_vector, 0 ) ), ( std::math::detail::access( dyn_matrix, 2, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 1 ) ), ( std::math::detail::access( dyn_matrix, 2, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 2 ) ), ( std::math::detail::access( dyn_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 3 ) ), ( std::math::detail::access( dyn_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 4 ) ), ( std::math::detail::access( dyn_matrix, 2, 4 ) ) );
  }

  TEST( DR_MATRIX, CONST_COLUMN_VECTOR )
  {
    // Construct
    std::math::dr_matrix<double> dyn_matrix{ std::experimental::extents<size_t,5,5>(), std::experimental::extents<size_t,10,10>() };
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( dyn_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const std::math::dr_matrix<double>& const_dyn_matrix( dyn_matrix );
    auto column_vector = const_dyn_matrix.column( 2 );
    // Assert submatrix maps to original matrix
    EXPECT_EQ( ( std::math::detail::access( column_vector, 0 ) ), ( std::math::detail::access( dyn_matrix, 0, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 1 ) ), ( std::math::detail::access( dyn_matrix, 1, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 2 ) ), ( std::math::detail::access( dyn_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 3 ) ), ( std::math::detail::access( dyn_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 4 ) ), ( std::math::detail::access( dyn_matrix, 4, 2 ) ) );
  }

  TEST( DR_MATRIX, SUBMATRIX )
  {
    // Construct
    std::math::dr_matrix<double> dyn_matrix{ std::experimental::extents<size_t,5,5>(), std::experimental::extents<size_t,10,10>() };
    // Set values in tensor engine
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( dyn_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    // Get submatrix
    auto submatrix = dyn_matrix.submatrix( std::tuple(2,2), std::tuple(5,4) );
    // Modify view
    for ( auto i : { 0, 1, 2 } )
    {
      for ( auto j : { 0, 1 } )
      {
        std::math::detail::access( submatrix, i, j ) = val;
        val = 2 * val;
      }
    }
    // Assert original matrix has been modified as well
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 0 ) ), ( std::math::detail::access( dyn_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 0 ) ), ( std::math::detail::access( dyn_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 0 ) ), ( std::math::detail::access( dyn_matrix, 4, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 1 ) ), ( std::math::detail::access( dyn_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 1 ) ), ( std::math::detail::access( dyn_matrix, 3, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 1 ) ), ( std::math::detail::access( dyn_matrix, 4, 3 ) ) );
  }

  TEST( DR_MATRIX, ROW_VECTOR )
  {
    // Construct
    std::math::dr_matrix<double> dyn_matrix{ std::experimental::extents<size_t,5,5>(), std::experimental::extents<size_t,10,10>() };
    // Set values in tensor engine
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( dyn_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    // Get row vector
    auto row_vector = dyn_matrix.row( 2 );
    // Modify view
    for ( auto i : { 0, 1, 2 } )
    {
      std::math::detail::access( row_vector, i ) = val;
      val = 2 * val;
    }
    // Assert original matrix has been modified as well
    EXPECT_EQ( ( std::math::detail::access( row_vector, 0 ) ), ( std::math::detail::access( dyn_matrix, 2, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 1 ) ), ( std::math::detail::access( dyn_matrix, 2, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 2 ) ), ( std::math::detail::access( dyn_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 3 ) ), ( std::math::detail::access( dyn_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 4 ) ), ( std::math::detail::access( dyn_matrix, 2, 4 ) ) );
  }

  TEST( DR_MATRIX, COLUMN_VECTOR )
  {
    // Construct
    std::math::dr_matrix<double> dyn_matrix{ std::experimental::extents<size_t,5,5>(), std::experimental::extents<size_t,10,10>() };
    // Set values in tensor engine
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( dyn_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    // Get column vector
    auto column_vector = dyn_matrix.column( 2 );
    // Modify view
    for ( auto i : { 0, 1, 2 } )
    {
      std::math::detail::access( column_vector, i ) = val;
      val = 2 * val;
    }
    // Assert original matrix has been modified as well
    EXPECT_EQ( ( std::math::detail::access( column_vector, 0 ) ), ( std::math::detail::access( dyn_matrix, 0, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 1 ) ), ( std::math::detail::access( dyn_matrix, 1, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 2 ) ), ( std::math::detail::access( dyn_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 3 ) ), ( std::math::detail::access( dyn_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 4 ) ), ( std::math::detail::access( dyn_matrix, 4, 2 ) ) );
  }

  TEST( DR_MATRIX, NEGATION )
  {
    using matrix_type = std::math::dr_matrix<double>;
    // Construct
    matrix_type matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Negate the matrix
    matrix_type negate_matrix { -matrix };
    // Access elements from const matrix
    auto val1 = std::math::detail::access( negate_matrix, 0, 0 );
    auto val2 = std::math::detail::access( negate_matrix, 0, 1 );
    auto val3 = std::math::detail::access( negate_matrix, 1, 0 );
    auto val4 = std::math::detail::access( negate_matrix, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, -1.0 );
    EXPECT_EQ( val2, -2.0 );
    EXPECT_EQ( val3, -3.0 );
    EXPECT_EQ( val4, -4.0 );
  }

  TEST( DR_MATRIX, TRANSPOSE )
  {
    using matrix_type = std::math::dr_matrix<double>;
    // Construct
    matrix_type matrix { std::experimental::extents<size_t,3,2>(), std::experimental::extents<size_t,3,2>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    std::math::detail::access( matrix, 2, 0 ) = 5.0;
    std::math::detail::access( matrix, 2, 1 ) = 6.0;
    // Transpose the matrix
    auto transpose_matrix { trans(matrix) };
    // Access elements from transpose matrix
    auto val1 = std::math::detail::access( transpose_matrix, 0, 0 );
    auto val2 = std::math::detail::access( transpose_matrix, 1, 0 );
    auto val3 = std::math::detail::access( transpose_matrix, 0, 1 );
    auto val4 = std::math::detail::access( transpose_matrix, 1, 1 );
    auto val5 = std::math::detail::access( transpose_matrix, 0, 2 );
    auto val6 = std::math::detail::access( transpose_matrix, 1, 2 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
  }

  TEST( DR_MATRIX, CONJUGATE )
  {
    using matrix_type = std::math::dr_matrix< std::complex<double> >;
    // Construct
    matrix_type matrix { std::experimental::extents<size_t,3,2>(), std::experimental::extents<size_t,3,2>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = std::complex<double>( 1.0, 1.0 );
    std::math::detail::access( matrix, 0, 1 ) = std::complex<double>( 2.0, 2.0 );
    std::math::detail::access( matrix, 1, 0 ) = std::complex<double>( 3.0, 3.0 );
    std::math::detail::access( matrix, 1, 1 ) = std::complex<double>( 4.0, 4.0 );
    std::math::detail::access( matrix, 2, 0 ) = std::complex<double>( 5.0, 5.0 );
    std::math::detail::access( matrix, 2, 1 ) = std::complex<double>( 6.0, 6.0 );
    // Conjugate the matrix
    auto conjugate_matrix { conj(matrix) };
    // Access elements from conjugate matrix
    auto val1 = std::math::detail::access( conjugate_matrix, 0, 0 );
    auto val2 = std::math::detail::access( conjugate_matrix, 1, 0 );
    auto val3 = std::math::detail::access( conjugate_matrix, 0, 1 );
    auto val4 = std::math::detail::access( conjugate_matrix, 1, 1 );
    auto val5 = std::math::detail::access( conjugate_matrix, 0, 2 );
    auto val6 = std::math::detail::access( conjugate_matrix, 1, 2 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, std::complex<double>( 1.0, -1.0 ) );
    EXPECT_EQ( val2, std::complex<double>( 2.0, -2.0 ) );
    EXPECT_EQ( val3, std::complex<double>( 3.0, -3.0 ) );
    EXPECT_EQ( val4, std::complex<double>( 4.0, -4.0 ) );
    EXPECT_EQ( val5, std::complex<double>( 5.0, -5.0 ) );
    EXPECT_EQ( val6, std::complex<double>( 6.0, -6.0 ) );
  }

  TEST( DR_MATRIX, ADD )
  {
    using matrix_type = std::math::dr_matrix<double>;
    // Construct
    matrix_type matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Copy construct
    matrix_type matrix_copy{ matrix };
    // Add the two matrixs together
    matrix_type matrix_sum { matrix + matrix_copy };
    // Access elements from const matrix
    auto val1 = std::math::detail::access( matrix_sum, 0, 0 );
    auto val2 = std::math::detail::access( matrix_sum, 0, 1 );
    auto val3 = std::math::detail::access( matrix_sum, 1, 0 );
    auto val4 = std::math::detail::access( matrix_sum, 1, 1 );
    // Check the matrix copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( DR_MATRIX, ADD_ASSIGN )
  {
    using matrix_type = std::math::dr_matrix<double>;
    // Construct
    matrix_type matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Copy construct
    matrix_type matrix_copy{ matrix };
    // Add the two matrixs together
    static_cast<void>( matrix += matrix_copy );
    // Access elements from matrix
    auto val1 = std::math::detail::access( matrix, 0, 0 );
    auto val2 = std::math::detail::access( matrix, 0, 1 );
    auto val3 = std::math::detail::access( matrix, 1, 0 );
    auto val4 = std::math::detail::access( matrix, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( DR_MATRIX, SUBTRACT )
  {
    using matrix_type = std::math::dr_matrix<double>;
    // Construct
    matrix_type matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Copy construct
    matrix_type matrix_copy{ matrix };
    // Subtract the two matrixs
    matrix_type matrix_diff { matrix - matrix_copy };
    // Access elements from const matrix
    auto val1 = std::math::detail::access( matrix_diff, 0, 0 );
    auto val2 = std::math::detail::access( matrix_diff, 0, 1 );
    auto val3 = std::math::detail::access( matrix_diff, 1, 0 );
    auto val4 = std::math::detail::access( matrix_diff, 1, 1 );
    // Check the matrix copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
  }

  TEST( DR_MATRIX, SUBTRACT_ASSIGN )
  {
    using matrix_type = std::math::dr_matrix<double>;
    // Construct
    matrix_type matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Copy construct
    matrix_type matrix_copy{ matrix };
    // Subtract the two matrixs
    static_cast<void>( matrix -= matrix_copy );
    // Access elements from matrix
    auto val1 = std::math::detail::access( matrix, 0, 0 );
    auto val2 = std::math::detail::access( matrix, 0, 1 );
    auto val3 = std::math::detail::access( matrix, 1, 0 );
    auto val4 = std::math::detail::access( matrix, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
  }

  TEST( DR_MATRIX, SCALAR_PREMULTIPLY )
  {
    using matrix_type = std::math::dr_matrix<double>;
    // Construct
    matrix_type matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Pre multiply
    matrix_type matrix_prod { 2 * matrix };
    // Access elements from const matrix
    auto val1 = std::math::detail::access( matrix_prod, 0, 0 );
    auto val2 = std::math::detail::access( matrix_prod, 0, 1 );
    auto val3 = std::math::detail::access( matrix_prod, 1, 0 );
    auto val4 = std::math::detail::access( matrix_prod, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( DR_MATRIX, SCALAR_POSTMULTIPLY )
  {
    using matrix_type = std::math::dr_matrix<double>;
    // Construct
    matrix_type matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Post multiply
    matrix_type matrix_prod { matrix * 2 };
    // Access elements from const matrix
    auto val1 = std::math::detail::access( matrix_prod, 0, 0 );
    auto val2 = std::math::detail::access( matrix_prod, 0, 1 );
    auto val3 = std::math::detail::access( matrix_prod, 1, 0 );
    auto val4 = std::math::detail::access( matrix_prod, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( DR_MATRIX, SCALAR_MULTIPLY_ASSIGN )
  {
    using matrix_type = std::math::dr_matrix<double>;
    // Construct
    matrix_type matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Post multiply
    static_cast<void>( matrix *= 2 );
    // Access elements from matrix
    auto val1 = std::math::detail::access( matrix, 0, 0 );
    auto val2 = std::math::detail::access( matrix, 0, 1 );
    auto val3 = std::math::detail::access( matrix, 1, 0 );
    auto val4 = std::math::detail::access( matrix, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( DR_MATRIX, SCALAR_DIVIDE )
  {
    using matrix_type = std::math::dr_matrix<double>;
    // Construct
    matrix_type matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Divide
    matrix_type matrix_divide { matrix / 2 };
    // Access elements from const matrix
    auto val1 = std::math::detail::access( matrix_divide, 0, 0 );
    auto val2 = std::math::detail::access( matrix_divide, 0, 1 );
    auto val3 = std::math::detail::access( matrix_divide, 1, 0 );
    auto val4 = std::math::detail::access( matrix_divide, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
  }

  TEST( DR_MATRIX, SCALAR_DIVIDE_ASSIGN )
  {
    using matrix_type = std::math::dr_matrix<double>;
    // Construct
    matrix_type matrix{ std::experimental::extents<size_t,2,2>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Divide
    static_cast<void>( matrix /= 2 );
    // Access elements from matrix
    auto val1 = std::math::detail::access( matrix, 0, 0 );
    auto val2 = std::math::detail::access( matrix, 0, 1 );
    auto val3 = std::math::detail::access( matrix, 1, 0 );
    auto val4 = std::math::detail::access( matrix, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
  }

  TEST( DR_MATRIX, MATRIX_MULTIPLY )
  {
    using matrix_type = std::math::dr_matrix<double>;
    // Construct
    matrix_type matrix{ std::experimental::extents<size_t,2,3>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 0, 2 ) = 3.0;
    std::math::detail::access( matrix, 1, 0 ) = 4.0;
    std::math::detail::access( matrix, 1, 1 ) = 5.0;
    std::math::detail::access( matrix, 1, 2 ) = 6.0;
    // Multiply matrix with its transpose
    auto matrix_multiply { matrix * trans(matrix) };
    // Access elements from const matrix
    auto val1 = std::math::detail::access( matrix_multiply, 0, 0 );
    auto val2 = std::math::detail::access( matrix_multiply, 0, 1 );
    auto val3 = std::math::detail::access( matrix_multiply, 1, 0 );
    auto val4 = std::math::detail::access( matrix_multiply, 1, 1 );
    // Check the matrices were properly multiplied
    EXPECT_EQ( val1, 14.0 );
    EXPECT_EQ( val2, 32.0 );
    EXPECT_EQ( val3, 32.0 );
    EXPECT_EQ( val4, 77.0 );
  }

  TEST( DR_MATRIX, MATRIX_MULTIPLY_ASSIGN )
  {
    using matrix_type = std::math::dr_matrix<double>;
    // Construct
    matrix_type matrix{ std::experimental::extents<size_t,2,3>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 0, 2 ) = 3.0;
    std::math::detail::access( matrix, 1, 0 ) = 4.0;
    std::math::detail::access( matrix, 1, 1 ) = 5.0;
    std::math::detail::access( matrix, 1, 2 ) = 6.0;
    // Multiply matrix with its transpose
    static_cast<void>( matrix *= trans(matrix) );
    // Access elements from const matrix
    auto val1 = std::math::detail::access( matrix, 0, 0 );
    auto val2 = std::math::detail::access( matrix, 0, 1 );
    auto val3 = std::math::detail::access( matrix, 1, 0 );
    auto val4 = std::math::detail::access( matrix, 1, 1 );
    // Check the matrices were properly multiplied
    EXPECT_EQ( val1, 14.0 );
    EXPECT_EQ( val2, 32.0 );
    EXPECT_EQ( val3, 32.0 );
    EXPECT_EQ( val4, 77.0 );
  }

  TEST( DR_MATRIX, VECTOR_PREMULTIPLY )
  {
    using matrix_type = std::math::dr_matrix<double>;
    using vector_type = std::math::dr_vector<double>;
    // Construct matrix
    matrix_type matrix{ std::experimental::extents<size_t,2,3>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 0, 2 ) = 3.0;
    std::math::detail::access( matrix, 1, 0 ) = 4.0;
    std::math::detail::access( matrix, 1, 1 ) = 5.0;
    std::math::detail::access( matrix, 1, 2 ) = 6.0;
    // Construct vector
    vector_type vector{ std::experimental::extents<size_t,2>(), std::experimental::extents<size_t,4>() };
    // Populate via mutable index access
    std::math::detail::access( vector, 0 ) = 1.0;
    std::math::detail::access( vector, 1 ) = 2.0;
    // Multiply vector with matrix
    auto vector_multiply { vector * matrix };
    // Access elements from vector
    auto val1 = std::math::detail::access( vector_multiply, 0 );
    auto val2 = std::math::detail::access( vector_multiply, 1 );
    auto val3 = std::math::detail::access( vector_multiply, 2 );
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 9.0 );
    EXPECT_EQ( val2, 12.0 );
    EXPECT_EQ( val3, 15.0 );
  }

  TEST( DR_MATRIX, VECTOR_POSTMULTIPLY )
  {
    using matrix_type = std::math::dr_matrix<double>;
    using vector_type = std::math::dr_vector<double>;
    // Construct matrix
    matrix_type matrix{ std::experimental::extents<size_t,2,3>(), std::experimental::extents<size_t,3,3>() };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 0, 2 ) = 3.0;
    std::math::detail::access( matrix, 1, 0 ) = 4.0;
    std::math::detail::access( matrix, 1, 1 ) = 5.0;
    std::math::detail::access( matrix, 1, 2 ) = 6.0;
    // Construct vector
    vector_type vector{ std::experimental::extents<size_t,3>(), std::experimental::extents<size_t,4>() };
    // Populate via mutable index access
    std::math::detail::access( vector, 0 ) = 1.0;
    std::math::detail::access( vector, 1 ) = 2.0;
    std::math::detail::access( vector, 2 ) = 3.0;
    // Multiply matrix with vector
    auto vector_multiply { matrix * vector };
    // Access elements from vector
    auto val1 = std::math::detail::access( vector_multiply, 0 );
    auto val2 = std::math::detail::access( vector_multiply, 1 );
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 14.0 );
    EXPECT_EQ( val2, 32.0 );
  }

  TEST( FS_MATRIX, DEFAULT_CONSTRUCTOR_AND_DESTRUCTOR )
  {
    // Default construction
    std::math::fs_matrix<double,2,2> fs_matrix;
    // Destructor will be called when unit test ends and the fs matrix exits scope
  }

  TEST( FS_MATRIX, MUTABLE_AND_CONST_INDEX_ACCESS )
  {
    using fs_matrix_type = std::math::fs_matrix<double,2,2>;
    // Default construct
    fs_matrix_type fs_matrix;
    // Populate via mutable index access
    std::math::detail::access( fs_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( fs_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( fs_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( fs_matrix, 1, 1 ) = 4.0;
    // Get a const reference
    const fs_matrix_type& const_fs_matrix( fs_matrix );
    // Access elements from const fs matrix engine
    auto val1 = std::math::detail::access( const_fs_matrix, 0, 0 );
    auto val2 = std::math::detail::access( const_fs_matrix, 0, 1 );
    auto val3 = std::math::detail::access( const_fs_matrix, 1, 0 );
    auto val4 = std::math::detail::access( const_fs_matrix, 1, 1 );
    // Check the fs matrix engine was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_MATRIX, COPY_CONSTRUCTOR )
  {
    using fs_matrix_type = std::math::fs_matrix<double,2,2>;
    // Default construct
    fs_matrix_type fs_matrix;
    // Populate via mutable index access
    std::math::detail::access( fs_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( fs_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( fs_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( fs_matrix, 1, 1 ) = 4.0;
    // Copy construct
    fs_matrix_type fs_matrix_copy{ fs_matrix };
    // Get a const reference to copy
    const fs_matrix_type& const_fs_matrix( fs_matrix_copy );
    // Access elements from const fs matrix
    auto val1 = std::math::detail::access( const_fs_matrix, 0, 0 );
    auto val2 = std::math::detail::access( const_fs_matrix, 0, 1 );
    auto val3 = std::math::detail::access( const_fs_matrix, 1, 0 );
    auto val4 = std::math::detail::access( const_fs_matrix, 1, 1 );
    // Check the fs matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_MATRIX, MOVE_CONSTRUCTOR )
  {
    using fs_matrix_type = std::math::fs_matrix<double,2,2>;
    // Default construct
    fs_matrix_type fs_matrix;
    // Populate via mutable index access
    std::math::detail::access( fs_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( fs_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( fs_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( fs_matrix, 1, 1 ) = 4.0;
    // Move construct
    fs_matrix_type fs_matrix_move{ std::move( fs_matrix ) };
    // Get a const reference to moved matrix
    const fs_matrix_type& const_fs_matrix( fs_matrix_move );
    // Access elements from const fs matrix
    auto val1 = std::math::detail::access( const_fs_matrix, 0, 0 );
    auto val2 = std::math::detail::access( const_fs_matrix, 0, 1 );
    auto val3 = std::math::detail::access( const_fs_matrix, 1, 0 );
    auto val4 = std::math::detail::access( const_fs_matrix, 1, 1 );
    // Check the fs matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_MATRIX, CONSTRUCT_FROM_VIEW )
  {
    using fs_matrix_type = std::math::fs_matrix<double,2,2>;
    // Default construct
    fs_matrix_type fs_matrix;
    // Populate via mutable index access
    std::math::detail::access( fs_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( fs_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( fs_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( fs_matrix, 1, 1 ) = 4.0;
    // Construct from view
    fs_matrix_type fs_matrix_view{ fs_matrix.span() };
    // Get a const reference to constructed matrix
    const fs_matrix_type& const_fs_matrix( fs_matrix_view );
    // Access elements from const fs matrix matrix
    auto val1 = std::math::detail::access( const_fs_matrix, 0, 0 );
    auto val2 = std::math::detail::access( const_fs_matrix, 0, 1 );
    auto val3 = std::math::detail::access( const_fs_matrix, 1, 0 );
    auto val4 = std::math::detail::access( const_fs_matrix, 1, 1 );
    // Check the fs matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_MATRIX, TEMPLATE_COPY_CONSTRUCTOR )
  {
    using float_left_matrix_type   = std::math::fs_matrix<float,2,2>;
    using double_right_matrix_type = std::math::fs_matrix<double,2,2>;
    // Default construct
    float_left_matrix_type fs_matrix;
    // Populate via mutable index access
    std::math::detail::access( fs_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( fs_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( fs_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( fs_matrix, 1, 1 ) = 4.0;
    // Construct from float matrix
    double_right_matrix_type fs_matrix_copy{ fs_matrix };
    // Access elements from const fs matrix
    auto val1 = std::math::detail::access( fs_matrix_copy, 0, 0 );
    auto val2 = std::math::detail::access( fs_matrix_copy, 0, 1 );
    auto val3 = std::math::detail::access( fs_matrix_copy, 1, 0 );
    auto val4 = std::math::detail::access( fs_matrix_copy, 1, 1 );
    // Check the fs matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_MATRIX, CONSTRUCT_FROM_LAMBDA_EXPRESSION )
  {
    using fs_matrix_type = std::math::fs_matrix<double,2,2>;
    // Default construct
    fs_matrix_type fs_matrix;
    // Populate via mutable index access
    std::math::detail::access( fs_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( fs_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( fs_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( fs_matrix, 1, 1 ) = 4.0;
    // Get underling view
    auto view = fs_matrix.span();
    // Create a lambda expression from view
    auto lambda = [&view]( auto index1, auto index2 ) { return std::math::detail::access( view, index1, index2 ); };
    // Construct from lambda
    fs_matrix_type fs_matrix_copy( lambda );
    // Access elements from const fs matrix
    auto val1 = std::math::detail::access( fs_matrix_copy, 0, 0 );
    auto val2 = std::math::detail::access( fs_matrix_copy, 0, 1 );
    auto val3 = std::math::detail::access( fs_matrix_copy, 1, 0 );
    auto val4 = std::math::detail::access( fs_matrix_copy, 1, 1 );
    // Check the fs matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_MATRIX, ASSIGNMENT_OPERATOR )
  {
    using fs_matrix_type = std::math::fs_matrix<double,2,2>;
    // Default construct
    fs_matrix_type fs_matrix;
    // Populate via mutable index access
    std::math::detail::access( fs_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( fs_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( fs_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( fs_matrix, 1, 1 ) = 4.0;
    // Construct from lambda
    fs_matrix_type fs_matrix_copy;
    fs_matrix_copy = fs_matrix;
    // Access elements from const fs matrix
    auto val1 = std::math::detail::access( fs_matrix_copy, 0, 0 );
    auto val2 = std::math::detail::access( fs_matrix_copy, 0, 1 );
    auto val3 = std::math::detail::access( fs_matrix_copy, 1, 0 );
    auto val4 = std::math::detail::access( fs_matrix_copy, 1, 1 );
    // Check the fs matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_MATRIX, TEMPLATE_ASSIGNMENT_OPERATOR )
  {
    using float_left_matrix_type   = std::math::fs_matrix<float,2,2>;
    using double_right_matrix_type = std::math::fs_matrix<double,2,2>;
    // Default construct
    float_left_matrix_type fs_matrix;
    // Populate via mutable index access
    std::math::detail::access( fs_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( fs_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( fs_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( fs_matrix, 1, 1 ) = 4.0;
    // Default construct and then assign
    double_right_matrix_type fs_matrix_copy;
    fs_matrix_copy = fs_matrix;
    // Access elements from const fs matrix
    auto val1 = std::math::detail::access( fs_matrix_copy, 0, 0 );
    auto val2 = std::math::detail::access( fs_matrix_copy, 0, 1 );
    auto val3 = std::math::detail::access( fs_matrix_copy, 1, 0 );
    auto val4 = std::math::detail::access( fs_matrix_copy, 1, 1 );
    // Check the fs matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_MATRIX, ASSIGN_FROM_VIEW )
  {
    using fs_matrix_type = std::math::fs_matrix<double,2,2>;
    // Default construct
    fs_matrix_type fs_matrix;
    // Populate via mutable index access
    std::math::detail::access( fs_matrix, 0, 0 ) = 1.0;
    std::math::detail::access( fs_matrix, 0, 1 ) = 2.0;
    std::math::detail::access( fs_matrix, 1, 0 ) = 3.0;
    std::math::detail::access( fs_matrix, 1, 1 ) = 4.0;
    // Default construct and assign from view
    fs_matrix_type fs_matrix_view;
    fs_matrix_view = fs_matrix.span();
    // Get a const reference to constructed matrix
    const fs_matrix_type& const_fs_matrix( fs_matrix_view );
    // Access elements from const fs matrix
    auto val1 = std::math::detail::access( const_fs_matrix, 0, 0 );
    auto val2 = std::math::detail::access( const_fs_matrix, 0, 1 );
    auto val3 = std::math::detail::access( const_fs_matrix, 1, 0 );
    auto val4 = std::math::detail::access( const_fs_matrix, 1, 1 );
    // Check the fs matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
  }

  TEST( FS_MATRIX, SIZE_AND_CAPACITY )
  {
    using fs_matrix_type = std::math::fs_matrix<double,2,5>;
    // Default construct
    fs_matrix_type fs_matrix;
    EXPECT_TRUE( ( fs_matrix.size().extent(0) == 2 ) );
    EXPECT_TRUE( ( fs_matrix.size().extent(1) == 5 ) );
    EXPECT_TRUE( ( fs_matrix.rows() == 2 ) );
    EXPECT_TRUE( ( fs_matrix.columns() == 5 ) );
    EXPECT_TRUE( ( fs_matrix.capacity().extent(0) == 2 ) );
    EXPECT_TRUE( ( fs_matrix.capacity().extent(1) == 5 ) );
    EXPECT_TRUE( ( fs_matrix.row_capacity() == 2 ) );
    EXPECT_TRUE( ( fs_matrix.column_capacity() == 5 ) );
  }

  TEST( FS_MATRIX, CONST_SUBMATRIX )
  {
    using fs_matrix_type = std::math::fs_matrix<double,5,5>;
    // Default construct
    fs_matrix_type fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const fs_matrix_type& const_fs_matrix( fs_matrix );
    auto submatrix = const_fs_matrix.submatrix( std::tuple(2,2), std::tuple(5,4) );
    
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 0 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 0 ) ), ( std::math::detail::access( fs_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 0 ) ), ( std::math::detail::access( fs_matrix, 4, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 1 ) ), ( std::math::detail::access( fs_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 1 ) ), ( std::math::detail::access( fs_matrix, 3, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 1 ) ), ( std::math::detail::access( fs_matrix, 4, 3 ) ) );
  }

  TEST( FS_MATRIX, CONST_ROW_VECTOR )
  {
    // Construct
    std::math::fs_matrix<double,5,5> fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const std::math::fs_matrix<double,5,5>& const_fs_matrix( fs_matrix );
    auto row_vector = const_fs_matrix.row( 2 );
    // Assert submatrix maps to original matrix
    EXPECT_EQ( ( std::math::detail::access( row_vector, 0 ) ), ( std::math::detail::access( fs_matrix, 2, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 1 ) ), ( std::math::detail::access( fs_matrix, 2, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 2 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 3 ) ), ( std::math::detail::access( fs_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 4 ) ), ( std::math::detail::access( fs_matrix, 2, 4 ) ) );
  }

  TEST( FS_MATRIX, CONST_COLUMN_VECTOR )
  {
    // Construct
    std::math::fs_matrix<double,5,5> fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const std::math::fs_matrix<double,5,5>& const_fs_matrix( fs_matrix );
    auto column_vector = const_fs_matrix.column( 2 );
    // Assert submatrix maps to original matrix
    EXPECT_EQ( ( std::math::detail::access( column_vector, 0 ) ), ( std::math::detail::access( fs_matrix, 0, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 1 ) ), ( std::math::detail::access( fs_matrix, 1, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 2 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 3 ) ), ( std::math::detail::access( fs_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 4 ) ), ( std::math::detail::access( fs_matrix, 4, 2 ) ) );
  }

  TEST( FS_MATRIX, SUBMATRIX )
  {
    using fs_matrix_type = std::math::fs_matrix<double,5,5>;
    // Default construct
    fs_matrix_type fs_matrix;
    // Set values in matrix
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    // Get submatrix
    auto submatrix = fs_matrix.submatrix( std::tuple(2,2), std::tuple(5,4) );
    // Modify view
    for ( auto i : { 0, 1, 2 } )
    {
      for ( auto j : { 0, 1 } )
      {
        std::math::detail::access( submatrix, i, j ) = val;
        val = 2 * val;
      }
    }
    // Assert original matrix has been modified as well
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 0 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 0 ) ), ( std::math::detail::access( fs_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 0 ) ), ( std::math::detail::access( fs_matrix, 4, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 1 ) ), ( std::math::detail::access( fs_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 1 ) ), ( std::math::detail::access( fs_matrix, 3, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 1 ) ), ( std::math::detail::access( fs_matrix, 4, 3 ) ) );
  }

  TEST( FS_MATRIX, ROW_VECTOR )
  {
    // Construct
    std::math::fs_matrix<double,5,5> fs_matrix;
    // Set values in matrix
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    // Get row vector
    auto row_vector = fs_matrix.row( 2 );
    // Modify view
    for ( auto i : { 0, 1, 2 } )
    {
      std::math::detail::access( row_vector, i ) = val;
      val = 2 * val;
    }
    // Assert original matrix has been modified as well
    EXPECT_EQ( ( std::math::detail::access( row_vector, 0 ) ), ( std::math::detail::access( fs_matrix, 2, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 1 ) ), ( std::math::detail::access( fs_matrix, 2, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 2 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 3 ) ), ( std::math::detail::access( fs_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row_vector, 4 ) ), ( std::math::detail::access( fs_matrix, 2, 4 ) ) );
  }

  TEST( FS_MATRIX, COLUMN_VECTOR )
  {
    // Construct
    std::math::fs_matrix<double,5,5> fs_matrix;
    // Set values in matrix
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    // Get column vector
    auto column_vector = fs_matrix.column( 2 );
    // Modify view
    for ( auto i : { 0, 1, 2 } )
    {
      std::math::detail::access( column_vector, i ) = val;
      val = 2 * val;
    }
    // Assert original matrix has been modified as well
    EXPECT_EQ( ( std::math::detail::access( column_vector, 0 ) ), ( std::math::detail::access( fs_matrix, 0, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 1 ) ), ( std::math::detail::access( fs_matrix, 1, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 2 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 3 ) ), ( std::math::detail::access( fs_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column_vector, 4 ) ), ( std::math::detail::access( fs_matrix, 4, 2 ) ) );
  }

  TEST( FS_MATRIX, NEGATION )
  {
    using matrix_type = std::math::fs_matrix<double,2,2>;
    // Construct
    matrix_type matrix{ };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Negate the matrix
    matrix_type negate_matrix { -matrix };
    // Access elements from const matrix
    auto val1 = std::math::detail::access( negate_matrix, 0, 0 );
    auto val2 = std::math::detail::access( negate_matrix, 0, 1 );
    auto val3 = std::math::detail::access( negate_matrix, 1, 0 );
    auto val4 = std::math::detail::access( negate_matrix, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, -1.0 );
    EXPECT_EQ( val2, -2.0 );
    EXPECT_EQ( val3, -3.0 );
    EXPECT_EQ( val4, -4.0 );
  }

  TEST( FS_MATRIX, TRANSPOSE )
  {
    using matrix_type = std::math::fs_matrix<double,3,2>;
    // Construct
    matrix_type matrix { };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    std::math::detail::access( matrix, 2, 0 ) = 5.0;
    std::math::detail::access( matrix, 2, 1 ) = 6.0;
    // Transpose the matrix
    auto transpose_matrix { trans(matrix) };
    // Access elements from transpose matrix
    auto val1 = std::math::detail::access( transpose_matrix, 0, 0 );
    auto val2 = std::math::detail::access( transpose_matrix, 1, 0 );
    auto val3 = std::math::detail::access( transpose_matrix, 0, 1 );
    auto val4 = std::math::detail::access( transpose_matrix, 1, 1 );
    auto val5 = std::math::detail::access( transpose_matrix, 0, 2 );
    auto val6 = std::math::detail::access( transpose_matrix, 1, 2 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 4.0 );
    EXPECT_EQ( val5, 5.0 );
    EXPECT_EQ( val6, 6.0 );
  }

  TEST( FS_MATRIX, CONJUGATE )
  {
    using matrix_type = std::math::fs_matrix<std::complex<double>,3,2>;
    // Construct
    matrix_type matrix { };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = std::complex<double>( 1.0, 1.0 );
    std::math::detail::access( matrix, 0, 1 ) = std::complex<double>( 2.0, 2.0 );
    std::math::detail::access( matrix, 1, 0 ) = std::complex<double>( 3.0, 3.0 );
    std::math::detail::access( matrix, 1, 1 ) = std::complex<double>( 4.0, 4.0 );
    std::math::detail::access( matrix, 2, 0 ) = std::complex<double>( 5.0, 5.0 );
    std::math::detail::access( matrix, 2, 1 ) = std::complex<double>( 6.0, 6.0 );
    // Conjugate the matrix
    auto conjugate_matrix { conj(matrix) };
    // Access elements from conjugate matrix
    auto val1 = std::math::detail::access( conjugate_matrix, 0, 0 );
    auto val2 = std::math::detail::access( conjugate_matrix, 1, 0 );
    auto val3 = std::math::detail::access( conjugate_matrix, 0, 1 );
    auto val4 = std::math::detail::access( conjugate_matrix, 1, 1 );
    auto val5 = std::math::detail::access( conjugate_matrix, 0, 2 );
    auto val6 = std::math::detail::access( conjugate_matrix, 1, 2 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, std::complex<double>( 1.0, -1.0 ) );
    EXPECT_EQ( val2, std::complex<double>( 2.0, -2.0 ) );
    EXPECT_EQ( val3, std::complex<double>( 3.0, -3.0 ) );
    EXPECT_EQ( val4, std::complex<double>( 4.0, -4.0 ) );
    EXPECT_EQ( val5, std::complex<double>( 5.0, -5.0 ) );
    EXPECT_EQ( val6, std::complex<double>( 6.0, -6.0 ) );
  }

  TEST( FS_MATRIX, ADD )
  {
    using matrix_type = std::math::fs_matrix<double,2,2>;
    // Construct
    matrix_type matrix{ };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Copy construct
    matrix_type matrix_copy{ matrix };
    // Add the two matrixs together
    matrix_type matrix_sum { matrix + matrix_copy };
    // Access elements from const matrix
    auto val1 = std::math::detail::access( matrix_sum, 0, 0 );
    auto val2 = std::math::detail::access( matrix_sum, 0, 1 );
    auto val3 = std::math::detail::access( matrix_sum, 1, 0 );
    auto val4 = std::math::detail::access( matrix_sum, 1, 1 );
    // Check the matrix copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( FS_MATRIX, ADD_ASSIGN )
  {
    using matrix_type = std::math::fs_matrix<double,2,2>;
    // Construct
    matrix_type matrix{ };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Copy construct
    matrix_type matrix_copy{ matrix };
    // Add the two matrixs together
    static_cast<void>( matrix += matrix_copy );
    // Access elements from matrix
    auto val1 = std::math::detail::access( matrix, 0, 0 );
    auto val2 = std::math::detail::access( matrix, 0, 1 );
    auto val3 = std::math::detail::access( matrix, 1, 0 );
    auto val4 = std::math::detail::access( matrix, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( FS_MATRIX, SUBTRACT )
  {
    using matrix_type = std::math::fs_matrix<double,2,2>;
    // Construct
    matrix_type matrix{ };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Copy construct
    matrix_type matrix_copy{ matrix };
    // Subtract the two matrixs
    matrix_type matrix_diff { matrix - matrix_copy };
    // Access elements from const matrix
    auto val1 = std::math::detail::access( matrix_diff, 0, 0 );
    auto val2 = std::math::detail::access( matrix_diff, 0, 1 );
    auto val3 = std::math::detail::access( matrix_diff, 1, 0 );
    auto val4 = std::math::detail::access( matrix_diff, 1, 1 );
    // Check the matrix copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
  }

  TEST( FS_MATRIX, SUBTRACT_ASSIGN )
  {
    using matrix_type = std::math::fs_matrix<double,2,2>;
    // Construct
    matrix_type matrix{ };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Copy construct
    matrix_type matrix_copy{ matrix };
    // Subtract the two matrixs
    static_cast<void>( matrix -= matrix_copy );
    // Access elements from matrix
    auto val1 = std::math::detail::access( matrix, 0, 0 );
    auto val2 = std::math::detail::access( matrix, 0, 1 );
    auto val3 = std::math::detail::access( matrix, 1, 0 );
    auto val4 = std::math::detail::access( matrix, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0 );
    EXPECT_EQ( val2, 0 );
    EXPECT_EQ( val3, 0 );
    EXPECT_EQ( val4, 0 );
  }

  TEST( FS_MATRIX, SCALAR_PREMULTIPLY )
  {
    using matrix_type = std::math::fs_matrix<double,2,2>;
    // Construct
    matrix_type matrix{ };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Pre multiply
    matrix_type matrix_prod { 2 * matrix };
    // Access elements from const matrix
    auto val1 = std::math::detail::access( matrix_prod, 0, 0 );
    auto val2 = std::math::detail::access( matrix_prod, 0, 1 );
    auto val3 = std::math::detail::access( matrix_prod, 1, 0 );
    auto val4 = std::math::detail::access( matrix_prod, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( FS_MATRIX, SCALAR_POSTMULTIPLY )
  {
    using matrix_type = std::math::fs_matrix<double,2,2>;
    // Construct
    matrix_type matrix{ };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Post multiply
    matrix_type matrix_prod { matrix * 2 };
    // Access elements from const matrix
    auto val1 = std::math::detail::access( matrix_prod, 0, 0 );
    auto val2 = std::math::detail::access( matrix_prod, 0, 1 );
    auto val3 = std::math::detail::access( matrix_prod, 1, 0 );
    auto val4 = std::math::detail::access( matrix_prod, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( FS_MATRIX, SCALAR_MULTIPLY_ASSIGN )
  {
    using matrix_type = std::math::fs_matrix<double,2,2>;
    // Construct
    matrix_type matrix{ };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Post multiply
    static_cast<void>( matrix *= 2 );
    // Access elements from matrix
    auto val1 = std::math::detail::access( matrix, 0, 0 );
    auto val2 = std::math::detail::access( matrix, 0, 1 );
    auto val3 = std::math::detail::access( matrix, 1, 0 );
    auto val4 = std::math::detail::access( matrix, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 2.0 );
    EXPECT_EQ( val2, 4.0 );
    EXPECT_EQ( val3, 6.0 );
    EXPECT_EQ( val4, 8.0 );
  }

  TEST( FS_MATRIX, SCALAR_DIVIDE )
  {
    using matrix_type = std::math::fs_matrix<double,2,2>;
    // Construct
    matrix_type matrix{ };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Divide
    matrix_type matrix_divide { matrix / 2 };
    // Access elements from const matrix
    auto val1 = std::math::detail::access( matrix_divide, 0, 0 );
    auto val2 = std::math::detail::access( matrix_divide, 0, 1 );
    auto val3 = std::math::detail::access( matrix_divide, 1, 0 );
    auto val4 = std::math::detail::access( matrix_divide, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
  }

  TEST( FS_MATRIX, SCALAR_DIVIDE_ASSIGN )
  {
    using matrix_type = std::math::fs_matrix<double,2,2>;
    // Construct
    matrix_type matrix{ };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 1, 0 ) = 3.0;
    std::math::detail::access( matrix, 1, 1 ) = 4.0;
    // Divide
    static_cast<void>( matrix /= 2 );
    // Access elements from matrix
    auto val1 = std::math::detail::access( matrix, 0, 0 );
    auto val2 = std::math::detail::access( matrix, 0, 1 );
    auto val3 = std::math::detail::access( matrix, 1, 0 );
    auto val4 = std::math::detail::access( matrix, 1, 1 );
    // Check the matrix was populated correctly and provided the correct values
    EXPECT_EQ( val1, 0.5 );
    EXPECT_EQ( val2, 1.0 );
    EXPECT_EQ( val3, 1.5 );
    EXPECT_EQ( val4, 2.0 );
  }

  TEST( FS_MATRIX, MATRIX_MULTIPLY )
  {
    using matrix_type = std::math::fs_matrix<double,2,3>;
    // Construct
    matrix_type matrix{ };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 0, 2 ) = 3.0;
    std::math::detail::access( matrix, 1, 0 ) = 4.0;
    std::math::detail::access( matrix, 1, 1 ) = 5.0;
    std::math::detail::access( matrix, 1, 2 ) = 6.0;
    // Multiply matrix with its transpose
    auto matrix_multiply { matrix * trans(matrix) };
    // Access elements from const matrix
    auto val1 = std::math::detail::access( matrix_multiply, 0, 0 );
    auto val2 = std::math::detail::access( matrix_multiply, 0, 1 );
    auto val3 = std::math::detail::access( matrix_multiply, 1, 0 );
    auto val4 = std::math::detail::access( matrix_multiply, 1, 1 );
    // Check the matrices were properly multiplied
    EXPECT_EQ( val1, 14.0 );
    EXPECT_EQ( val2, 32.0 );
    EXPECT_EQ( val3, 32.0 );
    EXPECT_EQ( val4, 77.0 );
  }

  TEST( FS_MATRIX, MATRIX_MULTIPLY_ASSIGN )
  {
    using matrix_type = std::math::fs_matrix<double,3,3>;
    // Construct
    matrix_type matrix{ };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 0, 2 ) = 3.0;
    std::math::detail::access( matrix, 1, 0 ) = 4.0;
    std::math::detail::access( matrix, 1, 1 ) = 5.0;
    std::math::detail::access( matrix, 1, 2 ) = 6.0;
    std::math::detail::access( matrix, 2, 0 ) = 7.0;
    std::math::detail::access( matrix, 2, 1 ) = 8.0;
    std::math::detail::access( matrix, 2, 2 ) = 9.0;
    // Multiply matrix with its transpose
    static_cast<void>( matrix *= trans(matrix) );
    // Access elements from matrix
    auto val1 = std::math::detail::access( matrix, 0, 0 );
    auto val2 = std::math::detail::access( matrix, 0, 1 );
    auto val3 = std::math::detail::access( matrix, 0, 2 );
    auto val4 = std::math::detail::access( matrix, 1, 0 );
    auto val5 = std::math::detail::access( matrix, 1, 1 );
    auto val6 = std::math::detail::access( matrix, 1, 2 );
    auto val7 = std::math::detail::access( matrix, 2, 0 );
    auto val8 = std::math::detail::access( matrix, 2, 1 );
    auto val9 = std::math::detail::access( matrix, 2, 2 );
    // Check the matrices were properly multiplied
    EXPECT_EQ( val1, 14.0 );
    EXPECT_EQ( val2, 32.0 );
    EXPECT_EQ( val3, 50.0 );
    EXPECT_EQ( val4, 32.0 );
    EXPECT_EQ( val5, 77.0 );
    EXPECT_EQ( val6, 122.0 );
    EXPECT_EQ( val7, 50.0 );
    EXPECT_EQ( val8, 122.0 );
    EXPECT_EQ( val9, 194.0 );
  }

  TEST( FS_MATRIX, VECTOR_PREMULTIPLY )
  {
    using matrix_type = std::math::fs_matrix<double,2,3>;
    using vector_type = std::math::fs_vector<double,2>;
    // Construct matrix
    matrix_type matrix{ };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 0, 2 ) = 3.0;
    std::math::detail::access( matrix, 1, 0 ) = 4.0;
    std::math::detail::access( matrix, 1, 1 ) = 5.0;
    std::math::detail::access( matrix, 1, 2 ) = 6.0;
    // Construct vector
    vector_type vector{ };
    // Populate via mutable index access
    std::math::detail::access( vector, 0 ) = 1.0;
    std::math::detail::access( vector, 1 ) = 2.0;
    // Multiply vector with matrix
    auto vector_multiply { vector * matrix };
    // Access elements from vector
    auto val1 = std::math::detail::access( vector_multiply, 0 );
    auto val2 = std::math::detail::access( vector_multiply, 1 );
    auto val3 = std::math::detail::access( vector_multiply, 2 );
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 9.0 );
    EXPECT_EQ( val2, 12.0 );
    EXPECT_EQ( val3, 15.0 );
  }

  TEST( FS_MATRIX, VECTOR_POSTMULTIPLY )
  {
    using matrix_type = std::math::fs_matrix<double,2,3>;
    using vector_type = std::math::fs_vector<double,3>;
    // Construct matrix
    matrix_type matrix{ };
    // Populate via mutable index access
    std::math::detail::access( matrix, 0, 0 ) = 1.0;
    std::math::detail::access( matrix, 0, 1 ) = 2.0;
    std::math::detail::access( matrix, 0, 2 ) = 3.0;
    std::math::detail::access( matrix, 1, 0 ) = 4.0;
    std::math::detail::access( matrix, 1, 1 ) = 5.0;
    std::math::detail::access( matrix, 1, 2 ) = 6.0;
    // Construct vector
    vector_type vector{ };
    // Populate via mutable index access
    std::math::detail::access( vector, 0 ) = 1.0;
    std::math::detail::access( vector, 1 ) = 2.0;
    std::math::detail::access( vector, 2 ) = 3.0;
    // Multiply matrix with vector
    auto vector_multiply { matrix * vector };
    // Access elements from vector
    auto val1 = std::math::detail::access( vector_multiply, 0 );
    auto val2 = std::math::detail::access( vector_multiply, 1 );
    // Check the vector was populated correctly and provided the correct values
    EXPECT_EQ( val1, 14.0 );
    EXPECT_EQ( val2, 32.0 );
  }

  TEST( MATRIX_VIEW, SIZE_AND_CAPACITY )
  {
    using fs_matrix_type = std::math::fs_matrix<double,5,5>;
    // Default construct
    fs_matrix_type fs_matrix { []( auto, auto ) { return 0.0; } };
    auto submatrix = fs_matrix.submatrix( std::tuple(0,2), std::tuple(0,3) );
    EXPECT_TRUE( ( submatrix.size().extent(0) == 2 ) );
    EXPECT_TRUE( ( submatrix.size().extent(1) == 3 ) );
    EXPECT_TRUE( ( submatrix.rows() == 2 ) );
    EXPECT_TRUE( ( submatrix.columns() == 3 ) );
    EXPECT_TRUE( ( submatrix.capacity().extent(0) == 2 ) );
    EXPECT_TRUE( ( submatrix.capacity().extent(1) == 3 ) );
    EXPECT_TRUE( ( submatrix.row_capacity() == 2 ) );
    EXPECT_TRUE( ( submatrix.column_capacity() == 3 ) );
  }

  TEST( MATRIX_VIEW, CONST_COLUMN_VECTOR )
  {
    using fs_matrix_type = std::math::fs_matrix<double,5,5>;
    // Default construct
    fs_matrix_type fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const fs_matrix_type& const_fs_matrix( fs_matrix );
    auto submatrix = const_fs_matrix.submatrix( std::tuple(2,5), std::tuple(0,3) );
    auto column = ( (const decltype(submatrix)&)( submatrix ) ).column( 0 );
    
    EXPECT_EQ( ( std::math::detail::access( column, 0 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column, 1 ) ), ( std::math::detail::access( fs_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column, 2 ) ), ( std::math::detail::access( fs_matrix, 4, 2 ) ) );
  }

  TEST( MATRIX_VIEW, CONST_ROW_ROW )
  {
    using fs_matrix_type = std::math::fs_matrix<double,5,5>;
    // Default construct
    fs_matrix_type fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const fs_matrix_type& const_fs_matrix( fs_matrix );
    auto submatrix = const_fs_matrix.submatrix( std::tuple(2,5), std::tuple(0,3) );
    auto row = ( (const decltype(submatrix)&)( submatrix ) ).row( 0 );
    
    EXPECT_EQ( ( std::math::detail::access( row, 0 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row, 1 ) ), ( std::math::detail::access( fs_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row, 2 ) ), ( std::math::detail::access( fs_matrix, 2, 4 ) ) );
  }

  TEST( MATRIX_VIEW, CONST_SUBMATRIX )
  {
    using fs_matrix_type = std::math::fs_matrix<double,5,5>;
    // Default construct
    fs_matrix_type fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const fs_matrix_type& const_fs_matrix( fs_matrix );
    auto submatrix  = const_fs_matrix.submatrix( std::tuple(2,5), std::tuple(0,3) );
    auto submatrix2 = ( (const decltype(submatrix)&)( submatrix ) ).submatrix( std::tuple(0,1), std::tuple(0,1) );
    
    EXPECT_EQ( ( std::math::detail::access( submatrix2, 0, 0 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix2, 0, 1 ) ), ( std::math::detail::access( fs_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix2, 1, 0 ) ), ( std::math::detail::access( fs_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix2, 1, 1 ) ), ( std::math::detail::access( fs_matrix, 3, 3 ) ) );
  }

  TEST( MATRIX_VIEW, COLUMN_VECTOR )
  {
    using fs_matrix_type = std::math::fs_matrix<double,5,5>;
    // Default construct
    fs_matrix_type fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    auto submatrix = fs_matrix.submatrix( std::tuple(2,5), std::tuple(0,3) );
    auto column = submatrix.column( 0 );
    for ( auto i : { 0, 1, 2 } )
    {
      std::math::detail::access( column, i ) = val;
      val = 2 * val;
    }
    
    EXPECT_EQ( ( std::math::detail::access( column, 0 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column, 1 ) ), ( std::math::detail::access( fs_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( column, 2 ) ), ( std::math::detail::access( fs_matrix, 4, 2 ) ) );
  }

  TEST( MATRIX_VIEW, ROW_VECTOR )
  {
    using fs_matrix_type = std::math::fs_matrix<double,5,5>;
    // Default construct
    fs_matrix_type fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    auto submatrix = fs_matrix.submatrix( std::tuple(2,5), std::tuple(0,3) );
    auto row = submatrix.column( 0 );
    for ( auto i : { 0, 1, 2 } )
    {
      std::math::detail::access( row, i ) = val;
      val = 2 * val;
    }
    
    EXPECT_EQ( ( std::math::detail::access( row, 0 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row, 1 ) ), ( std::math::detail::access( fs_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( row, 2 ) ), ( std::math::detail::access( fs_matrix, 2, 4 ) ) );
  }

  TEST( MATRIX_VIEW, SUBMATRIX )
  {
    using fs_matrix_type = std::math::fs_matrix<double,5,5>;
    // Default construct
    fs_matrix_type fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    auto submatrix  = fs_matrix.submatrix( std::tuple(2,5), std::tuple(0,3) );
    auto submatrix2 = submatrix.subvector( std::tuple(1,3), std::tuple(1,2) );
    for ( auto i : { 0, 1 } )
    {
      for ( auto j : { 0 } )
      {
        std::math::detail::access( submatrix2, i, j ) = val;
        val = 2 * val;
      }
    }
    
    EXPECT_EQ( ( std::math::detail::access( submatrix2, 0, 0 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix2, 0, 1 ) ), ( std::math::detail::access( fs_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix2, 1, 0 ) ), ( std::math::detail::access( fs_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix2, 1, 1 ) ), ( std::math::detail::access( fs_matrix, 3, 3 ) ) );
  }

  TEST( MATRIX_VIEW, NEGATION )
  {
    // Construct
    std::math::fs_matrix<double,5,5> fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const std::math::fs_matrix<double,5,5>& const_fs_matrix( fs_matrix );
    auto submatrix = const_fs_matrix.submatrix( std::tuple(2,5), std::tuple(2,4) );
    // Negate submatrix
    auto negate_submatrix = -submatrix;

    EXPECT_EQ( ( std::math::detail::access( negate_submatrix, 0, 0 ) ), ( -std::math::detail::access( submatrix, 0, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( negate_submatrix, 1, 0 ) ), ( -std::math::detail::access( submatrix, 1, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( negate_submatrix, 2, 0 ) ), ( -std::math::detail::access( submatrix, 2, 0 ) ) );
    EXPECT_EQ( ( std::math::detail::access( negate_submatrix, 0, 1 ) ), ( -std::math::detail::access( submatrix, 0, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( negate_submatrix, 1, 1 ) ), ( -std::math::detail::access( submatrix, 1, 1 ) ) );
    EXPECT_EQ( ( std::math::detail::access( negate_submatrix, 2, 1 ) ), ( -std::math::detail::access( submatrix, 2, 1 ) ) );
  }

  TEST( MATRIX_VIEW, ADD )
  {
    // Construct
    std::math::fs_matrix<double,5,5> fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const std::math::fs_matrix<double,5,5>& const_fs_matrix( fs_matrix );
    auto submatrix = const_fs_matrix.submatrix( std::tuple(2,5), std::tuple(2,4) );
    // Add the submatrix with itself
    auto submatrix_sum = submatrix + submatrix;

    EXPECT_EQ( ( std::math::detail::access( submatrix_sum, 0, 0 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_sum, 0, 1 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_sum, 0, 2 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 2, 4 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_sum, 1, 0 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_sum, 1, 1 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 3, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_sum, 1, 2 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 3, 4 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_sum, 2, 0 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 4, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_sum, 2, 1 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 4, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_sum, 2, 2 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 4, 4 ) ) );
  }

  TEST( MATRIX_VIEW, ADD_ASSIGN )
  {
    // Construct
    std::math::fs_matrix<double,5,5> fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    auto submatrix = fs_matrix.submatrix( std::tuple(2,5), std::tuple(2,4) );
    // Add the submatrix with itself
    static_cast<void>( submatrix += submatrix );

    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 0 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 1 ) ), ( std::math::detail::access( fs_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 2 ) ), ( std::math::detail::access( fs_matrix, 2, 4 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 0 ) ), ( std::math::detail::access( fs_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 1 ) ), ( std::math::detail::access( fs_matrix, 3, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 2 ) ), ( std::math::detail::access( fs_matrix, 3, 4 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 0 ) ), ( std::math::detail::access( fs_matrix, 4, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 1 ) ), ( std::math::detail::access( fs_matrix, 4, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 2 ) ), ( std::math::detail::access( fs_matrix, 4, 4 ) ) );
  }

  TEST( MATRIX_VIEW, SUBTRACT )
  {
    // Construct
    std::math::fs_matrix<double,5,5> fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const std::math::fs_matrix<double,5,5>& const_fs_matrix( fs_matrix );
    auto submatrix = const_fs_matrix.submatrix( std::tuple(2,5), std::tuple(2,4) );
    // Subtract the submatrix with itself
    auto submatrix_diff = submatrix - submatrix;

    EXPECT_EQ( ( std::math::detail::access( submatrix_diff, 0, 0 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix_diff, 0, 1 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix_diff, 0, 2 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix_diff, 1, 0 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix_diff, 1, 1 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix_diff, 1, 2 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix_diff, 2, 0 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix_diff, 2, 1 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix_diff, 2, 2 ) ), 0 );
  }

  TEST( MATRIX_VIEW, SUBTRACT_ASSIGN )
  {
    // Construct
    std::math::fs_matrix<double,5,5> fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    auto submatrix = fs_matrix.submatrix( std::tuple(2,5), std::tuple(2,4) );
    // Subtract the submatrix with itself
    static_cast<void>( submatrix -= fs_matrix.submatrix( std::tuple(2,5), std::tuple(2,4) ) );

    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 0 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 1 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 2 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 0 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 1 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 2 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 0 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 1 ) ), 0 );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 2 ) ), 0 );
  }

  TEST( MATRIX_VIEW, SCALAR_PREMULTIPLY )
  {
    // Construct
    std::math::fs_matrix<double,5,5> fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const std::math::fs_matrix<double,5,5>& const_fs_matrix( fs_matrix );
    auto submatrix = const_fs_matrix.submatrix( std::tuple(2,5), std::tuple(2,4) );
    // Multiply the submatrix with a constant
    auto submatrix_prod = 2.0 * submatrix;

    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 0, 0 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 0, 1 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 0, 2 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 2, 4 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 1, 0 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 1, 1 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 3, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 1, 2 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 3, 4 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 2, 0 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 4, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 2, 1 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 4, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 2, 2 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 4, 4 ) ) );
  }

  TEST( MATRIX_VIEW, SCALAR_POSTMULTIPLY )
  {
    // Construct
    std::math::fs_matrix<double,5,5> fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const std::math::fs_matrix<double,5,5>& const_fs_matrix( fs_matrix );
    auto submatrix = const_fs_matrix.submatrix( std::tuple(2,5), std::tuple(2,4) );
    // Multiply the submatrix with a constant
    auto submatrix_prod = submatrix * 2.0;

    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 0, 0 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 0, 1 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 0, 2 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 2, 4 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 1, 0 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 1, 1 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 3, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 1, 2 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 3, 4 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 2, 0 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 4, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 2, 1 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 4, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_prod, 2, 2 ) ), ( 2.0 * std::math::detail::access( fs_matrix, 4, 4 ) ) );
  }

  TEST( MATRIX_VIEW, SCALAR_MULTIPLY_ASSIGN )
  {
    // Construct
    std::math::fs_matrix<double,5,5> fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    auto submatrix = fs_matrix.submatrix( std::tuple(2,5), std::tuple(2,4) );
    // Multiply the submatrix with a constant
    submatrix *= 2.0;

    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 0 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 1 ) ), ( std::math::detail::access( fs_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 2 ) ), ( std::math::detail::access( fs_matrix, 2, 4 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 0 ) ), ( std::math::detail::access( fs_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 1 ) ), ( std::math::detail::access( fs_matrix, 3, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 2 ) ), ( std::math::detail::access( fs_matrix, 3, 4 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 0 ) ), ( std::math::detail::access( fs_matrix, 4, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 1 ) ), ( std::math::detail::access( fs_matrix, 4, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 2 ) ), ( std::math::detail::access( fs_matrix, 4, 4 ) ) );
  }

  TEST( MATRIX_VIEW, SCALAR_DIVIDE )
  {
    // Construct
    std::math::fs_matrix<double,5,5> fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    const std::math::fs_matrix<double,5,5>& const_fs_matrix( fs_matrix );
    auto submatrix = const_fs_matrix.submatrix( std::tuple(2,5), std::tuple(2,4) );
    // DIvide the submatrix with a constant
    auto submatrix_divide = submatrix / 2.0;

    EXPECT_EQ( ( std::math::detail::access( submatrix_divide, 0, 0 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) / 2.0 ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_divide, 0, 1 ) ), ( std::math::detail::access( fs_matrix, 2, 3 ) / 2.0 ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_divide, 0, 2 ) ), ( std::math::detail::access( fs_matrix, 2, 4 ) / 2.0 ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_divide, 1, 0 ) ), ( std::math::detail::access( fs_matrix, 3, 2 ) / 2.0 ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_divide, 1, 1 ) ), ( std::math::detail::access( fs_matrix, 3, 3 ) / 2.0 ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_divide, 1, 2 ) ), ( std::math::detail::access( fs_matrix, 3, 4 ) / 2.0 ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_divide, 2, 0 ) ), ( std::math::detail::access( fs_matrix, 4, 2 ) / 2.0 ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_divide, 2, 1 ) ), ( std::math::detail::access( fs_matrix, 4, 3 ) / 2.0 ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix_divide, 2, 2 ) ), ( std::math::detail::access( fs_matrix, 4, 4 ) / 2.0 ) );
  }

  TEST( MATRIX_VIEW, SCALAR_DIVIDE_ASSIGN )
  {
    // Construct
    std::math::fs_matrix<double,5,5> fs_matrix;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        std::math::detail::access( fs_matrix, i, j ) = val;
        val = 2 * val;
      }
    }
    auto submatrix = fs_matrix.submatrix( std::tuple(2,5), std::tuple(2,4) );
    // Divide the submatrix with a constant
    static_cast<void>( submatrix /= 2.0 );

    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 0 ) ), ( std::math::detail::access( fs_matrix, 2, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 1 ) ), ( std::math::detail::access( fs_matrix, 2, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 0, 2 ) ), ( std::math::detail::access( fs_matrix, 2, 4 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 0 ) ), ( std::math::detail::access( fs_matrix, 3, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 1 ) ), ( std::math::detail::access( fs_matrix, 3, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 1, 2 ) ), ( std::math::detail::access( fs_matrix, 3, 4 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 0 ) ), ( std::math::detail::access( fs_matrix, 4, 2 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 1 ) ), ( std::math::detail::access( fs_matrix, 4, 3 ) ) );
    EXPECT_EQ( ( std::math::detail::access( submatrix, 2, 2 ) ), ( std::math::detail::access( fs_matrix, 4, 4 ) ) );
  }

}