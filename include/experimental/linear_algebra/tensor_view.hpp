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
namespace experimental
{
namespace math
{

/// @brief Non-owning ensor view.
//         Implementation satisfies the following concepts:
//         concepts::tensor
//         concepts::writable_tensor if MDS::element_type is non-const
//         concepts::readable_tensor if MDS::element_type is const
/// @tparam MDS mdspan
template < class MDS
#ifdef LINALG_ENABLE_CONCEPTS
  > requires ( detail::is_mdspan_v<MDS> &&
             MDS::is_always_unique() ) // Each element in the mdspan must have a unique mapping. (i.e. span_type and const_underlying_span_type should be the same.)
#else
  , typename >
#endif
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
    using size_type                  = ::std::size_t;
    /// @brief Type used to represent a node in the tensor
    using tuple_type                 = typename detail::template extents_helper<size_type,extents_type::rank()>::tuple_type;
    /// @brief Type used to const view memory
    using const_underlying_span_type = ::std::experimental::mdspan<const ::std::remove_const_t<element_type>,extents_type,layout_type,accessor_type>;
    /// @brief Type used to view memory
    using underlying_span_type       = ::std::experimental::mdspan<element_type,extents_type,layout_type,accessor_type>;
    /// @brief Type used to portray tensor as an N dimensional view
    using span_type                  = const_underlying_span_type;
    /// @brief Type returned by mutable index access
    using reference                  = typename MDS::reference;
    
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

    //- Data access
    
    /// @brief returns a const view
    /// @returns const view
    [[nodiscard]] constexpr span_type                  span() const noexcept;
    /// @brief the view
    /// @returns returns the view
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename Elem = element_type, typename = ::std::enable_if_t< !::std::is_const_v<Elem> > >
    #endif
    [[nodiscard]] constexpr underlying_span_type       underlying_span() noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( !is_const_v<element_type> );
    #else
      ;
    #endif
    /// @brief returns a const view
    /// @returns const view
    [[nodiscard]] constexpr const_underlying_span_type underlying_span() const noexcept;

    //- Const views

    /// @brief Returns the value at (indices...)
    /// @param indices set indices representing a node in the tensor
    /// @returns value at row i, column j, depth k, etc.
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... IndexType >
    [[nodiscard]] constexpr value_type operator[]( IndexType ... indices ) const noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(IndexType) == extents_type::rank() ) && ( ::std::is_convertible_v<IndexType,index_type> && ... );
    #else
      ;
    #endif
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... IndexType >
    [[nodiscard]] constexpr value_type operator()( IndexType ... indices ) const noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(IndexType) == extents_type::rank() ) && ( ::std::is_convertible_v<IndexType,index_type> && ... );
    #else
      ;
    #endif
    #endif
    /// @brief Returns a const vector view
    /// @tparam ...SliceArgs argument types used to get a const vector view
    /// @param ...args aguments to get a const vector view
    /// @return const vector view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto subvector( SliceArgs ... args ) const
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 1 );
    #else
      ;
    #endif
    /// @brief Returns a const matrix view
    /// @tparam ...SliceArgs argument types used to get a const matrix view
    /// @param ...args aguments to get a const matrix view
    /// @return const matrix view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto submatrix( SliceArgs ... args ) const
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 2 );
    #else
      ;
    #endif
    /// @brief Returns a const view of the specified subtensor
    /// @tparam ...SliceArgs argument types used to get a tensor view
    /// @param ...args aguments to get a tensor view
    /// @return const tensor view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto subtensor( SliceArgs ... args ) const;

    //- Mutable views

    /// @brief Returns a mutable value at (indices...)
    /// @param indices set indices representing a node in the tensor
    /// @returns mutable value at row i, column j, depth k, etc.
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... IndexType >
    [[nodiscard]] constexpr reference operator[]( IndexType ... indices ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(IndexType) == extents_type::rank() ) && ( ::std::is_convertible_v<IndexType,index_type> && ... ) &&
               ( !::std::is_const_v<element_type> );
    #else
      ;
    #endif
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... IndexType >
    [[nodiscard]] constexpr reference operator()( IndexType ... indices ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(IndexType) == extents_type::rank() ) && ( ::std::is_convertible_v<IndexType,index_type> && ... ) &&
               ( !::std::is_const_v<element_type> );
    #else
      ;
    #endif
    #endif
    /// @brief Returns a vector view
    /// @tparam ...SliceArgs argument types used to get a vector view
    /// @param ...args aguments to get a vector view
    /// @return vector view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto subvector( SliceArgs ... args )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 1 );
    #else
      ;
    #endif
    /// @brief Returns a matrix view
    /// @tparam ...SliceArgs argument types used to get a matrix view
    /// @param ...args aguments to get a matrix view
    /// @return matrix view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto submatrix( SliceArgs ... args )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( decltype( ::std::experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 2 );
    #else
      ;
    #endif
    /// @brief Returns a mutable view of the specified subtensor
    /// @tparam ...SliceArgs argument types used to get a tensor view
    /// @param ...args aguments to get a tensor view
    /// @return mutable tensor view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto subtensor( SliceArgs ... args );

  private:
    
    //- Data

    /// @brief non-owning view of the elements in the tensor
    underlying_span_type view_;
};

}       //- math namespace
}       //- experimental namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_FIXED_SIZE_MATRIX_ENGINE_HPP
