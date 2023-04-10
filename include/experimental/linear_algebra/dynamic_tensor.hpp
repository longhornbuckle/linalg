//==================================================================================================
//  File:       dynamic_tensor.hpp
//
//  Summary:    This header defines a dynamic tensor.  In this context, dynamic means that
//              the row, column, depth, etc. extents of such objects are not known at compile-time.
//==================================================================================================
//
#ifndef LINEAR_ALGEBRA_DYNAMIC_TENSOR_HPP
#define LINEAR_ALGEBRA_DYNAMIC_TENSOR_HPP

#include <experimental/linear_algebra.hpp>

namespace std
{
namespace math
{

/// @brief Dynamic-size, dynamic-capacity tensor.
//         Implementation satisfies the following concepts:
//         concepts::dynamic_tensor
//         concepts::writable_tensor
//         ASSUMPTION 1: addressof(Access::reference) points to the element referenced
//         ASSUMPTION 2: Layout L satisfies: Given a constant and sufficient capacity, the memory layout doesn't change during resize.
/// @tparam T element_type
/// @tparam R rank
/// @tparam Alloc allocator
/// @tparam L layout defines the ordering of elements in memory
/// @tparam Access accessor policy defines how elements are accessed
template < class  T,
           size_t R,
           class  Alloc  = allocator<T>,
           class  L      = experimental::layout_right,
           class  Access = experimental::default_accessor<T> >
class dr_tensor
{
  public:
    //- Types

    /// @brief Type used to define memory layout
    using layout_type                = L;
    /// @brief Type used to define access into memory
    using accessor_type              = Access;
    /// @brief Type contained by the tensor
    using element_type               = typename accessor_type::element_type;
    /// @brief Type used for size along any dimension
    using size_type                  = size_t;
    /// @brief Type used to express size of tensor
    using extents_type               = typename detail::template extents_helper<size_type,R>::extents_type;

  private:
    //- Types

    /// @brief Type used to view the const memory within capacity
    using const_capacity_span_type   = experimental::mdspan< const element_type,
                                                           extents_type,
                                                           layout_type,
                                                           typename detail::rebind_accessor_t< accessor_type,const element_type > >;
    /// @brief Type used to view the memory within capacity
    using capacity_span_type         = experimental::mdspan< element_type, extents_type, layout_type, accessor_type >;

  public:
    //- Types

    /// @brief Type returned by const index access
    using value_type                 = remove_cv_t<element_type>;
    /// @brief Type of allocator used to get memory
    using allocator_type             = allocator_traits<Alloc>::template rebind_alloc<element_type>;
    /// @brief Type used for indexing
    using index_type                 = ptrdiff_t;
    /// @brief Type used to represent a node in the tensor
    using tuple_type                 = typename detail::template extents_helper<size_type,R>::tuple_type;
    /// @brief Type used to view memory within size
    using underlying_span_type       = decltype( detail::submdspan( declval<capacity_span_type>(), declval<tuple_type>(), declval<tuple_type>() ) );
    /// @brief Type used to const view memory within size
    using const_underlying_span_type = decltype( detail::submdspan( declval<const_capacity_span_type>(), declval<tuple_type>(), declval<tuple_type>() ) );
    /// @brief Type used to portray tensor as an N dimensional view
    using span_type                  = const_underlying_span_type;
    /// @brief Type returned by mutable index access
    using reference_type             = typename accessor_type::reference;

  private:
    //- Implementation details
    
    // Helper class for defining allocator behavior
    template < class U, class propogate_on_copy_true >
    struct Alloc_copy_helper { [[nodiscard]] static inline constexpr auto propogate( const U& u ) noexcept { return u.get_allocator(); } };
    template < class U >
    struct Alloc_copy_helper< U, false_type > { [[nodiscard]] static inline constexpr auto propogate( [[maybe_unused]] const U& u ) noexcept { return move( allocator_type() ); } };
    // Helper class for defining allocator behavior
    template < class U, class propogate_on_move_true >
    struct Alloc_move_helper { [[nodiscard]] static inline constexpr auto propogate( U&& u ) noexcept { return move( u.get_allocator() ); } };
    template < class U >
    struct Alloc_move_helper< U, false_type > { [[nodiscard]] static inline constexpr auto propogate( [[maybe_unused]] U&& u ) noexcept { return move( allocator_type() ); } };
    // Verifies lambda expression takes a set of indices and produces an output convertible to element type
    template < class Lambda, class Seq, bool > struct convertible_lambda_expression_impl : public false_type { };
    template < class Lambda,
               template < class, auto ... > class Seq,
               class IndexType,
               IndexType ... Indices >
    struct convertible_lambda_expression_impl< Lambda, Seq<IndexType,Indices...>, true >  : public
      conditional_t< is_convertible_v< decltype( declval<Lambda&&>()( Indices ... ) ), element_type >, true_type, false_type > { };
    template < class Lambda, class Seq > struct convertible_lambda_expression : public false_type { };
    template < class Lambda,
               template < class, auto ... > class Seq,
               class IndexType,
               IndexType ... Indices >
    struct convertible_lambda_expression< Lambda, Seq<IndexType,Indices...> > :
      public convertible_lambda_expression_impl< Lambda, Seq<IndexType,Indices...>, detail::has_index_operator<Lambda,IndexType,Indices...> > { };
    template < class Lambda,
               class Seq >
    static inline constexpr bool convertible_lambda_expression_v = convertible_lambda_expression<Lambda,Seq>::value;

  public:
    //- Rebind

    /// @brief Rebind defines a type for a rebinding a dynamic tensor to the new type parameters
    /// @tparam ValueType  rebound value type
    /// @tparam LayoutType rebound layout policy
    /// @tparam AccessType rebound access policy
    template < class ValueType,
               class LayoutType   = layout_type,
               class AccessorType = accessor_type >
    class rebind
    {
    private:
      using rebind_accessor_type = typename detail::rebind_accessor_t<AccessorType,ValueType>;
      using rebind_element_type  = typename rebind_accessor_type::element_type;
    public:
      using type = dr_tensor< ValueType,
                              R,
                              typename allocator_traits<allocator_type>::template rebind_alloc<rebind_element_type>,
                              LayoutType,
                              rebind_accessor_type >;
    };
    /// @brief Helper for defining rebound type
    /// @tparam ValueType  rebound value type
    /// @tparam LayoutType rebound layout policy
    /// @tparam AccessType rebound access policy
    template < class ValueType,
               class LayoutType   = layout_type,
               class AccessorType = accessor_type >
    using rebind_t = typename rebind< ValueType, LayoutType, AccessorType >::type;

    //- Destructor / Constructors / Assignments

    /// @brief Destructor
    constexpr ~dr_tensor() noexcept( is_nothrow_destructible_v<element_type> );
    /// @brief Default constructor
    constexpr dr_tensor() noexcept;
    /// @brief Move constructor
    /// @param dr_tensor to be moved
    constexpr dr_tensor( dr_tensor&& rhs )
      noexcept( typename allocator_traits<allocator_type>::propagate_on_container_move_assignment{} ||
                typename allocator_traits<allocator_type>::is_always_equal{} );
    /// @brief Copy constructor
    /// @param dr_tensor to be copied
    constexpr dr_tensor( const dr_tensor& rhs );
    // TODO: Define noexcept specification
    /// @brief Template copy constructor
    /// @tparam tensor to be copied
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::tensor_may_be_constructible< dr_tensor > T2 >
    #else
    template < class T2, typename = enable_if_t< concepts::tensor_may_be_constructible< T2, dr_tensor > > >
    #endif
    explicit constexpr dr_tensor( const T2& rhs );
    /// @brief Construct from a view
    /// @tparam An N dimensional view
    /// @param view view of tensor elements
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< dr_tensor > MDS >
    #else
    template < class MDS, typename = enable_if_t< concepts::view_may_be_constructible_to_tensor<MDS,dr_tensor> && is_default_constructible_v<allocator_type> > >
    #endif
    explicit constexpr dr_tensor( const MDS& view )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires is_default_constructible_v<allocator_type>;
    #else
      ;
    #endif
    /// @brief Attempt to allocate sufficient resources for a size tensor and construct
    /// @param s defines the length of each dimension of the tensor
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename = enable_if_t< is_default_constructible_v<allocator_type> > >
    #endif
    explicit constexpr dr_tensor( extents_type s )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires is_default_constructible_v<allocator_type>;
    #else
      ;
    #endif
    /// @brief Attempt to allocate sufficient resources for a size tensor with the input capacity and construct
    /// @param s defines the length of each dimension of the tensor
    /// @param cap defines the capacity along each of the dimensions of the tensor
    #ifndef LINALG_ENABLE_CONCEPTS
    template < typename = enable_if_t< is_default_constructible_v<allocator_type> > >
    #endif
    constexpr dr_tensor( extents_type s, extents_type cap )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires is_default_constructible_v<allocator_type>;
    #else
      ;
    #endif
    /// @brief Construct by applying lambda to every element in the tensor
    /// @tparam Lambda lambda expression with an operator()( indices ... ) defined
    /// @param s defines the length of each dimension of the tensor
    /// @param lambda lambda expression to be performed on each element
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = enable_if_t< is_default_constructible_v<allocator_type> &&
                                       convertible_lambda_expression_v< Lambda, make_integer_sequence<index_type,R> > >
    #endif
    constexpr dr_tensor( extents_type s, Lambda&& lambda )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires is_default_constructible_v<allocator_type> && convertible_lambda_expression_v< Lambda, make_integer_sequence<index_type,R> >;
    #else
      ;
    #endif
    /// @brief Construct by applying lambda to every element in the tensor
    /// @tparam Lambda lambda expression with an operator()( indices ... ) defined
    /// @param s defines the length of each dimension of the tensor
    /// @param cap defines the capacity along each of the dimensions of the tensor
    /// @param lambda lambda expression to be performed on each element
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = enable_if_t< is_default_constructible_v<allocator_type> &&
                                       convertible_lambda_expression_v< Lambda, make_integer_sequence<index_type,R> > >
    #endif
    constexpr dr_tensor( extents_type s, extents_type cap, Lambda&& lambda )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires is_default_constructible_v<allocator_type> && convertible_lambda_expression_v< Lambda, make_integer_sequence<index_type,R> >;
    #else
      ;
    #endif
    /// @brief Construct empty dimensionless tensor with an allocator
    /// @param alloc allocator to construct with
    explicit constexpr dr_tensor( const allocator_type& alloc ) noexcept;
    /// @brief Construct from a view
    /// @tparam An N dimensional view
    /// @param view view of tensor elements
    /// @param alloc allocator to construct with
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< dr_tensor > MDS >
    #else
    template < class MDS, typename = enable_if_t< concepts::view_may_be_constructible_to_tensor<MDS,dr_tensor> > >
    #endif
    constexpr dr_tensor( const MDS& view, const allocator_type& alloc );
    /// @brief Attempt to allocate sufficient resources for a size tensor and construct
    /// @param s defines the length of each dimension of the tensor
    /// @param alloc allocator used to construct with
    constexpr dr_tensor( extents_type s, const allocator_type& alloc );
    /// @brief Attempt to allocate sufficient resources for a size tensor with the input capacity and construct
    /// @param s defines the length of each dimension of the tensor
    /// @param cap defines the capacity along each of the dimensions of the tensor
    /// @param alloc allocator used to construct with
    constexpr dr_tensor( extents_type s, extents_type cap, const allocator_type& alloc );
    /// @brief Construct by applying lambda to every element in the tensor
    /// @tparam Lambda lambda expression with an operator()( indices ... ) defined
    /// @param s defines the length of each dimension of the tensor
    /// @param lambda lambda expression to be performed on each element
    /// @param alloc allocator used to construct with
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = enable_if_t< convertible_lambda_expression_v< Lambda, make_integer_sequence<index_type,R> > >
    #endif
    constexpr dr_tensor( extents_type s, Lambda&& lambda, const allocator_type& alloc )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires convertible_lambda_expression_v< Lambda, make_integer_sequence<index_type,R> >;
    #else
      ;
    #endif
    /// @brief Construct by applying lambda to every element in the tensor
    /// @tparam Lambda lambda expression with an operator()( indices ... ) defined
    /// @param s defines the length of each dimension of the tensor
    /// @param cap defines the capacity along each of the dimensions of the tensor
    /// @param lambda lambda expression to be performed on each element
    /// @param alloc allocator used to construct with
    #ifdef LINALG_ENABLE_CONCEPTS
    template < class Lambda >
    #else
    template < class Lambda,
               typename = enable_if_t< convertible_lambda_expression_v< Lambda, make_integer_sequence<index_type,R> > >
    #endif
    constexpr dr_tensor( extents_type s, extents_type cap, Lambda&& lambda, const allocator_type& alloc )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires convertible_lambda_expression_v< Lambda, make_integer_sequence<index_type,R> >;
    #else
      ;
    #endif
    /// @brief Move assignment
    /// @param  dr_tensor to be moved
    /// @return self
    constexpr dr_tensor& operator = ( dr_tensor&& rhs )
      noexcept( typename allocator_traits<allocator_type>::propagate_on_container_move_assignment() ||
                typename allocator_traits<allocator_type>::is_always_equal() );
    /// @brief Copy assignment
    /// @param  fs_tensor to be copied
    /// @return self
    constexpr dr_tensor& operator = ( const dr_tensor& rhs );
    // TODO: Define noexcept specification.
    /// @brief Template copy assignment
    /// @tparam type of tensor to be copied
    /// @param  tensor to be copied
    /// @returns self
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::tensor_may_be_constructible< dr_tensor > T2 >
    #else
    template < class T2, typename = enable_if_t< concepts::tensor_may_be_constructible< T2, dr_tensor > > >
    #endif
    constexpr dr_tensor& operator = ( const T2& rhs );
    // TODO: Define noexcept specification.
    /// @brief Construct from an N dimensional view
    /// @tparam type of view to be copied
    /// @param  view to be copied
    /// @returns self
    #ifdef LINALG_ENABLE_CONCEPTS
    template < concepts::view_may_be_constructible_to_tensor< dr_tensor > MDS >
    #else
    template < class MDS, typename = enable_if_t< concepts::view_may_be_constructible_to_tensor<MDS,dr_tensor> && is_default_constructible_v<allocator_type> > >
    #endif
    constexpr dr_tensor& operator = ( const MDS& view );

    //- Size / Capacity

    /// @brief Returns the current number of (rows,columns,depth,etc.)
    /// @return number of (rows,columns,depth,etc.)
    [[nodiscard]] constexpr extents_type size() const noexcept;
    /// @brief Returns the current capacity of (rows,columns,depth,etc.)
    /// @return capacity of (rows,columns,depth,etc.)
    [[nodiscard]] constexpr extents_type capacity() const noexcept;
    /// @brief Attempts to resize the tensor to the input extents
    /// @param new_size extents type defining the new length of each dimension of the tensor
    constexpr void resize( extents_type new_size );
    /// @brief Attempts to reserve the capacity of the tensor to the input extents
    /// @param new_size extents type defining the new capacity along each dimension of the tensor
    constexpr void reserve( extents_type new_cap );

    //- Memory access

    /// @brief Sets a new allocator. Attempts to copy existing state into new
    /// memory managed by the new allocator
    /// @param alloc new allocator to be set
    constexpr void set_allocator( const allocator_type& alloc );
    /// @brief returns the rvalue allocator being used
    /// @returns rvalue allocator being used
    [[nodiscard]] constexpr const allocator_type&& get_allocator() && noexcept;
    /// @brief returns the allocator being used
    /// @returns the allocator being used
    [[nodiscard]] constexpr const allocator_type& get_allocator() const & noexcept;

    //- Data access
    
    /// @brief returns a const N dimensional span of extents(Ds...)
    /// @returns const N dimensional span of extents(Ds...) 
    [[nodiscard]] constexpr span_type                  span() const noexcept;
    /// @brief returns a mutable N dimensional span of extents(Ds...)
    /// @returns mutable N dimensional span of extents(Ds...) 
    [[nodiscard]] constexpr underlying_span_type       underlying_span() noexcept;
    /// @brief returns a const N dimensional span of extents(Ds...)
    /// @returns const N dimensional span of extents(Ds...) 
    [[nodiscard]] constexpr const_underlying_span_type underlying_span() const noexcept;

    //- Const views

    /// @brief Returns the value at (indices...) without index bounds checking
    /// @param indices set indices representing a node in the tensor
    /// @returns value at row i, column j, depth k, etc.
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... IndexType >
    [[nodiscard]] constexpr value_type operator[]( IndexType ... indices ) const noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(IndexType) == R ) && ( is_convertible_v<IndexType,index_type> && ... );
    #endif
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... IndexType >
    [[nodiscard]] constexpr value_type operator()( IndexType ... indices ) const noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(IndexType) == R ) && ( is_convertible_v<IndexType,index_type> && ... );
    #endif
    #endif
    /// @brief Returns the value at (indices...) with index bounds checking
    /// @param indices set indices representing a node in the tensor
    /// @returns value at row i, column j, depth k, etc.
    template < class ... IndexType >
    [[nodiscard]] constexpr value_type at( IndexType ... indices ) const
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(IndexType) == R ) && ( is_convertible_v<IndexType,index_type> && ... );
    #endif
    /// @brief Returns a const vector view
    /// @tparam ...SliceArgs argument types used to get a const vector view
    /// @param ...args aguments to get a const vector view
    /// @return const vector view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto subvector( SliceArgs ... args ) const
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( decltype( experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 1 );
    #endif
    /// @brief Returns a const matrix view
    /// @tparam ...SliceArgs argument types used to get a const matrix view
    /// @param ...args aguments to get a const matrix view
    /// @return const matrix view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto submatrix( SliceArgs ... args ) const
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( decltype( experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 2 );
    #endif
    /// @brief Returns a const view of the specified subtensor
    /// @tparam ...SliceArgs argument types used to get a tensor view
    /// @param ...args aguments to get a tensor view
    /// @return const tensor view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto subtensor( SliceArgs ... args ) const;

    //- Mutable views

    /// @brief Returns a mutable value at (indices...) without index bounds checking
    /// @param indices set indices representing a node in the tensor
    /// @returns mutable value at row i, column j, depth k, etc.
    #if LINALG_USE_BRACKET_OPERATOR
    template < class ... IndexType >
    [[nodiscard]] constexpr reference_type operator[]( IndexType ... indices ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(IndexType) == R ) && ( is_convertible_v<IndexType,index_type> && ... );
    #endif
    #endif
    #if LINALG_USE_PAREN_OPERATOR
    template < class ... IndexType >
    [[nodiscard]] constexpr reference_type operator()( IndexType ... indices ) noexcept
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(IndexType) == R ) && ( is_convertible_v<IndexType,index_type> && ... );
    #endif
    #endif
    /// @brief Returns a mutable value at (indices...) with index bounds checking
    /// @param indices set indices representing a node in the tensor
    /// @returns mutable value at row i, column j, depth k, etc.
    template < class ... IndexType >
    [[nodiscard]] constexpr reference_type at( IndexType ... indices )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( sizeof...(IndexType) == R ) && ( is_convertible_v<IndexType,index_type> && ... );
    #endif
    /// @brief Returns a vector view
    /// @tparam ...SliceArgs argument types used to get a vector view
    /// @param ...args aguments to get a vector view
    /// @return mutable vector view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto subvector( SliceArgs ... args )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( decltype( experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 1 );
    #endif
    /// @brief Returns a matrix view
    /// @tparam ...SliceArgs argument types used to get a matrix view
    /// @param ...args aguments to get a matrix view
    /// @return mutable matrix view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto submatrix( SliceArgs ... args )
    #ifdef LINALG_ENABLE_CONCEPTS
      requires ( decltype( experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 2 );
    #endif
    /// @brief Returns a mutable view of the specified subtensor
    /// @tparam ...SliceArgs argument types used to get a tensor view
    /// @param ...args aguments to get a tensor view
    /// @return mutable tensor view
    template < class ... SliceArgs >
    [[nodiscard]] constexpr auto subtensor( SliceArgs ... args );

  private:
    //- Data
  
    /// @brief Allocator used for memory management
    [[no_unique_address]] allocator_type alloc_;
    /// @brief Maintains current capacity
    extents_type                         cap_;
    /// @brief Pointer to beginning of elements
    element_type*                        elems_;
    /// @brief Maintains current multidimensional view
    underlying_span_type                 view_;

    //- Implementation details
    
    // Create a new view given the current capacity and a new size
    [[nodiscard]] constexpr underlying_span_type create_view( const extents_type s ) noexcept;
    // Calls destructor on all elements and deallocates the allocator
    // If an exception is thrown, the last exception to be thrown will be re-thrown.
    constexpr void destroy_all() noexcept( is_nothrow_destructible_v<element_type> );
    // Calls destructor on all elements and deallocates the allocator
    // If an exception is thrown, the last exception to be thrown will be re-thrown.
    inline void destroy_all_except();
    // Default constructs all elements
    // If an exception is to be thrown, will first destruct elements which have been constructed and deallocate
    constexpr void construct_all() noexcept( is_nothrow_constructible_v<element_type> );
    // Default constructs all elements
    // If an exception is to be thrown, will first destruct elements which have been constructed and deallocate
    inline void construct_all_except();
    // Implementation of resize. (Needed a parameter pack of indices for implementation.)
    template < class SizeType, SizeType ... Indices >
    constexpr void resize_impl( extents_type new_size, [[maybe_unused]] integer_sequence<SizeType,Indices...> );
    // Returns an extents which is the maximum of the two inputs
    static constexpr extents_type max_extents( extents_type extents_a, extents_type extents_b ) noexcept;
    // Returns the total number of elements allocated
    [[nodiscard]] constexpr size_t linear_capacity() noexcept;

};

//-------------------------------------------------------
// Implementation of dr_tensor<T,R,Alloc,L,Access>
//-------------------------------------------------------

//- Destructor / Constructors / Assignments

template < class T, size_t R, class Alloc, class L , class Access >
constexpr dr_tensor<T,R,Alloc,L,Access>::~dr_tensor()
  noexcept( is_nothrow_destructible_v<typename dr_tensor<T,R,Alloc,L,Access>::element_type> )
{
  // If the elements pointer has been set, then destroy and deallocate
  if ( this->elems_ )
  {
    this->destroy_all();
  }
}

template < class T, size_t R, class Alloc, class L , class Access >
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor() noexcept :
  dr_tensor<T,R,Alloc,L,Access>( allocator_type() )
{
}

template < class T, size_t R, class Alloc, class L , class Access >
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor( dr_tensor&& rhs )
  noexcept( typename allocator_traits<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type>::propagate_on_container_move_assignment{}||
            typename allocator_traits<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type>::is_always_equal{} ) :
  // Default construct or move construct allocator depending on allocator_type::propagate_on_container_move_assignment
  alloc_( dr_tensor<T,R,Alloc,L,Access>::
    template Alloc_move_helper< dr_tensor<T,R,Alloc,L,Access>,
                                typename allocator_traits<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type>::propagate_on_container_move_assignment >::
      propogate( move( rhs ) ) ),
  // Move capacity extents
  cap_( move( rhs.capacity() ) ),
  // If the allocator has moved or if all allocator types are equal, then just set element pointer; otherwise, allocate new pointer
  elems_( [ &rhs, this ] () constexpr { if constexpr ( typename allocator_traits<allocator_type>::propagate_on_container_move_assignment{} ||
                                                       typename allocator_traits<allocator_type>::is_always_equal{} )
                                        { return move( rhs.elems_ ); } else
                                        { return allocator_traits<allocator_type>::allocate( this->alloc_, this->linear_capacity() ); } }() ),
  // If the allocator has moved or if all allocator types are equal, then move view; otherwise, construct new view on new elements
  view_( [ &rhs, this ] () constexpr { if constexpr ( typename allocator_traits<allocator_type>::propagate_on_container_move_assignment() ||
                                                      typename allocator_traits<allocator_type>::is_always_equal() )
                                       { return move( rhs.underlying_span() ); } else
                                       { return move( this->create_view( rhs.underlying_span().extents() ) ); } }() )
{
  // If the allocator was not moved, then elements have to be copied.
  if constexpr ( !allocator_traits<allocator_type>::propagate_on_container_move_assignment::value )
  {
    if constexpr ( is_nothrow_copy_constructible_v<element_type> )
    {
      // Copy construct all elements
      detail::copy_view( this->view_, rhs.span() );
    }
    else
    {
      try
      {
        // Copy construct all elements
        detail::copy_view( this->view_, rhs.span() );
      }
      catch ( ... )
      {
        // Deallocate
        allocator_traits<allocator_type>::deallocate( this->alloc_, this->elems_, this->linear_capacity() );
        // Rethrow
        rethrow_exception( current_exception() );
      }
    }
  }
  else
  {
    // Set the pointer in the moved tensor to null so its destruction doesn't deallocate
    rhs.elems_ = nullptr;
  }
}

template < class T, size_t R, class Alloc, class L , class Access >
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor( const dr_tensor& rhs ) :
  // Default construct or copy construct allocator depending on allocator_type::propagate_on_container_copy_assignment
  alloc_( dr_tensor<T,R,Alloc,L,Access>::
    template Alloc_copy_helper< dr_tensor<T,R,Alloc,L,Access>,
                                typename allocator_traits<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type>::propagate_on_container_copy_assignment >::
      propogate( rhs ) ),
  // Copy capacity extents
  cap_( rhs.capacity() ),
  // Allocate elements
  elems_( allocator_traits<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type>::allocate( this->alloc_, this->linear_capacity() ) ),
  // Create new view over elements
  view_( this->create_view( rhs.span().extents() ) )
{
  if constexpr ( is_nothrow_copy_constructible_v<element_type> )
  {
    // Copy construct all elements
    detail::copy_view( this->view_, rhs.span() );
  }
  else
  {
    try
    {
      // Copy construct all elements
      detail::copy_view( this->view_, rhs.span() );
    }
    catch ( ... )
    {
      // Deallocate
      allocator_traits<allocator_type>::deallocate( this->alloc_, this->elems_, this->linear_capacity() );
      // Rethrow
      rethrow_exception( current_exception() );
    }
  }
}

template < class T, size_t R, class Alloc, class L , class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_may_be_constructible< dr_tensor<T,R,Alloc,L,Access> > T2 >
#else
template < class T2, typename >
#endif
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor( const T2& rhs ) :
  // Default construct or copy construct allocator depending on allocator_type::propagate_on_container_copy_assignment
  alloc_( dr_tensor<T,R,Alloc,L,Access>::
    template Alloc_copy_helper< T2,
                                integral_constant<bool,allocator_traits<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type>::propagate_on_container_copy_assignment::value &&
                                                       concepts::dynamic_tensor<T2> > >::
      propogate( rhs ) ),
  // Copy capacity extents
  cap_( rhs.capacity() ),
  // Allocate elements
  elems_( allocator_traits<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type>::allocate( this->alloc_, this->linear_capacity() ) ),
  // Create new view over elements
  view_( this->create_view( rhs.span().extents() ) )
{
  // Copy construct all elements
  detail::copy_view( this->view_, rhs.span() );
}

template < class T, size_t R, class Alloc, class L , class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::view_may_be_constructible_to_tensor< dr_tensor<T,R,Alloc,L,Access> > MDS >
#else
template < class MDS, typename >
#endif
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor( const MDS& view )
#ifdef LINALG_ENABLE_CONCEPTS
  requires is_default_constructible_v<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type> :
#else
  :
#endif
  dr_tensor<T,R,Alloc,L,Access>( view, allocator_type() )
{
}

template < class T, size_t R, class Alloc, class L , class Access >
#ifndef LINALG_ENABLE_CONCEPTS
template < typename >
#endif
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor( extents_type s )
#ifdef LINALG_ENABLE_CONCEPTS
  requires is_default_constructible_v<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type> :
#else
  :
#endif
  dr_tensor<T,R,Alloc,L,Access>( s, allocator_type() )
{
}

template < class T, size_t R, class Alloc, class L , class Access >
#ifndef LINALG_ENABLE_CONCEPTS
template < typename >
#endif
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor( extents_type s, extents_type cap )
#ifdef LINALG_ENABLE_CONCEPTS
  requires is_default_constructible_v<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type> :
#else
  :
#endif
  dr_tensor<T,R,Alloc,L,Access>( s, cap, allocator_type() )
{
}

template < class T, size_t R, class Alloc, class L , class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < class Lambda >
#else
template < class Lambda, typename >
#endif
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor( extents_type s, Lambda&& lambda )
#ifdef LINALG_ENABLE_CONCEPTS
  requires is_default_constructible_v<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type> &&
           convertible_lambda_expression_v< Lambda, make_integer_sequence<typename dr_tensor<T,R,Alloc,L,Access>::index_type,R> > :
#else
  :
#endif
  dr_tensor<T,R,Alloc,L,Access>( s, lambda, allocator_type() )
{
}

template < class T, size_t R, class Alloc, class L , class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < class Lambda >
#else
template < class Lambda, typename >
#endif
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor( extents_type s, extents_type cap, Lambda&& lambda )
#ifdef LINALG_ENABLE_CONCEPTS
  requires is_default_constructible_v<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type> &&
           convertible_lambda_expression_v< Lambda, make_integer_sequence<typename dr_tensor<T,R,Alloc,L,Access>::index_type,R> > :
#else
  :
#endif
  dr_tensor<T,R,Alloc,L,Access>( s, cap, lambda, allocator_type() )
{
}

template < class T, size_t R, class Alloc, class L , class Access >
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor( const allocator_type& alloc ) noexcept :
  alloc_( alloc ),
  cap_( detail::template extents_helper<size_type,R>::zero() ),
  elems_(),
  view_( this->create_view( this->cap_ ) )
{
}

template < class T, size_t R, class Alloc, class L , class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::view_may_be_constructible_to_tensor< dr_tensor<T,R,Alloc,L,Access> > MDS >
#else
template < class MDS, typename >
#endif
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor( const MDS& view, const allocator_type& alloc ) :
  alloc_( alloc ),
  cap_( view.extents() ),
  elems_( allocator_traits<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type>::allocate( this->alloc_, this->linear_capacity() ) ),
  view_( this->create_view( this->cap_ ) )
{
  detail::copy_view( this->view_, view );
}

template < class T, size_t R, class Alloc, class L , class Access >
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor( extents_type s, const allocator_type& alloc ) :
  alloc_( alloc ),
  cap_( s ),
  elems_( allocator_traits<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type>::allocate( this->alloc_, this->linear_capacity() ) ),
  view_( this->create_view( this->cap_ ) )
{
  // If construct, assign, and destruct are not trivial, then initialize data
  if constexpr ( !( is_trivially_default_constructible_v<element_type> &&
                    is_trivially_copy_assignable_v<element_type> &&
                    is_trivially_destructible_v<element_type> ) )
  {
    this->construct_all();
  }
}

template < class  T, size_t R, class Alloc, class L , class Access >
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor( extents_type s, extents_type cap, const allocator_type& alloc ) :
  alloc_( alloc ),
  cap_( cap ),
  elems_( allocator_traits<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type>::allocate( this->alloc_, this->linear_capacity() ) ),
  view_( this->create_view( s ) )
{
  // TODO: Assume each dimension of s is less than cap or check through an assert or exception?

  // If construct, assign, and destruct are not trivial, then initialize data
  if constexpr ( !( is_trivially_default_constructible_v<element_type> &&
                    is_trivially_copy_assignable_v<element_type> &&
                    is_trivially_destructible_v<element_type> ) )
  {
    this->construct_all();
  }
}

template < class  T, size_t R, class Alloc, class L , class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < class Lambda >
#else
template < class Lambda, typename >
#endif
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor( extents_type s, Lambda&& lambda, const allocator_type& alloc )
#ifdef LINALG_ENABLE_CONCEPTS
  requires convertible_lambda_expression_v< Lambda, make_integer_sequence<index_type,R> > :
#else
  :
#endif
  alloc_( alloc ),
  cap_( s ),
  elems_( allocator_traits<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type>::allocate( this->alloc_, this->linear_capacity() ) ),
  view_( this->create_view( this->cap_ ) )
{
  // Construct all elements from lambda expression
  auto lambda_ctor = [this,&lambda]< class ... SizeType >( SizeType ... indices ) constexpr noexcept( is_nothrow_copy_constructible_v<element_type> )
  {
    // TODO: This requires reference returned from mdspan to be the address of the element
    ::new ( addressof( this->view_[ indices ... ] ) ) element_type( lambda( indices ... ) );
  };
  detail::apply_all( this->view_, lambda_ctor, LINALG_EXECUTION_UNSEQ );
}
  
template < class  T, size_t R, class Alloc, class L , class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < class Lambda >
#else
template < class Lambda, typename >
#endif
constexpr dr_tensor<T,R,Alloc,L,Access>::dr_tensor( extents_type s, extents_type cap, Lambda&& lambda, const allocator_type& alloc )
#ifdef LINALG_ENABLE_CONCEPTS
  requires convertible_lambda_expression_v< Lambda, make_integer_sequence<index_type,R> > :
#else
  :
#endif
  alloc_( alloc ),
  cap_( cap ),
  elems_( allocator_traits<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type>::allocate( this->alloc_, this->linear_capacity() ) ),
  view_( this->create_view( s ) )
{
  // Construct all elements from lambda expression
  auto lambda_ctor = [this,&lambda]< class ... SizeType >( SizeType ... indices ) constexpr noexcept( is_nothrow_copy_constructible_v<element_type> )
  {
    // TODO: This requires reference returned from mdspan to be the address of the element
    ::new ( addressof( this->view_[ indices ... ] ) ) element_type( lambda( indices ... ) );
  };
  detail::apply_all( this->view_, lambda_ctor, LINALG_EXECUTION_UNSEQ );
}

template < class T, size_t R, class Alloc, class L , class Access >
constexpr dr_tensor<T,R,Alloc,L,Access>& dr_tensor<T,R,Alloc,L,Access>::operator = ( dr_tensor&& rhs )
  noexcept( typename allocator_traits<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type>::propagate_on_container_move_assignment() ||
            typename allocator_traits<typename dr_tensor<T,R,Alloc,L,Access>::allocator_type>::is_always_equal() )
{
  // If the allocator is moved, then move everything
  if constexpr ( typename allocator_traits<allocator_type>::propagate_on_container_move_assignment() ||
                 typename allocator_traits<allocator_type>::is_always_equal() )
  {
    this->alloc_ = move( rhs.get_allocator() );
    this->elems_ = rhs.elems_;
    this->cap_   = move( rhs.cap_ );
    this->view_  = this->create_view( rhs.size() );
    // Set moved tensor element pointer to null so its destruction doesn't deallocate
    rhs.elems_   = nullptr;
  }
  else
  {
    if constexpr ( is_trivially_destructible_v<element_type> )
    {
      if ( this->capacity() != rhs.capacity() )
      {
        // Deallocate
        allocator_traits<allocator_type>::deallocate( this->alloc_, this->elems_, this->linear_capacity() );
        // Set new capacity
        this->cap_   = rhs.capacity();
        // Allocate to new capacity
        this->elems_ = allocator_traits<allocator_type>::allocate( this->alloc_, this->linear_capacity() );
        // Define new view
        this->view_  = this->create_view( rhs.size() );
        // Copy construct all elements
        detail::copy_view( this->view_, rhs.span() );
      }
      else
      {
        // Define new view
        this->view_ = this->create_view( rhs.size() );
        // Copy construct all elements
        detail::copy_view( this->view_, rhs.span() );
      }
    }
    else
    {
      // Destroy all elements
      this->destroy_all();
      // Set new capacity
      this->cap_   = rhs.capacity();
      // Allocate to new capacity
      this->elems_ = allocator_traits<allocator_type>::allocate( this->alloc_, this->linear_capacity() );
      // Define new view
      this->view_  = this->create_view( rhs.size() );
      // Copy construct all elements
      detail::copy_view( this->view_, rhs.span() );
    }
  }
  return *this;
}

template < class T, size_t R, class Alloc, class L , class Access >
constexpr dr_tensor<T,R,Alloc,L,Access>& dr_tensor<T,R,Alloc,L,Access>::operator = ( const dr_tensor& rhs )
{
  if constexpr ( is_trivially_destructible_v<element_type> )
  {
    if ( this->capacity() != rhs.capacity() )
    {
      // Deallocate
      allocator_traits<allocator_type>::deallocate( this->alloc_, this->elems_, this->linear_capacity() );
      // Propogate allocator
      if constexpr ( typename allocator_traits<allocator_type>::propagate_on_container_copy_assignment() )
      {
        this->alloc_ = rhs.get_allocator();
      }
      // Set new capacity
      this->cap_   = rhs.capacity();
      // Allocate to new capacity
      this->elems_ = allocator_traits<allocator_type>::allocate( this->alloc_, this->linear_capacity() );
      // Define new view
      this->view_  = this->create_view( rhs.size() );
      // Copy construct all elements
      detail::copy_view( this->view_, rhs.span() );
    }
    else
    {
      if constexpr ( typename allocator_traits<allocator_type>::propagate_on_container_copy_assignment() )
      {
        // Deallocate
        allocator_traits<allocator_type>::deallocate( this->alloc_, this->elems_, this->linear_capacity() );
        // Propogate allocator
        this->alloc_ = rhs.get_allocator();
        // Allocate
        this->elems_ = allocator_traits<allocator_type>::allocate( this->alloc_, this->linear_capacity() );
      }
      // Define new view
      this->view_ = this->create_view( rhs.size() );
      // Copy construct all elements
      detail::copy_view( this->view_, rhs.span() );
    }
  }
  else
  {
    // Destroy all
    this->destroy_all();
    // Propogate allocator
    if constexpr ( typename allocator_traits<allocator_type>::propagate_on_container_copy_assignment() )
    {
      this->alloc_ = rhs.get_allocator();
    }
    // Set new capacity
    this->cap_   = rhs.capacity();
    // Allocate to new capacity
    this->elems_ = allocator_traits<allocator_type>::allocate( this->alloc_, this->linear_capacity() );
    // Define new view
    this->view_  = this->create_view( rhs.size() );
    // Copy construct all elements
      detail::copy_view( this->view_, rhs.span() );
  }
  return *this;
}

template < class T, size_t R, class Alloc, class L , class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::tensor_may_be_constructible< dr_tensor<T,R,Alloc,L,Access> > T2 >
#else
template < class T2, typename >
#endif
constexpr dr_tensor<T,R,Alloc,L,Access>& dr_tensor<T,R,Alloc,L,Access>::operator = ( const T2& rhs )
{
  if constexpr ( is_trivially_destructible_v<element_type> )
  {
    if ( this->capacity() != rhs.capacity() )
    {
      // Deallocate
      allocator_traits<allocator_type>::deallocate( this->alloc_, this->elems_, this->linear_capacity() );
      // Propogate allocator
      if constexpr ( typename allocator_traits<allocator_type>::propagate_on_container_copy_assignment() &&
                     concepts::dynamic_tensor<T2> )
      {
        this->alloc_ = rhs.get_allocator();
      }
      // Set new capacity
      this->cap_   = rhs.capacity();
      // Allocate to new capacity
      this->elems_ = allocator_traits<allocator_type>::allocate( this->alloc_, this->linear_capacity() );
      // Define new view
      this->view_  = this->create_view( rhs.size() );
      // Copy construct all elements
      detail::copy_view( this->view_, rhs.span() );
    }
    else
    {
      if constexpr ( typename allocator_traits<allocator_type>::propagate_on_container_copy_assignment() &&
                     concepts::dynamic_tensor<T2> )
      {
        // Deallocate
        allocator_traits<allocator_type>::deallocate( this->alloc_, this->elems_, this->linear_capacity() );
        // Propogate allocator
        this->alloc_ = rhs.get_allocator();
        // Allocate
        this->elems_ = allocator_traits<allocator_type>::allocate( this->alloc_, this->linear_capacity() );
      }
      // Define new view
      this->view_ = this->create_view( rhs.size() );
      // Copy construct all elements
      detail::copy_view( this->view_, rhs.span() );
    }
  }
  else
  {
    // Destroy all
    this->destroy_all();
    // Propogate allocator
    if constexpr ( typename allocator_traits<allocator_type>::propagate_on_container_copy_assignment() &&
                   concepts::dynamic_tensor<T2> )
    {
      this->alloc_ = rhs.get_allocator();
    }
    // Set new capacity
    this->cap_   = rhs.capacity();
    // Allocate to new capacity
    this->elems_ = allocator_traits<allocator_type>::allocate( this->alloc_, this->linear_capacity() );
    // Define new view
    this->view_  = this->create_view( rhs.size() );
    // Copy construct all elements
    detail::copy_view( this->view_, rhs.span() );
  }
  return *this;
}

template < class T, size_t R, class Alloc, class L , class Access >
#ifdef LINALG_ENABLE_CONCEPTS
template < concepts::view_may_be_constructible_to_tensor< dr_tensor<T,R,Alloc,L,Access> > MDS >
#else
template < class MDS, typename >
#endif
constexpr dr_tensor<T,R,Alloc,L,Access>& dr_tensor<T,R,Alloc,L,Access>::operator = ( const MDS& view )
{
  // If sizes are the same, then assign
  if ( this->view_.extents() == view.extents() )
  {
    static_cast<void>( detail::assign_view( this->view_, view ) );
  }
  else
  {
    // If capacity is large enough
    if ( detail::sufficient_extents( this->cap_, view.extents() ) )
    {
      // If elements are trivially constructible and assignable, then just resize view and assign
      if constexpr ( is_trivially_constructible_v<element_type> && is_trivially_assignable_v<element_type,typename MDS::reference> )
      {
        this->view_ = this->create_view( view.extents() );
        static_cast<void>( detail::assign_view( this->view_, view ) );
      }
      else
      {
        // Otherwise, destruct, build new view, and construct
        if constexpr ( is_trivially_destructible_v<element_type> )
        {
          this->destroy_all();
        }
        this->view_ = this->create_view( view.extents() );
        detail::copy_view( this->view_, view );
      }
    }
    else
    {
      // Destroy if needed
      if constexpr ( is_trivially_destructible_v<element_type> )
      {
        this->destroy_all();
      }
      // Deallocate
      allocator_traits<allocator_type>::deallocate( this->alloc_, this->elems_, this->linear_capacity() );
      // Set new capacity
      this->cap_   = view.extents();
      // Allocate
      this->elems_ = allocator_traits<allocator_type>::allocate( this->alloc_, this->linear_capacity() );
      // Construct new view
      this->view_  = this->create_view( view.extents() );
      // Construct
      detail::copy_view( this->view_, view );
    }
  }
  return *this;
}

//- Size / Capacity

template < class T, size_t R, class Alloc, class L , class Access >
[[nodiscard]] constexpr typename dr_tensor<T,R,Alloc,L,Access>::extents_type
dr_tensor<T,R,Alloc,L,Access>::size() const noexcept
{
  return this->view_.extents();
}

template < class T, size_t R, class Alloc, class L , class Access >
[[nodiscard]] constexpr typename dr_tensor<T,R,Alloc,L,Access>::extents_type
dr_tensor<T,R,Alloc,L,Access>::capacity() const noexcept
{
  return this->cap_;
}

template < class T, size_t R, class Alloc, class L , class Access >
constexpr void dr_tensor<T,R,Alloc,L,Access>::resize( extents_type new_size )
{
  // Check if the memory layout must change
  if ( detail::sufficient_extents( this->cap_, new_size ) )
  {
    this->resize_impl( new_size, make_integer_sequence<index_type,extents_type::rank()>() );
  }
  else
  {
    // Copy current state
    dr_tensor clone = move( *this );
    // Set to new size
    *this = dr_tensor( new_size, max_extents( new_size, this->capacity() ), this->get_allocator() );
    // Copy view
    detail::assign_view( this->view_, clone.underlying_span() );
  }
}

template < class T, size_t R, class Alloc, class L , class Access >
constexpr void dr_tensor<T,R,Alloc,L,Access>::reserve( extents_type new_cap )
{
  // Only expand if capacity is not currently sufficient
  if ( detail::sufficient_extents( this->cap_, new_cap ) )
  {
    // Copy current state
    dr_tensor clone = move( *this );
    // Set to new size
    *this = dr_tensor( this->size(), max_extents( new_cap, this->capacity() ), this->get_allocator() );
    // Copy view
    detail::assign_view( this->view_, clone.underlying_span() );
  }
}

template < class T, size_t R, class Alloc, class L , class Access >
constexpr void dr_tensor<T,R,Alloc,L,Access>::set_allocator( const allocator_type& alloc )
{
  // Create new eninge with current size and capacity and the new allocator
  dr_tensor new_tensor( this->size(), this->capacity(), alloc );
  // Assign current elements into new tensor
  new_tensor = this->view();
  // Move new tensor into this tensor
  *this = move( new_tensor );
}

template < class T, size_t R, class Alloc, class L , class Access >
[[nodiscard]] constexpr const typename dr_tensor<T,R,Alloc,L,Access>::allocator_type&&
dr_tensor<T,R,Alloc,L,Access>::get_allocator() && noexcept
{
  return move( this->alloc_ );
}

template < class T, size_t R, class Alloc, class L , class Access >
[[nodiscard]] constexpr const typename dr_tensor<T,R,Alloc,L,Access>::allocator_type&
dr_tensor<T,R,Alloc,L,Access>::get_allocator() const & noexcept
{
  return this->alloc_;
}

//- Const views

#if LINALG_USE_BRACKET_OPERATOR
template < class T, size_t R, class Alloc, class L , class Access >
template < class ... IndexType >
[[nodiscard]] constexpr dr_tensor<T,R,Alloc,L,Access>::value_type
dr_tensor<T,R,Alloc,L,Access>::operator[]( IndexType ... indices ) const noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(IndexType) == R ) && ( is_convertible_v<IndexType,typename dr_tensor<T,R,Alloc,L,Access>::index_type> && ... )
#endif
{
  return this->underlying_span()[ indices ... ];
}
#endif

#if LINALG_USE_PAREN_OPERATOR
template < class T, size_t R, class Alloc, class L , class Access >
template < class ... IndexType >
[[nodiscard]] constexpr dr_tensor<T,R,Alloc,L,Access>::value_type
dr_tensor<T,R,Alloc,L,Access>::operator()( IndexType ... indices ) const noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(IndexType) == R ) && ( is_convertible_v<IndexType,typename dr_tensor<T,R,Alloc,L,Access>::index_type> && ... )
#endif
{
  return this->underlying_span()( indices ... );
}
#endif

template < class T, size_t R, class Alloc, class L , class Access >
template < class ... IndexType >
[[nodiscard]] constexpr dr_tensor<T,R,Alloc,L,Access>::value_type
dr_tensor<T,R,Alloc,L,Access>::at( IndexType ... indices ) const
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(IndexType) == R ) && ( is_convertible_v<IndexType,typename dr_tensor<T,R,Alloc,L,Access>::index_type> && ... )
#endif
{
  return detail::access( this->underlying_span(), indices ... );
}

template < class T, size_t R, class Alloc, class L , class Access >
template < class ... SliceArgs >
[[nodiscard]] constexpr auto dr_tensor<T,R,Alloc,L,Access>::subvector( SliceArgs ... args ) const
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 1 )
#endif
{
  using subspan_type = decltype( experimental::submdspan( this->underlying_span(), args ... ) );
  return vector_view<subspan_type>( experimental::submdspan( this->underlying_span(), args ... ) );
}

template < class T, size_t R, class Alloc, class L , class Access >
template < class ... SliceArgs >
[[nodiscard]] constexpr auto dr_tensor<T,R,Alloc,L,Access>::submatrix( SliceArgs ... args ) const
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 2 )
#endif
{
  using subspan_type = decltype( experimental::submdspan( this->underlying_span(), args ... ) );
  return matrix_view<subspan_type>( experimental::submdspan( this->underlying_span(), args ... ) );
}

template < class T, size_t R, class Alloc, class L , class Access >
template < class ... SliceArgs >
[[nodiscard]] constexpr auto dr_tensor<T,R,Alloc,L,Access>::subtensor( SliceArgs ... args ) const
{
  using subspan_type = decltype( experimental::submdspan( this->underlying_span(), args ... ) );
  return tensor_view<subspan_type>( experimental::submdspan( this->underlying_span(), args ... ) );
}

//- Mutable views

#if LINALG_USE_BRACKET_OPERATOR
template < class T, size_t R, class Alloc, class L , class Access >
template < class ... IndexType >
[[nodiscard]] constexpr dr_tensor<T,R,Alloc,L,Access>::reference_type
dr_tensor<T,R,Alloc,L,Access>::operator[]( IndexType ... indices ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(IndexType) == R ) && ( is_convertible_v<IndexType,typename dr_tensor<T,R,Alloc,L,Access>::index_type> && ... )
#endif
{
  return this->underlying_span()[ indices ... ];
}
#endif

#if LINALG_USE_PAREN_OPERATOR
template < class T, size_t R, class Alloc, class L , class Access >
template < class ... IndexType >
[[nodiscard]] constexpr dr_tensor<T,R,Alloc,L,Access>::reference_type
dr_tensor<T,R,Alloc,L,Access>::operator()( IndexType ... indices ) noexcept
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(IndexType) == R ) && ( is_convertible_v<IndexType,typename dr_tensor<T,R,Alloc,L,Access>::index_type> && ... )
#endif
{
  return this->underlying_span()( indices ... );
}
#endif

template < class T, size_t R, class Alloc, class L , class Access >
template < class ... IndexType >
[[nodiscard]] constexpr dr_tensor<T,R,Alloc,L,Access>::reference_type
dr_tensor<T,R,Alloc,L,Access>::at( IndexType ... indices )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( sizeof...(IndexType) == R ) && ( is_convertible_v<IndexType,typename dr_tensor<T,R,Alloc,L,Access>::index_type> && ... )
#endif
{
  return detail::access( this->underlying_span(), indices ... );
}

template < class T, size_t R, class Alloc, class L , class Access >
template < class ... SliceArgs >
[[nodiscard]] constexpr auto dr_tensor<T,R,Alloc,L,Access>::subvector( SliceArgs ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 1 )
#endif
{
  using subspan_type = decltype( experimental::submdspan( this->underlying_span(), args ... ) );
  return vector_view<subspan_type>( experimental::submdspan( this->underlying_span(), args ... ) );
}

template < class T, size_t R, class Alloc, class L , class Access >
template < class ... SliceArgs >
[[nodiscard]] constexpr auto dr_tensor<T,R,Alloc,L,Access>::submatrix( SliceArgs ... args )
#ifdef LINALG_ENABLE_CONCEPTS
  requires ( decltype( experimental::submdspan( this->underlying_span(), args ... ) )::rank() == 2 )
#endif
{
  using subspan_type = decltype( experimental::submdspan( this->underlying_span(), args ... ) );
  return matrix_view<subspan_type>( experimental::submdspan( this->underlying_span(), args ... ) );
}

template < class T, size_t R, class Alloc, class L , class Access >
template < class ... SliceArgs >
[[nodiscard]] constexpr auto dr_tensor<T,R,Alloc,L,Access>::subtensor( SliceArgs ... args )
{
  using subspan_type = decltype( experimental::submdspan( this->underlying_span(), args ... ) );
  return tensor_view<subspan_type>( experimental::submdspan( this->underlying_span(), args ... ) );
}

//- Data access

template < class T, size_t R, class Alloc, class L , class Access >
[[nodiscard]] constexpr typename dr_tensor<T,R,Alloc,L,Access>::span_type
dr_tensor<T,R,Alloc,L,Access>::span() const noexcept
{
  return this->underlying_span();
}

template < class T, size_t R, class Alloc, class L , class Access >
[[nodiscard]] constexpr typename dr_tensor<T,R,Alloc,L,Access>::underlying_span_type
dr_tensor<T,R,Alloc,L,Access>::underlying_span() noexcept
{
  return this->view_;
}

template < class T, size_t R, class Alloc, class L , class Access >
[[nodiscard]] constexpr typename dr_tensor<T,R,Alloc,L,Access>::const_underlying_span_type
dr_tensor<T,R,Alloc,L,Access>::underlying_span() const noexcept
{
  return this->view_;
}

template < class T, size_t R, class Alloc, class L , class Access >
[[nodiscard]] constexpr typename dr_tensor<T,R,Alloc,L,Access>::underlying_span_type dr_tensor<T,R,Alloc,L,Access>::
create_view( const extents_type s ) noexcept
{
  return detail::submdspan( capacity_span_type( this->elems_, this->cap_ ),
                            detail::template extents_helper<size_type,R>::zero_tuple(),
                            detail::template extents_helper<size_type,R>::to_tuple( s ) );
}

template < class T, size_t R, class Alloc, class L , class Access >
constexpr void dr_tensor<T,R,Alloc,L,Access>::destroy_all()
  noexcept( is_nothrow_destructible_v<typename dr_tensor<T,R,Alloc,L,Access>::element_type> )
{
  // Is the destructor non-trivial?
  if constexpr ( is_trivially_destructible_v<element_type> )
  {
    // Deallocate
    allocator_traits<allocator_type>::deallocate( this->alloc_,
                                                  this->elems_,
                                                  this->linear_capacity() );
  }
  else
  {
    if constexpr ( is_nothrow_destructible_v<element_type> )
    {
      for_each( LINALG_EXECUTION_UNSEQ,
                this->elems_,
                this->elems_ + this->view_.size(),
                []( const element_type& elem ) constexpr noexcept { elem.~element_type(); } );
      // Deallocate
      allocator_traits<allocator_type>::deallocate( this->alloc_, this->elems_, this->linear_capacity() );
    }
    else
    {
      this->destroy_all_except();
    }
  }
}

template < class T, size_t R, class Alloc, class L , class Access >
inline void dr_tensor<T,R,Alloc,L,Access>::destroy_all_except()
{
  // Cache the last exception to be thrown
  exception_ptr eptr;
  // Attempt to destruct
  for_each( LINALG_EXECUTION_UNSEQ,
            this->elems_,
            this->elems_ + this->view_.size(),
            [this,&eptr]( const element_type& elem ) constexpr { try { this->elem_.~element_type(); } catch ( ... ) { eptr = current_exception(); } } );
  // Deallocate
  allocator_traits<allocator_type>::deallocate( this->alloc_, this->elems_, this->linear_capacity() );
  // If exceptions were thrown, rethrow the last
  if ( eptr ) [[unlikely]]
  {
    rethrow_exception( eptr );
  }
}

template < class T, size_t R, class Alloc, class L , class Access >
constexpr void dr_tensor<T,R,Alloc,L,Access>::construct_all()
  noexcept( is_nothrow_constructible_v<typename dr_tensor<T,R,Alloc,L,Access>::element_type> )
{
  if constexpr ( is_nothrow_constructible_v<element_type> )
  {
    for_each( LINALG_EXECUTION_UNSEQ,
              this->elems_,
              this->elems_ + this->view_.size(),
              []( auto elem ) constexpr noexcept { ::new ( &elem ) element_type; } );
  }
  else
  {
    this->construct_all_except();
  }
}

template < class T, size_t R, class Alloc, class L , class Access >
inline void dr_tensor<T,R,Alloc,L,Access>::construct_all_except()
{
  // Cache the last exception to be thrown
  exception_ptr eptr;
  // If the elements are trivially destructible, then construction can still be unsequential
  if constexpr ( is_trivially_destructible_v<element_type> )
  {
    // Attempt to construct
    for_each( LINALG_EXECUTION_UNSEQ,
              this->elems_,
              this->elems_ + this->view_.size(),
              [&eptr]( auto elem ) constexpr { try { ::new (&elem) element_type; } catch ( ... ) { eptr = current_exception(); } } );
    // If exceptions were thrown, rethrow the last
    if ( eptr )
    {
      // Deallocate
      allocator_traits<allocator_type>::deallocate( this->alloc_, this->elems_, this->linear_capacity() );
      // Rethrow
      rethrow_exception( eptr );
    }
  }
  else
  {
    // Cache pointer to constructor which threw exception
    element_type* elem_except_ptr;
    // Attempt to construct
    for_each( execution::seq,
              this->elems_,
              this->elems_ + this->view_.size(),
              [&eptr,&elem_except_ptr]( auto elem ) constexpr { try { ::new (&elem) element_type(); } catch ( ... ) { elem_except_ptr = &elem; eptr = current_exception(); } } );
    // If exceptions were thrown, destroy all which have already been constructed, then rethrow the last
    if ( eptr ) [[unlikely]]
    {
      // Attempt to destroy constructed elements.
      // If destruction also throws an exception, then just terminate.
      for_each( LINALG_EXECUTION_UNSEQ,
                this->elems_,
                elem_except_ptr,
                []( auto elem ) constexpr noexcept( is_nothrow_destructible_v<element_type> ){ elem.~element_type(); } );
      // Deallocate
      allocator_traits<allocator_type>::deallocate( this->alloc_, this->elems_, this->linear_capacity() );
      rethrow_exception( eptr );
    }
  }
}

template < class T, size_t R, class Alloc, class L , class Access >
template < class SizeType, SizeType ... Indices >
constexpr void
dr_tensor<T,R,Alloc,L,Access>::resize_impl( extents_type new_size,
                                            [[maybe_unused]] integer_sequence<SizeType,Indices...> )
{
  // If not trivially destructible, then elements descoped from resize must be deleted
  // and elements added to scope must be default constructed
  if constexpr ( !is_trivially_destructible_v<element_type> )
  {
    // Create subview of elements to be destroyed
    auto destroy_extent = [this,new_size]( SizeType index ) constexpr noexcept
    {
      return this->size().extent(index) > new_size(index) ?
               tuple( new_size.extent(index), this->size().extent(index) ) :
               tuple( this->size().extent(index), this->size().extent(index) );
    };
    auto destroy_subview = experimental::submdspan( this->underlying_span(),
                                                    destroy_extent(Indices) ...  );
    // Define destructor lambda
    auto destructor = [this]< class ... IndexType >( IndexType ... indices ) constexpr noexcept( is_nothrow_destructible_v<element_type> )
      { this->view_[ indices ... ].~element_type(); };
    // Destroy
    detail::apply_all( destroy_subview, destructor, LINALG_EXECUTION_UNSEQ );
    // Create subview of elements to be constructed
    auto construct_extent = [this,new_size]( SizeType index ) constexpr noexcept
    {
      return this->size().extent(index) < new_size(index) ?
               tuple( this->size().extent(index), new_size.extent(index) ) :
               tuple( this->size().extent(index), this->size().extent(index) );
    };
    auto construct_subview = experimental::submdspan( this->underlying_span(),
                                                      construct_extent(Indices) ...  );
    // Define constructor lambda
    auto constructor = [this]< class ... IndexType >( IndexType ... indices ) constexpr noexcept( is_nothrow_default_constructible_v<element_type> )
      { ::new ( addressof( this->view( indices ... ) ) ) element_type(); };
    // Construct
    detail::apply_all( construct_subview, constructor, LINALG_EXECUTION_UNSEQ );
  }
  // Create a new view
  this->view_ = this->create_view( new_size );
}

template < class T, size_t R, class Alloc, class L , class Access >
constexpr typename dr_tensor<T,R,Alloc,L,Access>::extents_type
dr_tensor<T,R,Alloc,L,Access>::max_extents( extents_type extents_a, extents_type extents_b ) noexcept
{
  // Construct array to contain max
  array<index_type,extents_type::rank()> max_extents;
  // Iterate over each dimension and set max
  for_each( LINALG_EXECUTION_UNSEQ,
            detail::faux_index_iterator<index_type>(0),
            detail::faux_index_iterator<index_type>( extents_type::rank() ),
            [&max_extents,&extents_a,&extents_b] ( index_type index ) constexpr noexcept
            {
              max_extents[index] = ( extents_a.extent(index) > extents_b.extent(index) ) ? extents_a.extent(index) : extents_b.extent(index);
            } );
  // Return max
  return extents_type( max_extents );
}

template < class T, size_t R, class Alloc, class L , class Access >
[[nodiscard]] constexpr size_t dr_tensor<T,R,Alloc,L,Access>::linear_capacity() noexcept
{
  return detail::template extents_helper<size_type,R>::size( this->cap_ );
}

}       //- math namespace
}       //- std namespace
#endif  //- LINEAR_ALGEBRA_DYNAMIC_TENSOR_ENGINE_HPP
