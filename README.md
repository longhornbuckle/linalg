Linear Algebra implementation
==========================================
This repository is intended to explore a possible implementation of a math vector, matrix, and tensor using [P0009](https://github.com/kokkos/mdspan)'s reference implementation for C++23 mdspan and incorporating C++20 concepts. Hopefully, this motivates further discussion and collaboration on the right direction to go.

TODO
----
- Additional testing - particularly of views.

Requirements
------------
- P0009 reference implementation for mdspan
- P2630 reference implementation for submdspan (same repo as P0009)

Acknowledgements
----------------
This work essentially started out by examining what P1673 might look like if it incorporated mdspan and concepts. Along the way, I started to take very different directions and pretty much started over; however, some of the terminology is kept.

Building
--------
This implementation is headers only.

Testing
-------
Running provided tests requires CMake.
- gcc-12 / C++23 / cmake 3.22.2
  - Warning free with `-Wall -pedantic -Wextra  -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-local-typedefs`

- gcc-12 / C++20 / cmake 3.22.2
  - Warning free with `-Wall -pedantic -Wextra  -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-local-typedefs`
  - No use of multi-dimensional index operator\[\](...). Use operator()(...).

- gcc-12 / C++17 / cmake 3.22.2
  - Warning free with `-Wall -pedantic -Wextra  -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-local-typedefs`
  - No use of multi-dimensional index operator\[\](...). Use operator()(...).
  - No use of concepts.
    -  Some constraints are not quite the same or are unenforceable without concepts.
  - Some destructors are no longer marked constexpr
  - \[\[likely\]\] and \[\[unlikely\]\] are not supported
  - lambda expressions cannot be used in unevaluated contexts resulting in some functions no longer having a noexcept specification.

- clang-14 / C++20 / cmake 3.22.2
  - Warning free with `-Wall -pedantic -Wextra  -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-local-typedefs`
  - No use of multi-dimensional index operator\[\](...). Use operator()(...).
  - No use of concepts. Out-of-class member function definitions with concepts unsupported until clang-16.
    -  Some constraints are not quite the same or are unenforceable without concepts.
  - No use of execution policies.

- clang-14 / C++17 / cmake 3.22.2
  - Warning free with `-Wall -pedantic -Wextra  -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-local-typedefs`
  - No use of multi-dimensional index operator\[\](...). Use operator()(...).
  - No use of concepts.
    -  Some constraints are not quite the same or are unenforceable without concepts.
  - No use of execution policies.
  - Some destructors are no longer marked constexpr
  - \[\[likely\]\] and \[\[unlikely\]\] are not supported
  - lambda expressions cannot be used in unevaluated contexts resulting in some functions no longer having a noexcept specification.

NOTE: If reviewing tests, detail::access( ... ) is used to obscure use of operator\(\)( ... ) or operator\[\]( ... ) depending on build environment. The appropriate index operator should be available for tested configurations.

Documentation
-------------
Doxygen documentation located [here](https://longhornbuckle.github.io/linalg/html/files.html).

Functionality Provided
----------------------
- fs_vector: fixed-size fixed-capacity vector
- fs_matrix: fixed-size fixed-capacity tensor
- fs_tensor: fixed-size fixed-capacity tensor
- dr_vector: dynamic-size dynamic-capacity vector
- dr_matrix: dynamic-size dynamic-capacity matrix
- dr_tensor: dynamic-size dynamic-capacity tensor
- vector_view: non-owning view of a vector into a possibly larger tensor
- matrix_view: non-owning view of a matrix into a possibly larger tensor
- tensor_view: non-owning view of a tensor into a possibly larger tensor
- vector concepts: provides concepts for vector, readable vector, writable vector, dynamic vector, fixed size vector
- matrix concepts: provides concepts for matrix, readable matrix, writable matrix, dynamic matrix, fixed size matrix
- tensor concepts: provides concepts for tensor, readable tensor, writable tensor, dynamic tensor, fixed size tensor
- basic operations: tensor negation, tensor addition, tensor subtraction, tensor scalar multiplication, tensor scalar division, matrix-matrix multiplication, matrix-vector multiplication, vector inner product, vector outer product

Critical Design Decisions
-------------------------
- fs_vector and fs_matrix inherit from fs_tensor. dr_vector and dr_matrix inherit from dr_tensor.

  *Rationale*: I felt like it was important to establish the relationship between a vector and a matrix. The tensor is a rank-N abstraction for which a base set of functionality is defined (add/subtract/negate/scalar multiply/scalar divide/...). However, some functionality (like matrix multiply) only makes sense for matrices. Other functionality (like inner or outer product) only makes sense for vectors.
- Construction or resize to a size zero extent for a tensor is not ill-formed. Use of a tensor with a size zero extent - other than size, capacity, resize, reserve, and special member functions - results in undefined behavior.

  *Rationale*: There were a couple motivating factors: I felt default construction of a dynamically allocated tensor should be supported; however, should not require immediate allocation. Additionally, resizing to zero is itself informative. Not allowing such would require additional special logic and code bloat for conveying the same information.
- Construction from constrained lambda expression

  *Rationale*: Allows for in-place construction of elements for a wide variety of use cases. Addition, subtraction, multiplication, etc. simply require the right lambda expression.
- Construction from mdspan - not from initializer lists

  *Rationale*: Construction from initializer lists could lead to easily misunderstood behavior. The client should be more explicit in how it casts data into a multidimensional space and then the tensor constructor may handle the mapping from the input mdspan to the native form of the tensor.
- If explicitly calling default construction of elements on resize can be avoided, then it is.

  *Rationale*: Efficiency. It can be awkard and difficult to read to shove all the desired element values into a constructor. Instead, the more natural syntax is to construct and then assign elements. If the element type has trivial special member functions (essentially POD), then the construction calls generally should be optimized away anyway.
- operator == is not defined

  *Rationale*: vectors and matrices are generally compared under some kind of *norm* like magnitude or largest eigenvalue. Equality compare on floating types should generally be avoided.
- iterators are not provided

  *Rationale*: Vectors, matrices, and tensors provide mathematical context and definition to an underlying multidimension data structure. Iterator support is the responsibility of the underlying data structure.
- concept usage

  *Rationale*: Concepts are used to help define appropriate implementation of relatively generic functions (like when an allocator is or is not needed). Part of this effort is to motivate more discussion on which concepts are useful and which are overly verbose.
- negate, transpose, and conjugate *do not* produce views.

  *Rationale*: Without better understanding some of the motivating use cases, this seemed like it would be confusing for non-const views and const-views really seem like expression types for unary operations that perhaps are better tackled in the context of explicitly adding expression type capabilities.
  
- No formal relationship between fixed size and dynamic size math objects

  *Rationale*: With appropriately defined concepts, there didn't seem to be a real need to do so.
 
 - Unlike P1385, vectors, matrices, and tensors do not split their behavior between engine types and operation types.
 
   *Rationale*: Overall, it felt clunky and unnecessary to do so. I got pretty far down this path and just didn't like the direction. There were a lot of functions that were disable or enabled dependent on the underlying engine type which made for a complicated interface. The methods for determining the operation type to use for performing a particular operation seemed too complicated and it felt unrealistic that anyone would delve deep enough into it to overload that behavior. As far as backwards compatibility, if the code were to eventually evolve to incorporate expression types - I would think those naturally would be convertible such that any legacy code would still work. If there was a desire to simultaneously support both instantly evaluated operations and expression types, then I think an approach similar to the std::pmr::allocators might be appropriate where different implementations exist in different namespaces (rather than being tied to the type of a more generic and complicated template class).

Critical Questions
------------------
- Is the inheritance relationship between tensors and matrices / vectors appropriate? Inheritance often is not the best choice. It will constrain future development of tensors; however, it seemed to fit naturally and reasonably approximate the mathematic relationship between tensors, matrices, and vectors. I could not think of reasonably generic tensor functions that should not work on vectors and matrices; however, I don't use higher order tensors all that much.

- What concept definitions are appropriate? I tended toward the side of verbosity for the sake of meaningful conversation.

- Which feels more natural?
  - submatrix( { start row, start column }, { end row, end column } )
  - submatrix( { start row, end row }, { start column, end column } )
  - The latter is similar to the interface used by subtensor( SliceArgs ... ) which essentially forwards to submdspan( ... ).

- Is a partially fixed size partially dynamic tensor useful? I felt like it would be complicated to implement and potentially confusing to use. There is some benefit if they are used in such a way that the final product size is now known at compile time; however, much of that benefit is negated by implementing expression types.

- Is the construction from a constrained lambda expression the right way to go? I found it *very* useful to support inplace construction. I could see additional benefit in somehow being to optionally express an iteration scheme to define the order in which the lambda expression is called. (Currently, it is called once for each possible index set in any order.)

- Should dense tensor operations somehow support parallel execution policies? I felt like such support should be reserved for an entirely different class if needed.

- Is at(...) desired? It felt like something I maybe should add for consistency with the way STL containers support index operations; however, mdspan doesn't support it. I could also see C++ taking a different approach to safe memory access that could make the function moot.

- Is there a good motivating use case for which an operation traits approach like what is done in [P1385](https://github.com/BobSteagall/wg21) is strongly desirable?

- Views neither classify as fixed-size or dynamic - they're size is not known at compile time and they have no allocator access. Should operations which return a new matix be allowed (and thus must use std allocator) or not be allowed?

In Regards To Existing Proposals
--------------------------------
- **[P2630](https://github.com/kokkos/mdspan)**: submdspan: submdspan is heavily used and this implementation does not work without it.
- **[P1385](https://github.com/BobSteagall/wg21)**: linear algebra support: In conflict / competition with. This effort really started out as what might P1385 look like with mdspan and concepts. Pretty quickly I realized a lot would change and mostly started over, but kept some of the naming convention.
- **[P1684](https://github.com/kokkos/mdspan)**: mdarray: I'm not yet sure I fully appreciate the motivating use cases. I could easily see adding explicit constructors and assignment operators which used an mdarray; however, I do not think I would implement a tensor in terms of an mdarray. I'm not sure there is a motivating use case for non-contiguous memory for a tensor. And, if there was such, one would likely want to use block implementation along those discontinuities.
- **P0478**: non-terminal parameter packs (rejected): The natural syntax for a fixed size tensor is fs_tensor<T,N1,N2,N3,...,Layout,Accessor> where Layout and Accessor have defaults. Lacking this proposal, the syntax is fs_tensor<T,Layout,Accessor,N1,N2,N3,...>. This does not allowed for desired defaults. Another approach would be to use an extents parameter (not unlike mdspan) though a syntax which simply expands on the fs_vector and fs_matrix syntax seems most natrual. It seems like absent P0478, whatever the choice - the end user will be motivated to create aliases that allow for the more natural syntax.
- **[P1673](https://github.com/kokkos/stdBLAS)**: As their proposal points out P1673 and P1385 don't collide, neither should this. More so, as P1673 has heavily integrated P0009's mdspan, I would think P1673 would dovetail nicely as a backend implementation. I've yet to critically review the proposal to figure out how best to do so.

Desired mdspan Improvements
---------------------------
- submdspan: submdpsan is a critical component used for creating vector, matrix, and tensor views.
- (Implemented here) Specialization for rank one extents: Rank one extents should support implicit conversion to and from integer types. This is needed to support more natural syntax for a vector without overriding functionality in the base tensor. Vector's size and capacity functions should return values which are assignable to integer types.
- mdspan iterators: Iterators should provide more efficient implementation than index accessing.
- submdspan should preserve compile-time knowledge of stride order: Generic functions on tensors (in particular functions which may perform on each element in any order) should be most efficient when iterating from largest to smallest stride and thus this information is useful to preserve. For example, any submdspan of an mdspan with layout_left must maintain the same stride order.
- mdspan of uninitialized data and pointer to element access: The current implementation assumes addressof( reference ) returned from operator\[\](...) points to the appropriate memory and thus can be used construct elements in place.
- mdspan to linear layout: A fixed size tensor has an array with a number of elements known at compile time. Being able to map the mdspan elements into the underlying array during the initialization of a fixed size tensor would improve performance.
- mdspan::at(...): Frankly, I don't have a strong use case for this; however, it seemed natural for a container which provides operator\[\](...) to also provide at(...) which performs additional index bound checking.
