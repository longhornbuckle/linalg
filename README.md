Linear Algebra implementation
==========================================
This repository is intended to explore a possible implementation of a math vector, matrix, and tensor using [P0009](https://github.com/kokkos/mdspan)'s reference implementation for C++23 mdspan and incorporating C++20 concepts and constraints.

In Work
-------
- Working on clang conformant implementation.

TODO
----
- Backport to C++20 and C++17
- Integrate P1673 where appropriate

Requirements
------------
- C++23 support for multidimensional subscript operator
- P0009 reference implementation for mdspan
- P2630 reference implementation for submdspan (same repo as P0009)

Capabilities
------------
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
- size zero extent for a tensor is not ill-formed. Use of a tensor with a size 0 extent - other than size, capacity, resize, reserve, and special member functions is undefined behavior.

  *Rationale*: There were a couple motivating factors: I felt default construction of a dynamically allocated tensor should be supported; however, should    not require immediate allocation. Additionally, resizing to zero is itself informative. Not allowing such would require additional special logic and code bloat for conveying the same information.
- Construction from constrained lambda expression

  *Rationale*: Allows for in-place construction of elements for a wide variety of use cases. Addition, subtraction, multiplication, etc. simply require the right lambda expression.
- Construction from mdspan - not from initializer lists

  *Rationale*: Construction from initializer lists could lead to easily misunderstood behavior. The client should be more explicit in how it casts data into a multidimensional space and then the tensor constructor may handle the mapping from the input mdspan to the native form of the tensor.
- If explicitly calling default construction of elements on resize can be avoided, then it is.

  *Rationale*: Efficiency. It can be awkard and difficult to read to shove all the desired element values into a constructor. Instead, the more natural syntax is to construct and then assign elements.
- operator == is not defined

  *Rationale*: vectors and matrices are generally compared under some kind of *norm* like magnitude or largest eigenvalue. Equality compare on floating types should generally be avoided.
- iterators are not provided

  *Rationale*: Vectors, matrices, and tensors provide mathematical context and definition to an underlying multidimension data structure. Iterator support is the responsibility of the underlying data structure.
- concept usage

  *Rationale*: Concepts are used to help define appropriate implementation of relatively generic functions (like when an allocator is or is not needed). Part of this effort is to motivate more discussion on which concepts are useful and which are overly verbose.
- No formal relationship between fixed size and dynamic size math objects

  *Rationale*: With appropriately defined concepts, there didn't seem to be a real need to do so.
 
 - Unlike P1385, vectors, matrices, and tensors do not split their behavior between engine types and operation types.
 
   *Rationale*: Overall, it felt clunky and unnecessary to do so. I got pretty far down this path and just didn't like the direction. There were a lot of functions that were disable or enabled dependent on the underlying engine type which made for a complicated interface. The methods for determining the operation type to use for performing a particular operation seemed too complicated and it felt unrealistic that anyone would delve deep enough into it to overload that behavior. As far as backwards compatibility, if the code were to eventually evolve to incorporate expression types - I would think those naturally would be convertible such that any legacy code would still work. If there was a desire to simultaneously support both instantly evaluated operations and expression types, then I think an approach similar to the std::pmr::allocators might be appropriate where different implementations exist in different namespaces (rather than being tied to the type of a more generic and complicated template class).

In Regards To Existing Proposals
--------------------------------
- P2630: submdspan: submdspan is heavily used and this implementation does not work without it.
- P1385: linear algebra support: In conflict / competition with. This effort really started out as what might P1385 look like with mdspan and concepts. Pretty quickly I realized a lot would change and mostly started over, but kept some of the naming convention.
- P1684: mdarray: I'm not yet sure I fully appreciate the motivating use cases. I could easily see adding explicit constructors and assignment operators which used an mdarray; however, I do not think I would implement a tensor in terms of an mdarray. I'm not sure there is a motivating use case for non-contiguous memory for a tensor. And, if there was such, one would likely want to use block implementation along those discontinuities.
- P0478: non-terminal parameter packs (rejected): The natural syntax for a fixed size tensor is fs_tensor<T,N1,N2,N3,...,Layout,Accessor> where Layout and Accessor have defaults. Lacking this proposal, the syntax is fs_tensor<T,Layout,Accessor,N1,N2,N3,...>. This does not allowed for desired defaults. Another approach would be to use an extents parameter (not unlike mdspan) though a syntax which simply expands on the fs_vector and fs_matrix syntax seems most natrual. It seems like absent P0478, whatever the choice - the end user will be motivated to create aliases that allow for the more natural syntax.
- P1673: As their proposal points out P1673 and P1385 don't collide, neither should this. More so, as P1673 has heavily integrated P0009's mdspan, I would think P1673 would dovetail nicely as a backend implementation. I've yet to critically review the proposal to figure out how best to do so.

Desired mdspan Improvements
---------------------------
- submdspan: submdpsan is a critical component used for creating vector, matrix, and tensor views.
- (Implemented here) Specialization for rank one extents: Rank one extents should support implicit conversion to and from integer types. This is needed to support more natural syntax for a vector without overriding functionality in the base tensor. Vector's size and capacity functions should return values which are assignable to integer types.
- mdspan iterators: Iterators should provide more efficient implementation than index accessing.
- submdspan should preserve compile-time knowledge of stride order: Generic functions on tensors (in particular functions which may perform on each element in any order) should be most efficient when iterating from largest to smallest stride and thus this information is useful to preserve. For example, any submdspan of an mdspan with layout_left must maintain the same stride order.
- mdspan of uninitialized data and pointer to element access: The current implementation assumes the reference type returned from operator[](...) points to the appropriate memory and thus can be used construct elements in place.
- mdspan to linear layout: A fixed size tensor has an array with a number of elements known at compile time. Being able to map the mdspan elements into the underlying array during the initialization of a fixed size tensor would improve performance.
- mdspan::at(...): Frankly, I don't have a strong use case for this; however, it seemed natural for a container which provides operator[](...) to also provide at(...) which performs additional index bound checking.
