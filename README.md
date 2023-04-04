Linear Algebra implementation
==========================================
This repository is intended to explore a possible implementation of a math vector, matrix, and tensor using [P0009](https://github.com/kokkos/mdspan)'s reference implementation for C++23 mdspan and incorporating C++20 concepts and constraints.

In Work
-------
- Working on clang conformant implementation.

Requirements
------------
- C++23 Support for multidimensional subscript operator
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
  *Rationale* Concepts are used to help define appropriate implementation of relatively generic functions (like when an allocator is or is not needed). Part of this effort is to motivate more discussion on which concepts are useful and which are overly verbose.

In Regards To Existing Proposals
--------------------------------
- P2630: submdspan: submdspan is heavily used and this implementation does not work without it.
- P1385: linear algebra support: In conflict / competition with. This effort really started out as what might P1385 look like with mdspan and concepts. Pretty quickly I realized a lot would change and mostly started over, but kept some of the naming convention.
- P1684: mdarray: I'm not yet sure I fully appreciate the motivating use cases. I could easily see adding explicit constructors and assignment operators which used an mdarray; however, I do not think I would implement a tensor in terms of an mdarray. I'm not sure there is a motivating use case for non-contiguous memory for a tensor. And, if there was such, one would likely want to use block implementation along those discontinuities.
