# del-ls

Sparse linear solver for research prototyping. 

Originally, the code is written in C++ in [DelFEM2](https://github.com/nobuyuki83/delfem2),  then it was ported to Rust. 

[The documentation generated from code](https://docs.rs/del-ls)

- [x] sparse square matrix
- [x] sparse block square matrix
- [x] conjugate gradient method
- [x] Incomplete LU preconditioner (ILU0 and ILUk)
- [x] incomplete Choleskey conjugate gradient method 