//! (block) sparse matrix class and functions

/// block sparse matrix class
/// * `nrowblk` - number of row blocks
/// * `ncolblk` - number of column blocks
pub struct Matrix<MAT> {
    pub num_blk: usize,
    pub row2idx: Vec<usize>,
    pub idx2col: Vec<usize>,
    pub idx2val: Vec<MAT>,
    pub row2val: Vec<MAT>,
}

impl<
    MAT: num_traits::Zero // set_zero
    + std::default::Default
    + std::ops::AddAssign // merge
    + Copy + std::fmt::Display>
Matrix<MAT> {
    pub fn new() -> Self {
        Matrix {
            num_blk: 0,
            row2idx: vec![0],
            idx2col: Vec::<usize>::new(),
            idx2val: Vec::<MAT>::new(),
            row2val: Vec::<MAT>::new(),
        }
    }

    pub fn clone(&self) -> Self {
        Matrix {
            num_blk: self.num_blk,
            row2idx: self.row2idx.clone(),
            idx2col: self.idx2col.clone(),
            idx2val: self.idx2val.clone(),
            row2val: self.row2val.clone(),
        }
    }

    pub fn initialize_as_square_matrix(
        &mut self,
        row2idx: &Vec<usize>,
        idx2col: &Vec<usize>) {
        self.num_blk = row2idx.len() - 1;
        self.row2idx = row2idx.clone();
        self.idx2col = idx2col.clone();
        let num_idx = self.row2idx[self.num_blk];
        assert_eq!(num_idx, idx2col.len());
        self.idx2val.resize_with(num_idx, Default::default);
        self.row2val.resize_with(self.num_blk, Default::default);
    }

    pub fn set_zero(&mut self) {
        assert_eq!(self.idx2val.len(), self.idx2col.len());
        for m in self.row2val.iter_mut() { m.set_zero() };
        for m in self.idx2val.iter_mut() { m.set_zero() };
    }

    pub fn merge(
        &mut self,
        node2row: &[usize],
        node2col: &[usize],
        emat: &[MAT],
        merge_buffer: &mut Vec<usize>) {
        assert_eq!(emat.len(), node2row.len() * node2col.len());
        merge_buffer.resize(self.num_blk, usize::MAX);
        let col2idx = merge_buffer;
        for inode in 0..node2row.len() {
            let irow = node2row[inode];
            assert!(irow < self.num_blk);
            for idx0 in self.row2idx[irow]..self.row2idx[irow + 1] {
                assert!(idx0 < self.idx2col.len());
                let icol = self.idx2col[idx0];
                col2idx[icol] = idx0;
            }
            for jnode in 0..node2col.len() {
                let jcol = node2col[jnode];
                assert!(jcol < self.num_blk);
                if irow == jcol {  // Marge Diagonal
                    self.row2val[irow] += emat[inode * node2col.len() + jnode];
                } else {  // Marge Non-Diagonal
                    assert!(col2idx[jcol] < self.idx2col.len());
                    let idx1 = col2idx[jcol];
                    assert_eq!(self.idx2col[idx1], jcol);
                    self.idx2val[idx1] += emat[inode * node2col.len() + jnode];
                }
            }
            for idx0 in self.row2idx[irow]..self.row2idx[irow + 1] {
                assert!(idx0 < self.idx2col.len());
                let jcol = self.idx2col[idx0];
                col2idx[jcol] = usize::MAX;
            }
        }
    }
}

/// generalized matrix-vector multiplication
/// where matrix is sparse (not block) matrix
/// y <- \alpha * a_mat * x_vec + \beta * y_vec
pub fn gemv_for_sparse_matrix<T>(
    y_vec: &mut Vec<T>,
    beta: T,
    alpha: T,
    a_mat: &Matrix<T>,
    x_vec: &Vec<T>)
    where T: std::ops::MulAssign // *=
    + std::ops::Mul<Output=T> // *
    + std::ops::AddAssign // +=
    + 'static + Copy // =
    + std::fmt::Display,
          f32: num_traits::AsPrimitive<T>

{
    assert_eq!(y_vec.len(), a_mat.num_blk);
    for m in y_vec.iter_mut() { *m *= beta; };
    for iblk in 0..a_mat.num_blk {
        for icrs in a_mat.row2idx[iblk]..a_mat.row2idx[iblk + 1] {
            assert!(icrs < a_mat.idx2col.len());
            let jblk0 = a_mat.idx2col[icrs];
            assert!(jblk0 < a_mat.num_blk);
            y_vec[iblk] += alpha * a_mat.idx2val[icrs] * x_vec[jblk0];
        }
        y_vec[iblk] += alpha * a_mat.row2val[iblk] * x_vec[iblk];
    }
}

#[test]
fn test_scalar() {
    let mut sparse = crate::sparse_square::Matrix::<f32>::new();
    let colind = vec![0, 2, 5, 8, 10];
    let rowptr = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    sparse.initialize_as_square_matrix(&colind, &rowptr);
    sparse.set_zero();
    {
        let emat = [1., 0., 0., 1.];
        let mut tmp_buffer = Vec::<usize>::new();
        sparse.merge(&[0, 1], &[0, 1], &emat, &mut tmp_buffer);
    }
    let nblk = colind.len() - 1;
    let mut rhs = Vec::<f32>::new();
    rhs.resize(nblk, Default::default());
    let mut lhs = Vec::<f32>::new();
    lhs.resize(nblk, Default::default());
    gemv_for_sparse_matrix(&mut lhs, 1.0, 1.0, &sparse, &rhs);
}