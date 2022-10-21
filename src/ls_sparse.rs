pub struct BlockSparseMatrix<MAT> {
    nrowblk: usize,
    ncolblk: usize,
    col_ind: Vec<usize>,
    row_ptr: Vec<usize>,
    val_crs: Vec<MAT>,
    pub val_dia: Vec<MAT>,
}

impl<
    MAT: num_traits::Zero // set_zero
    + std::default::Default
    + std::ops::AddAssign // merge
    + Copy + std::fmt::Display>
BlockSparseMatrix<MAT> {
    pub fn new() -> Self {
        BlockSparseMatrix {
            ncolblk: 0,
            nrowblk: 0,
            col_ind: vec![0],
            row_ptr: Vec::<usize>::new(),
            val_crs: Vec::<MAT>::new(),
            val_dia: Vec::<MAT>::new(),
        }
    }

    pub fn initialize_as_square_matrix(
        &mut self,
        colind: &Vec<usize>,
        rowptr: &Vec<usize>) {
        self.nrowblk = colind.len() - 1;
        self.ncolblk = self.nrowblk;
        self.col_ind = colind.clone();
        self.row_ptr = rowptr.clone();
        let ncrs = self.col_ind[self.nrowblk];
        assert!(ncrs == rowptr.len());
        self.val_crs.resize_with(ncrs, Default::default);
        self.val_dia.resize_with(self.nrowblk, Default::default);
    }

    pub fn set_zero(&mut self) {
        assert!(self.val_crs.len() == self.row_ptr.len());
        for m in self.val_dia.iter_mut() { m.set_zero() };
        for m in self.val_crs.iter_mut() { m.set_zero() };
    }

    pub fn merge(
        &mut self,
        rowblk_idxs: &[usize],
        colblk_idxs: &[usize],
        emat: &[MAT],
        merge_buffer: &mut Vec<usize>) {
        assert!(emat.len() == rowblk_idxs.len() * colblk_idxs.len());
        merge_buffer.resize(self.ncolblk, usize::MAX);
        for irow in 0..rowblk_idxs.len() {
            let iblk1 = rowblk_idxs[irow];
            assert!(iblk1 < self.nrowblk);
            for jpsup in self.col_ind[iblk1]..self.col_ind[iblk1 + 1] {
                assert!(jpsup < self.row_ptr.len());
                let jblk1 = self.row_ptr[jpsup];
                merge_buffer[jblk1] = jpsup;
            }
            for jcol in 0..colblk_idxs.len() {
                let jblk1 = colblk_idxs[jcol];
                assert!(jblk1 < self.ncolblk);
                if iblk1 == jblk1 {  // Marge Diagonal
                    self.val_dia[iblk1] += emat[irow * colblk_idxs.len() + jcol];
                } else {  // Marge Non-Diagonal
                    assert!(merge_buffer[jblk1] < self.row_ptr.len());
                    let jpsup1 = merge_buffer[jblk1];
                    assert!(self.row_ptr[jpsup1] == jblk1);
                    self.val_crs[jpsup1] += emat[irow * colblk_idxs.len() + jcol];
                }
            }
            for jpsup in self.col_ind[iblk1]..self.col_ind[iblk1 + 1] {
                assert!(jpsup < self.row_ptr.len());
                let jblk1 = self.row_ptr[jpsup];
                merge_buffer[jblk1] = usize::MAX;
            }
        }
    }
}

pub fn gemm_for_block_sparse_matrix_nalgebra<
    T: nalgebra::RealField + Copy,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    SVECM: nalgebra::StorageMut<T, R, C>,
    SVEC: nalgebra::Storage<T, R, C>,
    SMAT: nalgebra::Storage<T, R, R>>(
    lhs: &mut Vec<nalgebra::Matrix<T, R, C, SVECM>>,
    beta: T,
    alpha: T,
    mat: &BlockSparseMatrix<nalgebra::Matrix<T, R, R, SMAT>>,
    rhs: &Vec<nalgebra::Matrix<T, R, C, SVEC>>)
{
    assert!(lhs.len() == mat.nrowblk);
    for m in lhs.iter_mut() { (*m).scale_mut(beta); };
    let fone: T = T::from_f32(1.0).unwrap();
    for iblk in 0..mat.nrowblk {
        for icrs in mat.col_ind[iblk]..mat.col_ind[iblk + 1] {
            assert!(icrs < mat.row_ptr.len());
            let jblk0 = mat.row_ptr[icrs];
            assert!(jblk0 < mat.ncolblk);
            lhs[iblk].gemm(alpha, &mat.val_crs[icrs], &rhs[jblk0], fone); // SIMD?
        }
        lhs[iblk].gemm(alpha, &mat.val_dia[iblk], &rhs[iblk], fone);
    }
}

pub fn gemm_for_sparse_matrix<T>(
    lhs: &mut Vec<T>,
    beta: T,
    alpha: T,
    mat: &BlockSparseMatrix<T>,
    rhs: &Vec<T>)
    where T: std::ops::MulAssign // *=
    + std::ops::Mul<Output=T> // *
    + std::ops::AddAssign // +=
    + 'static + Copy // =
    + std::fmt::Display,
          f32: num_traits::AsPrimitive<T>

{
    assert!(lhs.len() == mat.nrowblk);
    for m in lhs.iter_mut() { *m *= beta; };
    for iblk in 0..mat.nrowblk {
        for icrs in mat.col_ind[iblk]..mat.col_ind[iblk + 1] {
            assert!(icrs < mat.row_ptr.len());
            let jblk0 = mat.row_ptr[icrs];
            assert!(jblk0 < mat.ncolblk);
            lhs[iblk] += alpha * mat.val_crs[icrs] * rhs[jblk0];
        }
        lhs[iblk] += alpha * mat.val_dia[iblk] * rhs[iblk];
    }
}


#[test]
fn test1() {
    type MAT = nalgebra::Matrix3<f32>;
    type VEC = nalgebra::Vector3<f32>;
    let mut sparse = BlockSparseMatrix::<MAT>::new();
    let colind = vec![0, 2, 5, 8, 10];
    let rowptr = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    sparse.initialize_as_square_matrix(&colind, &rowptr);
    sparse.set_zero();
    {
        let emat = [
            nalgebra::Matrix3::<f32>::identity(),
            nalgebra::Matrix3::<f32>::zeros(),
            nalgebra::Matrix3::<f32>::zeros(),
            nalgebra::Matrix3::<f32>::identity()];
        let mut tmp_buffer = Vec::<usize>::new();
        sparse.merge(&[0, 1], &[0, 1], &emat, &mut tmp_buffer);
    }
    let nblk = colind.len() - 1;
    let mut rhs = Vec::<VEC>::new();
    rhs.resize(nblk, Default::default());
    let mut lhs = Vec::<VEC>::new();
    lhs.resize(nblk, Default::default());
    gemm_for_block_sparse_matrix_nalgebra(&mut lhs, 1.0, 1.0, &sparse, &rhs);
}

#[test]
fn test2() {
    let mut sparse = BlockSparseMatrix::<f32>::new();
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
    gemm_for_sparse_matrix(&mut lhs, 1.0, 1.0, &sparse, &rhs);
}