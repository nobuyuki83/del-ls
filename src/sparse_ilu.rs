use num_traits::AsPrimitive;

pub struct Preconditioner<T> {
    pub num_blk: usize,
    pub row2idx: Vec<usize>,
    pub idx2col: Vec<usize>,
    pub row2idx_dia: Vec<usize>,
    pub idx2val: Vec<T>,
    pub row2val: Vec<T>,
}

impl<T> Preconditioner<T>
    where
        T: 'static + Copy + num_traits::Zero,
        f32: AsPrimitive<T>
{
    pub fn new() -> Self {
        Preconditioner {
            num_blk: 0,
            row2idx: vec!(0),
            idx2col: Vec::<usize>::new(),
            idx2val: Vec::<T>::new(),
            row2val: Vec::<T>::new(),
            row2idx_dia: Vec::<usize>::new(),
        }
    }

    pub fn clone(&self) -> Self {
        Preconditioner {
            num_blk: self.num_blk,
            row2idx: self.row2idx.clone(),
            idx2col: self.idx2col.clone(),
            idx2val: self.idx2val.clone(),
            row2val: self.row2val.clone(),
            row2idx_dia: self.row2idx_dia.clone(),
        }
    }

    pub fn initialize_ilu0(
        &mut self,
        a: &crate::sparse_square::Matrix<T>)
    {
        let num_row = a.num_blk;
        self.num_blk = num_row;
        self.row2idx = a.row2idx.clone();
        self.idx2col = a.idx2col.clone();
        self.idx2val = vec!(T::zero(); a.idx2col.len());
        self.row2val = vec!(T::zero(); num_row);
        self.row2idx_dia = vec!(0_usize; num_row);
        // ---------------
        // sort idx2col
        for i_row in 0..num_row {
            let idx0 = self.row2idx[i_row];
            let idx1 = self.row2idx[i_row + 1];
            self.idx2col[idx0..idx1].sort();
        }
        // set row2idx_dia
        for i_row in 0..num_row {
            self.row2idx_dia[i_row] = self.row2idx[i_row + 1];
            for ij_idx in self.row2idx[i_row]..self.row2idx[i_row + 1] {
                assert!(ij_idx < self.idx2col.len());
                let j_col = self.idx2col[ij_idx];
                assert!(j_col < num_row);
                if j_col > i_row {
                    self.row2idx_dia[i_row] = ij_idx;
                    break;
                }
            }
        }
    }

    pub fn initialize_full(
        &mut self,
        num_row: usize)
    {
        self.num_blk = num_row;
        self.row2idx.resize(num_row + 1, 0_usize);
        for i_row in 0..num_row + 1 {
            self.row2idx[i_row] = i_row * (num_row - 1);
        }
        self.idx2col = vec!(0_usize; num_row * (num_row - 1));
        for i_row in 0..num_row {
            for j_col in 0..i_row {
                self.idx2col[i_row * (num_row - 1) + j_col] = j_col;
            }
            for j_col in i_row + 1..num_row {
                self.idx2col[i_row * (num_row - 1) + j_col - 1] = j_col;
            }
        }
        self.row2idx_dia = vec!(0_usize; num_row);
        for i_row in 0..num_row {
            self.row2idx_dia[i_row] = self.row2idx[i_row + 1];
            for ij_idx in self.row2idx[i_row]..self.row2idx[i_row + 1] {
                assert!(ij_idx < self.idx2col.len());
                let j_col = self.idx2col[ij_idx];
                assert!(j_col < num_row);
                if j_col > i_row {
                    self.row2idx_dia[i_row] = ij_idx;
                    break;
                }
            }
        }
        self.idx2val = vec!(T::zero(); self.idx2col.len());
        self.row2val = vec!(T::zero(); num_row);
    }

    pub fn initialize_iluk(
        &mut self,
        a: &crate::sparse_square::Matrix<T>,
        lev_fill: usize) {
        if lev_fill == 0 {
            self.initialize_ilu0(a);
            return;
        }
        (self.row2idx, self.idx2col, self.row2idx_dia) = symbolic_iluk(
            &a.row2idx, a.idx2col.clone(), lev_fill);
        self.num_blk = self.row2idx.len() - 1;
        self.idx2val.resize(self.idx2col.len(), T::zero());
        self.row2val.resize(self.num_blk, T::zero());
    }
}

pub fn copy_value<T>(
    ilu: &mut Preconditioner<T>,
    a: &crate::sparse_square::Matrix<T>)
    where
        T: num_traits::Zero + Copy
{
    let num_row = ilu.num_blk;
    assert_eq!(a.num_blk, num_row);
    let mut col2idx = vec!(usize::MAX; num_row);
    // copy diagonal value
    crate::slice::copy(&mut ilu.row2val, &a.row2val);
    // copy off-diagonal values
    ilu.idx2val.iter_mut().for_each(|v| v.set_zero());
    for i_row in 0..num_row {
        for ij_idx0 in ilu.row2idx[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx0 < ilu.idx2col.len());
            let j_col0 = ilu.idx2col[ij_idx0];
            assert!(j_col0 < num_row);
            col2idx[j_col0] = ij_idx0;
        }
        for a_ij_idx in a.row2idx[i_row]..a.row2idx[i_row + 1] {
            assert!(a_ij_idx < a.idx2col.len());
            let j_col0 = a.idx2col[a_ij_idx];
            assert!(j_col0 < num_row);
            let ij_idx = col2idx[j_col0];
            if ij_idx != usize::MAX {
                ilu.idx2val[ij_idx] = a.idx2val[a_ij_idx];
            } else {
                panic!();
            }
        }
        for ij_idx0 in ilu.row2idx[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx0 < ilu.idx2col.len());
            let j_col0 = ilu.idx2col[ij_idx0];
            assert!(j_col0 < num_row);
            col2idx[j_col0] = usize::MAX;
        }
    }
}

pub fn decompose<T>(
    ilu: &mut Preconditioner<T>)
    where
        T: 'static + Copy + std::ops::Mul<Output=T> + std::ops::SubAssign + std::ops::Div<Output=T>,
        f32: AsPrimitive<T>
{
    let num_row = ilu.num_blk;
    let mut col2idx = vec!(usize::MAX; num_row);
    for i_row in 0..num_row {
        for ij_idx in ilu.row2idx[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx < ilu.idx2col.len());
            let j_col = ilu.idx2col[ij_idx];
            assert!(j_col < num_row);
            col2idx[j_col] = ij_idx;
        }
        // [L] * [D^-1*U]
        for ik_idx in ilu.row2idx[i_row]..ilu.row2idx_dia[i_row] {
            let k_colrow = ilu.idx2col[ik_idx];
            assert!(k_colrow < num_row);
            let ik_val = ilu.idx2val[ik_idx];
            for kj_idx in ilu.row2idx_dia[k_colrow]..ilu.row2idx[k_colrow + 1] {
                let j_col = ilu.idx2col[kj_idx];
                assert!(j_col < num_row);
                let kj_val = ilu.idx2val[kj_idx];
                if j_col != i_row {
                    let ij_idx = col2idx[j_col];
                    if ij_idx == usize::MAX { continue; }
                    ilu.idx2val[ij_idx] -= ik_val * kj_val;
                } else {
                    ilu.row2val[i_row] -= ik_val * kj_val;
                }
            }
        }
        // invserse diagonal
        ilu.row2val[i_row] = 1_f32.as_() / ilu.row2val[i_row];
        // [U] = [1/D][U]
        for ij_idx in ilu.row2idx_dia[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx < ilu.idx2col.len());
            ilu.idx2val[ij_idx] = ilu.idx2val[ij_idx] * ilu.row2val[i_row];
        }
        for ij_idx in ilu.row2idx[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx < ilu.idx2col.len());
            let j_col = ilu.idx2col[ij_idx];
            assert!(j_col < num_row);
            col2idx[j_col] = usize::MAX;
        }
    }    // end iblk
}

pub fn solve_preconditioning<T>(
    vec: &mut Vec<T>,
    ilu: &Preconditioner<T>)
    where
        T: Copy + std::ops::Mul<Output=T> + std::ops::SubAssign
{
    // forward
    let num_row = ilu.num_blk;
    for i_row in 0..num_row {
        for ij_idx in ilu.row2idx[i_row]..ilu.row2idx_dia[i_row] {
            assert!(ij_idx < ilu.idx2col.len());
            let j_col = ilu.idx2col[ij_idx];
            assert!(j_col < i_row);
            let v = ilu.idx2val[ij_idx] * vec[j_col];
            vec[i_row] -= v; // jblk0!=iblk
        }
        vec[i_row] = ilu.row2val[i_row] * vec[i_row];
    }
    // -----
    // backward
    for i_row in (0..num_row).rev() {
        assert!(i_row < num_row);
        for ij_idx in ilu.row2idx_dia[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx < ilu.idx2col.len());
            let j_col = ilu.idx2col[ij_idx];
            assert!(j_col > i_row && j_col < num_row);
            let v = ilu.idx2val[ij_idx] * vec[j_col];
            vec[i_row] -= v; // jblk0!=iblk
        }
    }
}

fn symbolic_iluk(
    a_row2idx: &Vec<usize>,
    mut a_idx2col: Vec<usize>,
    lev_fill: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>)
{
    let num_row = a_row2idx.len() - 1;
    for i_row in 0..num_row {
        let idx0 = a_row2idx[i_row];
        let idx1 = a_row2idx[i_row + 1];
        a_idx2col[idx0..idx1].sort();
    }
    let mut row2idx = vec!(0_usize; num_row + 1);
    let mut row2idx_dia = vec!(0; num_row);

    let mut idx2collev = Vec::<[usize; 2]>::new();
    idx2collev.reserve(a_idx2col.len() * 4);

    for i_row in 0..num_row {
        let mut col2lev = std::collections::BTreeMap::<usize, usize>::new();
        let mut que0 = std::collections::BTreeSet::<usize>::new();
        for j_col in &a_idx2col[a_row2idx[i_row]..a_row2idx[i_row + 1]] {
            col2lev.insert(*j_col, 0);
            if *j_col < i_row {
                que0.insert(*j_col);
            }
        }
        loop {
            let k_colrow = match que0.iter().next() {
                None => break,
                Some(x) => x.clone()
            };
            que0.remove(&k_colrow);
            assert!(k_colrow < i_row);
            let ik_lev0 = col2lev.get(&k_colrow).unwrap().clone();
            if ik_lev0 + 1 > lev_fill { continue; } // move next
            for kj_idx in row2idx_dia[k_colrow]..row2idx[k_colrow + 1] {
                let kj_lev0 = idx2collev[kj_idx][1];
                if kj_lev0 + 1 > lev_fill { continue; }
                let j_col = idx2collev[kj_idx][0];
                assert!(j_col > k_colrow && j_col < num_row);
                if j_col == i_row { continue; } // already filled-in on the diagonal
                let max_lev0 = if ik_lev0 > kj_lev0 { ik_lev0 } else { kj_lev0 };
                if j_col < i_row {
                    que0.insert(j_col);
                }
                match col2lev.get_mut(&j_col) {
                    None => { col2lev.insert(j_col, max_lev0 + 1); }
                    Some(lev) => {
                        *lev = if *lev < max_lev0 + 1 { *lev } else { max_lev0 + 1 };
                    }
                }
            }
        }
        {  // finalize this row
            let ij_idx1 = row2idx[i_row] + col2lev.len();
            idx2collev.reserve(row2idx[i_row] + col2lev.len());
            for collev in col2lev.iter() {
                idx2collev.push([*collev.0, *collev.1]);
            }
            row2idx[i_row + 1] = ij_idx1;
            row2idx_dia[i_row] = ij_idx1;
            for ij_idx1 in row2idx[i_row]..row2idx[i_row + 1] {
                let j_col = idx2collev[ij_idx1][0];
                if j_col > i_row {
                    row2idx_dia[i_row] = ij_idx1;
                    break;
                }
            }
        }
    }

    let mut idx2col = vec!(0_usize; idx2collev.len());
    for icrs in 0..idx2col.len() {
        idx2col[icrs] = idx2collev[icrs][0];
    }
    (row2idx, idx2col, row2idx_dia)
}