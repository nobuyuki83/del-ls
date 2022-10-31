

pub struct Solver {
    pub sparse : crate::sparse::BlockSparseMatrix<f32>,
    pub merge_buffer : Vec<usize>,
    pub r_vec : Vec<f32>,
    pub u_vec : Vec<f32>,
    ap_vec : Vec<f32>,
    p_vec : Vec<f32>
}

impl Solver {
    pub fn new() -> Self {
        Solver {
            sparse: crate::sparse::BlockSparseMatrix::<f32>::new(),
            merge_buffer : Vec::<usize>::new(),
            ap_vec : Vec::<f32>::new(),
            p_vec : Vec::<f32>::new(),
            r_vec : Vec::<f32>::new(),
            u_vec : Vec::<f32>::new()
        }
    }

    pub fn initialize(
        &mut self,
        colind: &Vec<usize>,
        rowptr: &Vec<usize>) {
        self.sparse.initialize_as_square_matrix(&colind, &rowptr);
        let nblk = colind.len() - 1;
        self.r_vec.resize(nblk, 0.);
    }

    pub fn begin_mearge(
        &mut self) {
        self.sparse.set_zero();
    }

    pub fn solve_cg (&mut self) {
        let conv = crate::solver_sparse::solve_cg(
            &mut self.r_vec, &mut self.u_vec,
            &mut self.ap_vec, &mut self.p_vec,
            1.0e-5, 100,
            &self.sparse);
        println!("number of iteration: {}", conv.len());
    }
}