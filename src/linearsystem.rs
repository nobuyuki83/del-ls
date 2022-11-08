

pub struct Solver {
    pub sparse : crate::sparse_square::Matrix<f32>,
    pub merge_buffer : Vec<usize>,
    pub r_vec : Vec<f32>,
    pub u_vec : Vec<f32>,
    pub conv : Vec<f32>,
    pub conv_ratio_tol: f32,
    pub max_num_iteration: usize,
    ap_vec : Vec<f32>,
    p_vec : Vec<f32>
}

impl Solver {
    pub fn new() -> Self {
        Solver {
            sparse: crate::sparse_square::Matrix::<f32>::new(),
            merge_buffer : Vec::<usize>::new(),
            ap_vec : Vec::<f32>::new(),
            p_vec : Vec::<f32>::new(),
            r_vec : Vec::<f32>::new(),
            u_vec : Vec::<f32>::new(),
            conv : Vec::<f32>::new(),
            conv_ratio_tol: 1.0e-5,
            max_num_iteration: 100
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
        self.conv = crate::solver_sparse::solve_cg(
            &mut self.r_vec, &mut self.u_vec,
            &mut self.ap_vec, &mut self.p_vec,
            self.conv_ratio_tol, self.max_num_iteration,
            &self.sparse);
    }

    pub fn clone(&self) -> Self {
        Solver {
            sparse: self.sparse.clone(),
            merge_buffer : self.merge_buffer.clone(),
            ap_vec : self.ap_vec.clone(),
            p_vec : self.p_vec.clone(),
            r_vec : self.r_vec.clone(),
            u_vec : self.u_vec.clone(),
            conv: self.conv.clone(),
            conv_ratio_tol: self.conv_ratio_tol,
            max_num_iteration: 100
        }
    }
}