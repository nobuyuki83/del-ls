fn dot_vectors(
    v0: &Vec<f32>,
    v1: &Vec<f32>) -> f32
{
    assert!(v0.len() == v1.len());
    let mut sum: f32 = 0.;
    for i in 0..v0.len() {
        sum += v0[i] * v1[i];
    }
    sum
}

fn add_scaled_vector(
    u: &mut Vec<f32>,
    alpha: f32,
    p: &Vec<f32>) {
    assert!(u.len() == p.len());
    for i in 0..p.len() {
        u[i] += alpha * p[i];
    }
}

fn scale_and_add_vec(
    p: &mut Vec<f32>,
    beta: f32,
    r: &Vec<f32>) { // {p} = {r} + beta*{p}
    assert!(r.len() == p.len());
    for i in 0..p.len() {
        p[i] = r[i] + beta * p[i];
    }
}


fn set_zero(
    p: &mut Vec<f32>) {
    for i in 0..p.len() {
        p[i] = 0.;
    }
}

fn copy(
    p: &mut Vec<f32>,
    u: &Vec<f32>) {
    assert!(p.len() == u.len());
    p.resize(u.len(), 0.);
    for i in 0..p.len() {
        p[i] = u[i];
    }
}

use crate::ls_sparse;

pub fn solve_cg(
    r_vec: &mut Vec<f32>,
    u_vec: &mut Vec<f32>,
    ap_vec: &mut Vec<f32>,
    p_vec: &mut Vec<f32>,
    conv_ratio_tol: f32,
    max_iteration: usize,
    mat: &ls_sparse::BlockSparseMatrix<f32>) -> Vec<f32> {
    {
        let n = r_vec.len();
        u_vec.resize(n, 0.);
        ap_vec.resize(n, 0.);
        p_vec.resize(n, 0.);
    }
    let mut conv_hist = Vec::<f32>::new();
    set_zero(u_vec);
    let mut sqnorm_res = dot_vectors(r_vec, r_vec);
    if sqnorm_res < 1.0e-30 { return conv_hist; }
    let inv_sqnorm_res_ini = 1.0 / sqnorm_res;
    copy(p_vec, &r_vec);  // {p} = {r}  (set initial serch direction, copy value not reference)
    for _iitr in 0..max_iteration {
        let alpha;
        {  // alpha = (r,r) / (p,Ap)
            ls_sparse::gemm_for_sparse_matrix(
                ap_vec,
                0.0, 1.0, &mat, &p_vec); // {Ap_vec} = [mat]*{p_vec}
            let pap = dot_vectors(p_vec, ap_vec);
            assert!(pap >=0.);
            alpha = sqnorm_res / pap;
        }
        add_scaled_vector(u_vec, alpha, p_vec);    // {u} = +alpha*{p} + {u} (update x)
        add_scaled_vector(r_vec, -alpha, ap_vec);  // {r} = -alpha*{Ap} + {r}
        let sqnorm_res_new = dot_vectors(r_vec, r_vec);
        let conv_ratio = (sqnorm_res_new * inv_sqnorm_res_ini).sqrt();
        conv_hist.push(conv_ratio);
        if conv_ratio < conv_ratio_tol { return conv_hist; }
        {
            let beta = sqnorm_res_new / sqnorm_res; // beta = (r1,r1) / (r0,r0)
            sqnorm_res = sqnorm_res_new;
            scale_and_add_vec(p_vec, beta, &r_vec); // {p} = {r} + beta*{p}
        }
    }
    conv_hist
}