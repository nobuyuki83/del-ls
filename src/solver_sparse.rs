use crate::sparse_square;

pub fn solve_cg(
    r_vec: &mut Vec<f32>,
    u_vec: &mut Vec<f32>,
    ap_vec: &mut Vec<f32>,
    p_vec: &mut Vec<f32>,
    conv_ratio_tol: f32,
    max_iteration: usize,
    mat: &sparse_square::Matrix<f32>) -> Vec<f32> {
    use crate::slice::{set_zero, dot, copy, add_scaled_vector, scale_and_add_vec};
    {
        let n = r_vec.len();
        u_vec.resize(n, 0.);
        ap_vec.resize(n, 0.);
        p_vec.resize(n, 0.);
    }
    let mut conv_hist = Vec::<f32>::new();
    set_zero(u_vec);
    let mut sqnorm_res = dot(r_vec, r_vec);
    if sqnorm_res < 1.0e-30 { return conv_hist; }
    let inv_sqnorm_res_ini = 1.0 / sqnorm_res;
    copy(p_vec, &r_vec);  // {p} = {r}  (set initial serch direction, copy value not reference)
    for _iitr in 0..max_iteration {
        let alpha;
        {  // alpha = (r,r) / (p,Ap)
            sparse_square::gemv_for_sparse_matrix(
                ap_vec,
                0.0, 1.0, &mat, &p_vec); // {Ap_vec} = [mat]*{p_vec}
            let pap = dot(p_vec, ap_vec);
            assert!(pap >=0.);
            alpha = sqnorm_res / pap;
        }
        add_scaled_vector(u_vec, alpha, p_vec);    // {u} = +alpha*{p} + {u} (update x)
        add_scaled_vector(r_vec, -alpha, ap_vec);  // {r} = -alpha*{Ap} + {r}
        let sqnorm_res_new = dot(r_vec, r_vec);
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