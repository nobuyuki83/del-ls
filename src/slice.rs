pub fn dot(
    v0: &[f32],
    v1: &[f32]) -> f32
{
    assert_eq!(v0.len(), v1.len());
    let mut sum: f32 = 0.;
    for i in 0..v0.len() {
        sum += v0[i] * v1[i];
    }
    sum
}

pub fn add_scaled_vector(
    u: &mut [f32],
    alpha: f32,
    p: &[f32]) {
    assert_eq!(u.len(), p.len());
    for i in 0..p.len() {
        u[i] += alpha * p[i];
    }
}

pub fn scale_and_add_vec(
    p: &mut [f32],
    beta: f32,
    r: &[f32]) { // {p} = {r} + beta*{p}
    assert_eq!(r.len(), p.len());
    for i in 0..p.len() {
        p[i] = r[i] + beta * p[i];
    }
}

pub fn set_zero(
    p: &mut [f32]) {
    p.iter_mut().for_each(|v| *v = 0_f32 );
}

pub fn copy(
    p: &mut [f32],
    u: &[f32]) {
    assert_eq!(p.len(), u.len());
    for i in 0..p.len() {
        p[i] = u[i];
    }
}

pub fn sub(
    p: &mut [f32],
    u: &[f32],
    v: &[f32])  {
    assert_eq!(p.len(), u.len());
    for i in 0..p.len() {
        p[i] = u[i] - v[i];
    }
}