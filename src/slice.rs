//!  operations on slice of an array

use num_traits::AsPrimitive;

/// dot product
pub fn dot<T>(
    v0: &[T],
    v1: &[T]) -> T
where
    T: 'static + Copy + std::ops::Mul<Output = T> + std::ops::AddAssign,
    f32: AsPrimitive<T>
{
    assert_eq!(v0.len(), v1.len());
    let mut sum: T = 0_f32.as_();
    for i in 0..v0.len() {
        sum += v0[i] * v1[i];
    }
    sum
}

pub fn add_scaled_vector<T>(
    u: &mut [T],
    alpha: T,
    p: &[T])
where T: std::ops::Mul<Output = T> + std::ops::AddAssign + Copy
{
    assert_eq!(u.len(), p.len());
    for i in 0..p.len() {
        u[i] += alpha * p[i];
    }
}

/// {p} = {r} + beta*{p}
pub fn scale_and_add_vec<T>(
    p: &mut [T],
    beta: T,
    r: &[T])
where T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy
{
    assert_eq!(r.len(), p.len());
    for i in 0..p.len() {
        p[i] = r[i] + beta * p[i];
    }
}

pub fn set_zero<T>(
    p: &mut [T])
where T: 'static + Copy,
    f32: AsPrimitive<T>
{
    p.iter_mut().for_each(|v| *v = 0_f32.as_() );
}

pub fn copy<T>(
    p: &mut [T],
    u: &[T])
where T: Copy
{
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