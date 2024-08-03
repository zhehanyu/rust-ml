mod linear_regression;
mod process_data;

use ndarray::prelude::*;
fn main() {
    let arr1: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> = Array::zeros((3,4));
    let x = arr1.columns();
    println!("{:#?}", arr1);
}
