use ndarray::{prelude::*, stack};

use crate::*;

pub fn cauculate_cost(x: ViewFloatMatrix, y: ViewFloatMatrix, theta: ViewFloatMatrix) -> f64 {
    let inner = (x.dot(&theta) - y).mapv(|v|{v*v});
    
    inner.sum() / (2.0 * inner.len() as f64)
}


pub fn gradient_descent(x: ViewFloatMatrix, y: ViewFloatMatrix, mut theta: ViewMutFloatMatrix, alpha: f64, iters: usize) {
    let number_parameter = theta.nrows();
    for _ in 0..iters {
        let error = x.dot(&theta) - y;
        for i in 0..number_parameter {
            theta[[i,0]] = theta[[i,0]] - alpha * (error.clone() * x.slice(s![..,i..i+1])).sum() / x.nrows() as f64;
        }
    }
}

pub fn normolized(mut raw_data: OwnedFloatMatrix) -> OwnedFloatMatrix {
    // let res = Array0::default(shape)
    // for column in raw_data.columns {
    //     let tmp = (column.clone() - column.mean().unwrap()) / column.std(1.0);
    // }

    let res = Array2::from_shape_fn(raw_data.dim(), |(i,j)| {
        (raw_data[[i,j]] - raw_data.mean_axis(Axis(0)).unwrap()[j]) / raw_data.std_axis(Axis(0), 1.0)[j]
    });

    res
}

pub fn process_linear_regression(raw_data: OwnedFloatMatrix) {
    // 拿到行数
    let n = raw_data.nrows();

    // 加上全为1的一列
    let mut data = Array::ones((n,1));
    for column in raw_data.columns() {
        let _ = data.push_column(column);
    }


    // 计算一遍
    let x = data.slice(s![..,..-1]); // 变量
    let y = data.slice(s![..,-1..]); // 结果
    let mut theta = Array::zeros((x.ncols(),1));
    println!("Before gradient descent, error: {}",cauculate_cost(x,y,theta.view()));
    gradient_descent(x, y, theta.view_mut(), 0.01, 1000);
    println!("After gradient descent, error: {}",cauculate_cost(x,y,theta.view()));
    let parameter_number = theta.nrows();
    for i in 0..parameter_number {
        println!("theta_{} = {}", i, theta[[i,0]]);
    }
}