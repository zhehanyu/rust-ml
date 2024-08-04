use ndarray::prelude::*;
use crate::*;

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

pub fn logistic_regression_cost(x: ViewFloatMatrix, y: ViewFloatMatrix, theta: ViewFloatMatrix) -> f64 {
    
    let first = x.dot(&theta).mapv(|v| {
        sigmoid(v).ln()
    }) *
    (y.mapv(|v| {
        -v
    }));

    let second = x.dot(&theta).mapv(|v| {
        (1.0 - sigmoid(v)).ln()
    }) *
    (y.mapv(|v| {
        1.0 - v
    }));

    (first - second).sum() / x.nrows() as f64

}

pub fn logistic_regression_gradient_descent(x: ViewFloatMatrix, y: ViewFloatMatrix, mut theta: ViewMutFloatMatrix, alpha: f64, iters: usize) {
    let number_parameter = theta.nrows();
    for _ in 0..iters {
        let error = x.dot(&theta).mapv(|v| sigmoid(v)) - y;
        for i in 0..number_parameter {
            theta[[i,0]] = theta[[i,0]] - alpha * (error.clone() * x.slice(s![..,i..i+1])).sum() / x.nrows() as f64;
        }
    }
}

pub fn process_logistic_regression(raw_data: OwnedFloatMatrix) {
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
 
    println!("Before gradient descent, error: {}",logistic_regression_cost(x,y,theta.view()));

    logistic_regression_gradient_descent(x, y, theta.view_mut(), 0.008, 100000);
    println!("After gradient descent, error: {}",logistic_regression_cost(x,y,theta.view()));
    let parameter_number = theta.nrows();
    for i in 0..parameter_number {
        println!("theta_{} = {}", i, theta[[i,0]]);
    }
}