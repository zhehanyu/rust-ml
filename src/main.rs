mod linear_regression;
mod process_data;

use rust_ml::*;

use anyhow::Result;
use ndarray::prelude::*;

fn cauculate_cost(x: ViewFloatMatrix, y: ViewFloatMatrix, theta: ViewFloatMatrix) -> f64 {
    let inner = (x.dot(&theta) - y).mapv(|v|{v*v});
    
    inner.sum() / (2.0 * inner.len() as f64)
}


fn gradient_descent(x: ViewFloatMatrix, y: ViewFloatMatrix, mut theta: ViewMutFloatMatrix, alpha: f64, iters: usize) {
    let number_parameter = theta.nrows();
    for _ in 0..iters {
        let error = x.dot(&theta) - y;
        for i in 0..number_parameter {
            theta[[i,0]] = theta[[i,0]] - alpha * (error.clone() * x.slice(s![..,i..i+1])).sum() / x.nrows() as f64;
        }
    }
}
fn main() -> Result<()>{

    // 读取数据
    let raw_data = process_data::read_float_matrix("./data/ex1data1.txt").unwrap();

    // plot数据
    // let _ = process_data::plot_data(raw_data.view(), "plot.svg");

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
    let mut theta = Array::zeros((2,1));


    println!("{}",cauculate_cost(x,y,theta.view()));
    gradient_descent(x, y, theta.view_mut(), 0.01, 1000);
    println!("{}",cauculate_cost(x,y,theta.view()));
    Ok(())
}
