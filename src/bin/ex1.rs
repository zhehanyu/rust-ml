use rust_ml::*;
use ndarray::prelude::*;

fn main() -> anyhow::Result<()> {

    // 读取数据
    let raw_data = process_data::read_float_matrix("./data/ex1data1.txt").unwrap();

    // plot数据
    // let _ = process_data::plot_data(raw_data.view(), "plot.svg");

    process_linear_regression(raw_data);

    // 读取数据
    let mut raw_data = process_data::read_float_matrix("./data/ex1data2.txt").unwrap();
    

    raw_data = normolized(raw_data);
    process_linear_regression(raw_data);

    Ok(())
}