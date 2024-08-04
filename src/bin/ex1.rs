use rust_ml::*;

fn main() -> anyhow::Result<()> {

    // 读取数据
    let raw_data = process_data::read_float_matrix("./data/ex1data1.txt").unwrap();
    process_linear_regression(raw_data);

    // 读取数据
    let mut raw_data = process_data::read_float_matrix("./data/ex1data2.txt").unwrap();
    raw_data = process_data::normolized(raw_data, false);
    process_linear_regression(raw_data);

    Ok(())
}