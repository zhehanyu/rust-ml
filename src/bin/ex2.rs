use rust_ml::*;

fn main() -> anyhow::Result<()> {

    // 读取数据
    let mut raw_data = process_data::read_float_matrix("./data/ex2data1.txt").unwrap();
    raw_data = process_data::normolized(raw_data, true);
    process_logistic_regression(raw_data);
    Ok(())
}