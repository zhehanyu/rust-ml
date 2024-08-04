use rust_ml::*;


fn main() {

    let data = process_data::read_float_matrix("./data/ex1data1.txt").unwrap();
    let _ = process_data::plot_data(data.view(), "plot.svg");
}