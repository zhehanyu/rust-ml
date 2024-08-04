use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use ndarray::prelude::*;
use plotters::prelude::*;

pub type OwnedFloatMatrix = ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>;
pub type ViewFloatMatrix<'a> = ArrayBase<ndarray::ViewRepr<&'a f64>, ndarray::Dim<[usize; 2]>>;
pub type ViewMutFloatMatrix<'a> = ArrayBase<ndarray::ViewRepr<&'a mut f64>, ndarray::Dim<[usize; 2]>>;

pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

pub fn read_float_matrix<P>(filename: P) -> Option<OwnedFloatMatrix>
where P: AsRef<Path>, {
    if let Ok(lines) = read_lines(filename) {
        let mut row_number = 0;
        let mut column_number = 0;

        let mut temp = Vec::new();
        for line in lines {
            row_number += 1;

            let row: Vec<f64> = line.unwrap().split(',').map(|x| {
                x.parse().unwrap()
            }).collect();

            if column_number != 0 && column_number != row.len() {
                return None;
            }
            column_number = row.len();
            temp.push(row);
        }
        let mut res = Array::zeros((row_number, column_number));
        for i in 0..row_number {
            for j in 0..column_number {
                res[[i,j]] = temp[i][j];
            }
        }
        Some(res)
    } else {
        None
    }
}

pub fn normolized(mut raw_data: OwnedFloatMatrix, skip_last_column: bool) -> OwnedFloatMatrix {
    let res = Array2::from_shape_fn(raw_data.dim(), |(i,j)| {
        if skip_last_column && j == raw_data.ncols() - 1 {
            raw_data[[i,j]]
        } else {
            (raw_data[[i,j]] - raw_data.mean_axis(Axis(0)).unwrap()[j]) / raw_data.std_axis(Axis(0), 1.0)[j]
        }
    });

    res
}

pub fn plot_data(data: ViewFloatMatrix, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 创建svg图片对象
    let root = SVGBackend::new(path, (640, 480)).into_drawing_area();
    // 图片对象的背景颜色填充
    root.fill(&WHITE)?;
    // 创建绘图对象
    let mut chart = ChartBuilder::on(&root)
        // 图表名称  (字体样式, 字体大小)
        .caption("散点图", ("sans-serif", 30))
        // 图表左侧与图片边缘的间距
        .set_label_area_size(LabelAreaPosition::Left, 40)
        // 图表底部与图片边缘的间距
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        // 构建二维图像, x轴 0.0 - 25.0； y轴 -5.0 - 25.0；
        .build_cartesian_2d(0.0..25.0, -5.0..25.0)?;
    // 配置网格线
    chart.configure_mesh().draw()?;
 
    // 绘制散点
    chart.draw_series(
        data.rows()
            .into_iter().map(|row|{
                Circle::new((row[0], row[1]), 5, BLUE.filled())
            }).collect::<Vec<_>>(),
    )?;
 
    Ok(())
}