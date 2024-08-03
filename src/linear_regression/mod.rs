use crate::process_data;

#[test]
fn test_linear_regression() {

    if let Ok(lines) = process_data::read_lines("./data/ex1data1.txt") {
        for line in lines {
            if let Ok(line) = line {
                println!("{}", line);
            }      
        }   
    }
    assert!(false);
}