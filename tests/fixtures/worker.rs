#[no_mangle]
pub extern "C" fn process(n: u32) -> f64 {
    let numbers: Vec<f64> = (0..n).map(|i| f64::from(i) * 1.25 + 0.5).collect();
    numbers.iter().sum()
}

#[no_mangle]
pub extern "C" fn file_process() -> i32 {
    let content = std::fs::read_to_string("numbers.txt").unwrap();
    let sum: f64 = content.lines().map(|line| line.parse::<f64>().unwrap()).sum();
    std::fs::write("rust-result.txt", format!("sum={sum:.3}\n")).unwrap();
    (sum * 1000.0) as i32
}
