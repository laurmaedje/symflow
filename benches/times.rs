use std::fs::{self, File};
use std::io::Write;

use symflow::Program;
use symflow::flow::{ControlFlowGraph, DataDependencyGraph};
use symflow::timings;


fn main() {
    bench("bufs");
    bench("paths");
    bench("deep");
    bench("overwrite");
}

fn bench(filename: &str) {
    let path = format!("target/bin/{}", filename);

    timings::reset();

    let program = Program::new(path);
    let cfg = ControlFlowGraph::new(&program);
    let _ddg = DataDependencyGraph::new(&cfg);

    let measurements = timings::get();

    fs::create_dir("target/bench").ok();
    let bench_path = format!("target/bench/{}.txt", filename);
    let mut bench_file = File::create(bench_path).unwrap();

    writeln!(bench_file, "Benchmark for {}\n", filename).unwrap();
    write!(bench_file, "{}", measurements).unwrap();
}
