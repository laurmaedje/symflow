//! Visualization of flow graphs.

use std::collections::HashMap;
use std::io::{Result, Write};
use crate::math::SymCondition;

pub const BR: &str = "<br align=\"left\"/>";


/// Write the preamble of the graphviz file.
pub fn write_header<W: Write>(mut f: W, title: &str) -> Result<()> {
    writeln!(f, "digraph Flow {{")?;
    writeln!(f, "label=<{}<br/><br/>>", title)?;
    writeln!(f, "labelloc=\"t\"")?;
    writeln!(f, "graph [fontname=\"Source Code Pro\"]")?;
    writeln!(f, "node [fontname=\"Source Code Pro\"]")?;
    writeln!(f, "edge [fontname=\"Source Code Pro\"]")
}

/// Write condition edges.
pub fn write_edges<W: Write, F>(
    mut f: W,
    edges: &HashMap<(usize, usize), SymCondition>,
    writer: F
) -> Result<()> where F: Fn(&mut W, ((usize, usize), &SymCondition)) -> Result<()> {
    // Export the edges, but sort them first to make the graphviz output
    // deterministic eventhough the hash map cannot be traversed in order.
    let mut edges = edges.iter().collect::<Vec<_>>();
    edges.sort_by_key(|edge| edge.0);
    for (&(start, end), condition) in edges {
        write!(f, "b{} -> b{} [", start, end)?;
        writer(&mut f, ((start, end), &condition))?;
        writeln!(f, "]")?;
    }
    Ok(())
}

/// Write the closing of the file.
pub fn write_footer<W: Write>(mut f: W) -> Result<()> {
    writeln!(f, "}}")
}


#[cfg(test)]
pub mod test {
    use std::fs::{self, File};
    use std::process::Command;
    use super::*;

    /// Compile the file with graphviz.
    pub fn compile<F>(dir: &str, filename: &str, writer: F)
    where F: FnOnce(File) -> Result<()> {
        // Visualize the graph into a PDF file.
        fs::create_dir(dir).ok();
        let flow_temp = "target/temp-flow.gv";
        let flow_file = File::create(flow_temp).unwrap();
        writer(flow_file).unwrap();
        let output = Command::new("dot")
            .arg("-Tpdf")
            .arg(flow_temp)
            .arg("-o")
            .arg(format!("{}/{}.pdf", dir, filename))
            .output()
            .expect("failed to run graphviz");
        std::io::stdout().write_all(&output.stdout).unwrap();
        std::io::stderr().write_all(&output.stderr).unwrap();
        // fs::remove_file(flow_temp).unwrap();
    }
}
