
pub mod elf;


#[cfg(test)]
mod tests {
    use std::fs;
    use super::elf::*;

    #[test]
    fn decode() {
        let bin = fs::read("test/block").unwrap();

        let mut reader = ElfReader::from_slice(&bin);
        let text = reader.get_section(".text").unwrap();

        println!(".text: {:?}", text);
    }
}
