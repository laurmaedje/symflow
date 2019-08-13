//! Parsing of the 64-bit `ELF` file format.

use std::ffi::CStr;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use byteorder::{ReadBytesExt, LE};


/// Handle for an `ELF` file.
#[derive(Debug)]
pub struct ElfFile<R> where R: Read + Seek {
    target: R,
    pub header: Header,
    pub section_headers: Vec<SectionHeader>,
}

/// Header of a file.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Header {
    pub identification: [u8; 16],
    pub file_type: u16,
    pub machine: u16,
    pub version: u32,
    pub entry: u64,
    pub program_headers_offset: u64,
    pub section_headers_offset: u64,
    pub flags: u32,
    pub header_size: u16,
    pub program_header_size: u16,
    pub program_header_entries: u16,
    pub section_header_size: u16,
    pub section_header_entries: u16,
    pub section_name_string_table_index: u16,
}

/// Section in the file.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Section {
    pub header: SectionHeader,
    pub data: Vec<u8>,
}

/// Header of a single section.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SectionHeader {
    pub name: String,
    pub name_offset: u32,
    pub section_type: u32,
    pub flags: u64,
    pub addr: u64,
    pub offset: u64,
    pub size: u64,
    pub link: u32,
    pub info: u32,
    pub addr_align: u64,
    pub entry_size: u64,
}

/// An entry in the symbol table.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SymbolTableEntry {
    pub name: String,
    pub info: u8,
    pub other: u8,
    pub section_table_index: u16,
    pub value: u64,
    pub size: u64,
}

impl ElfFile<File> {
    /// Load an `ELF` file from the file system.
    pub fn new<P: AsRef<Path>>(filename: P) -> ElfResult<ElfFile<File>> {
        let file = File::open(filename)?;
        ElfFile::from_readable(file)
    }
}

impl<'a> ElfFile<Cursor<&'a [u8]>> {
    /// Create a new `ELF` file over a byte slice.
    pub fn from_slice(target: &'a [u8]) -> ElfResult<ElfFile<Cursor<&'a [u8]>>> {
        ElfFile::from_readable(Cursor::new(target))
    }
}

impl<R> ElfFile<R> where R: Read + Seek {
    /// Create a new `ELF` file operating on a reader.
    pub fn from_readable(mut target: R) -> ElfResult<ElfFile<R>> {
        let header = parse_header(&mut target)?;
        let section_headers = parse_section_headers(header, &mut target)?;
        Ok(ElfFile { target, header, section_headers })
    }

    /// Retrieve all sections.
    pub fn sections(&mut self) -> ElfResult<Vec<Section>> {
        // Build up the sections.
        let mut sections = Vec::with_capacity(self.section_headers.len());
        for header in &self.section_headers {
            let mut data = vec![0; header.size as usize];
            self.target.seek(SeekFrom::Start(header.offset))?;
            self.target.read_exact(&mut data)?;

            let section = Section { header: header.clone(), data };
            sections.push(section);
        }

        Ok(sections)
    }

    /// Retrieve the section with a specific name if it is present.
    pub fn get_section(&mut self, name: &str) -> ElfResult<Section> {
        let header = self.get_section_header(name)?.clone();

        let mut data = vec![0; header.size as usize];
        self.target.seek(SeekFrom::Start(header.offset))?;
        self.target.read_exact(&mut data)?;

        Ok(Section { header, data })
    }

    /// Retrieve the symbols from the `.symtab` section if it is present.
    pub fn get_symbols(&mut self) -> ElfResult<Vec<SymbolTableEntry>> {
        let (size, offset) = {
            let header = self.get_section_header(".symtab")?;
            (header.size, header.offset)
        };

        let count = (size / 24) as usize;
        let mut symbols = Vec::with_capacity(count);
        let symbol_strings = self.get_section(".strtab")?.data;

        self.target.seek(SeekFrom::Start(offset))?;
        for _ in 0 .. count {
            let name_offset = self.target.read_u32::<LE>()?;
            symbols.push(SymbolTableEntry {
                name: parse_string(&symbol_strings, name_offset),
                info: self.target.read_u8()?,
                other: self.target.read_u8()?,
                section_table_index: self.target.read_u16::<LE>()?,
                value: self.target.read_u64::<LE>()?,
                size: self.target.read_u64::<LE>()?,
            })
        }

        Ok(symbols)
    }

    fn get_section_header(&mut self, name: &str) -> ElfResult<&SectionHeader> {
        self.section_headers.iter()
            .find(|header| header.name == name)
            .ok_or_else(|| ElfError::MissingSection(name.to_owned()))
    }
}

/// Parse the header of the file.
fn parse_header<R>(target: &mut R) -> ElfResult<Header> where R: Read + Seek {
    let header = Header {
        identification: {
            let mut buf = [0; 16];
            target.read_exact(&mut buf)?;
            buf
        },
        file_type: target.read_u16::<LE>()?,
        machine: target.read_u16::<LE>()?,
        version: target.read_u32::<LE>()?,
        entry: target.read_u64::<LE>()?,
        program_headers_offset: target.read_u64::<LE>()?,
        section_headers_offset: target.read_u64::<LE>()?,
        flags: target.read_u32::<LE>()?,
        header_size: target.read_u16::<LE>()?,
        program_header_size: target.read_u16::<LE>()?,
        program_header_entries: target.read_u16::<LE>()?,
        section_header_size: target.read_u16::<LE>()?,
        section_header_entries: target.read_u16::<LE>()?,
        section_name_string_table_index: target.read_u16::<LE>()?,
    };

    // Assure that this is `ELF`, 64-bit and little endian.
    // If not we don't know how to handle it and would return complete garbage.
    if (&header.identification[0..4] != b"\x7fELF")
       || (header.identification[4] != 2)
       || (header.identification[5] != 1) {
        return Err(ElfError::Invalid);
    }

    Ok(header)
}

/// Parse the section headers of the file and return the string table with it.
fn parse_section_headers<R>(header: Header, target: &mut R)
    -> ElfResult<Vec<SectionHeader>> where R: Read + Seek {
    // Read the section headers.
    target.seek(SeekFrom::Start(header.section_headers_offset))?;
    let mut headers = Vec::with_capacity(header.section_header_entries as usize);
    for _ in 0 .. header.section_header_entries {
        let header = SectionHeader {
            name: String::new(),
            name_offset: target.read_u32::<LE>()?,
            section_type: target.read_u32::<LE>()?,
            flags: target.read_u64::<LE>()?,
            addr: target.read_u64::<LE>()?,
            offset: target.read_u64::<LE>()?,
            size: target.read_u64::<LE>()?,
            link: target.read_u32::<LE>()?,
            info: target.read_u32::<LE>()?,
            addr_align: target.read_u64::<LE>()?,
            entry_size: target.read_u64::<LE>()?,
        };

        headers.push(header);
    }

    // Read the raw string table data.
    let string_index = header.section_name_string_table_index as usize;
    let string_table = &headers[string_index];
    let mut strings = vec![0; string_table.size as usize];
    target.seek(SeekFrom::Start(string_table.offset))?;
    target.read_exact(&mut strings)?;

    // Fill in the missing names for all sections.
    for table in headers.iter_mut() {
        table.name = parse_string(&strings, table.name_offset);
    }

    Ok(headers)
}

/// Parse a string from the string table.
fn parse_string(strings: &[u8], offset: u32) -> String {
    let mut zero = offset as usize;
    while strings[zero] != 0 {
        zero += 1;
    }

    CStr::from_bytes_with_nul(&strings[offset as usize .. zero + 1])
        .expect("invalid C string in elf string table")
        .to_string_lossy()
        .into_owned()
}


/// The error type for `ELF` loading.
pub enum ElfError {
    Invalid,
    MissingSection(String),
    Io(io::Error),
}

pub(in super) type ElfResult<T> = Result<T, ElfError>;

impl Display for ElfError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ElfError::Invalid => write!(f, "Invalid ELF file"),
            ElfError::MissingSection(name) => write!(f, "Missing section: {}", name),
            ElfError::Io(err) => write!(f, "I/O error: {}", err),
        }
    }
}

impl std::error::Error for ElfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ElfError::Io(err) => Some(err),
            _ => None,
        }
    }
}

debug_display!(ElfError);

impl From<io::Error> for ElfError {
    fn from(err: io::Error) -> ElfError {
        ElfError::Io(err)
    }
}
