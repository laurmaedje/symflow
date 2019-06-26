//! Parsing of the ELF file format.
//!
//! (Currently assumes 64-bit little endian).

use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::fmt::{self, Display, Formatter};
use std::ffi::CStr;
use byteorder::{ReadBytesExt, LE};


/// The header of an ELF file.
#[derive(Debug, Copy, Clone)]
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

/// The section header table.
#[derive(Debug, Clone)]
pub struct SectionHeader {
    pub name: Option<String>,
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

/// A section in the file.
#[derive(Debug, Clone)]
pub struct Section {
    pub header: SectionHeader,
    pub data: Vec<u8>,
}

/// Parses an ELF file.
#[derive(Debug)]
pub struct ElfFile<R> where R: Read + Seek {
    target: R,
    pub header: Header,
    pub section_headers: Vec<SectionHeader>,
}

impl<R> ElfFile<R> where R: Read + Seek {
    /// Create a new ELF file operating on a reader.
    pub fn new(mut target: R) -> ElfResult<ElfFile<R>> {
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
        let header = self.section_headers.iter().find(|header| match &header.name {
            Some(n) => n == name,
            None => false,
        }).ok_or_else(|| ElfError::MissingSection(name.to_owned()))?;

        let mut data = vec![0; header.size as usize];
        self.target.seek(SeekFrom::Start(header.offset))?;
        self.target.read_exact(&mut data)?;

        Ok(Section { header: header.clone(), data })
    }
}

impl<'a> ElfFile<Cursor<&'a [u8]>> {
    /// Create a new ELF reader operating on a byte slice.
    pub fn from_slice(target: &'a [u8]) -> ElfResult<ElfFile<Cursor<&'a [u8]>>> {
        ElfFile::new(Cursor::new(target))
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

    // Assure that this is ELF, 64-bit and little endian.
    // If not we don't know how to handle it and would return complete garbage.
    assert_eq!(&header.identification[0..4], b"\x7fELF");
    assert_eq!(header.identification[4], 2);
    assert_eq!(header.identification[5], 1);

    Ok(header)
}

/// Parse the section headers of the file.
fn parse_section_headers<R>(header: Header, target: &mut R)
    -> ElfResult<Vec<SectionHeader>> where R: Read + Seek {
    // Read the section headers.
    target.seek(SeekFrom::Start(header.section_headers_offset))?;
    let mut headers = Vec::with_capacity(header.section_header_entries as usize);
    for _ in 0 .. header.section_header_entries {
        let header = SectionHeader {
            name: None,
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
        table.name = if table.name_offset != 0 {
            let start = table.name_offset as usize;
            let mut zero = start;
            while strings[zero] != 0 {
                zero += 1;
            }

            let name_str = CStr::from_bytes_with_nul(&strings[start .. zero + 1]).unwrap();
            Some(name_str.to_string_lossy().into_owned())
        } else {
            None
        };
    }

    Ok(headers)
}


/// The error type for ELF loading.
#[derive(Debug)]
pub enum ElfError {
    MissingSection(String),
    Io(io::Error),
}

type ElfResult<T> = Result<T, ElfError>;
impl std::error::Error for ElfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ElfError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for ElfError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ElfError::MissingSection(name) => write!(f, "missing section: {}", name),
            ElfError::Io(err) => write!(f, "io error: {}", err),
        }
    }
}

impl From<io::Error> for ElfError {
    fn from(err: io::Error) -> ElfError {
        ElfError::Io(err)
    }
}
