//! Parsing of the ELF file format.
//!
//! Currently assumes 64-bit little endian.

use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::fmt::{self, Display, Formatter};
use std::ffi::CStr;
use byteorder::{ReadBytesExt, LE};


/// The header of an ELF file.
#[derive(Debug, Copy, Clone)]
pub struct ElfHeader {
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
#[derive(Debug, Copy, Clone)]
pub struct SectionHeader {
    pub name: u32,
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
    pub name: Option<String>,
    pub data: Vec<u8>,
}

/// Parses an ELF file.
#[derive(Debug)]
pub struct ElfReader<R> where R: Read + Seek {
    target: R,
    header: Option<ElfHeader>,
    section_headers: Option<Vec<SectionHeader>>,
}

impl<R> ElfReader<R> where R: Read + Seek {
    /// Create a new ELF reader.
    pub fn new(target: R) -> ElfReader<R> {
        ElfReader { target, header: None, section_headers: None }
    }

    /// Parse the header of the file.
    pub fn header(&mut self) -> ElfResult<ElfHeader> {
        if let Some(header) = self.header {
            Ok(header)
        } else {
            let header = ElfHeader {
                identification: {
                    let mut buf = [0; 16];
                    self.target.read_exact(&mut buf)?;
                    buf
                },
                file_type: self.target.read_u16::<LE>()?,
                machine: self.target.read_u16::<LE>()?,
                version: self.target.read_u32::<LE>()?,
                entry: self.target.read_u64::<LE>()?,
                program_headers_offset: self.target.read_u64::<LE>()?,
                section_headers_offset: self.target.read_u64::<LE>()?,
                flags: self.target.read_u32::<LE>()?,
                header_size: self.target.read_u16::<LE>()?,
                program_header_size: self.target.read_u16::<LE>()?,
                program_header_entries: self.target.read_u16::<LE>()?,
                section_header_size: self.target.read_u16::<LE>()?,
                section_header_entries: self.target.read_u16::<LE>()?,
                section_name_string_table_index: self.target.read_u16::<LE>()?,
            };

            // Assure that this is ELF, 64-bit and little endian.
            // If not we don't know how to handle it and would return complete garbage.
            assert_eq!(&header.identification[0..4], b"\x7fELF");
            assert_eq!(header.identification[4], 2);
            assert_eq!(header.identification[5], 1);

            self.header = Some(header);
            Ok(header)
        }
    }

    /// Parse the section headers of the file.
    pub fn section_headers(&mut self) -> ElfResult<&[SectionHeader]> {
        if let Some(ref headers) = self.section_headers {
            Ok(headers)
        } else {
            let header = self.header()?;

            // Read the section headers.
            self.target.seek(SeekFrom::Start(header.section_headers_offset))?;
            let mut headers = Vec::with_capacity(header.section_header_entries as usize);
            for _ in 0 .. header.section_header_entries {
                let header = SectionHeader {
                    name: self.target.read_u32::<LE>()?,
                    section_type: self.target.read_u32::<LE>()?,
                    flags: self.target.read_u64::<LE>()?,
                    addr: self.target.read_u64::<LE>()?,
                    offset: self.target.read_u64::<LE>()?,
                    size: self.target.read_u64::<LE>()?,
                    link: self.target.read_u32::<LE>()?,
                    info: self.target.read_u32::<LE>()?,
                    addr_align: self.target.read_u64::<LE>()?,
                    entry_size: self.target.read_u64::<LE>()?,
                };

                headers.push(header);
            }

            self.section_headers = Some(headers);
            Ok(self.section_headers.as_ref().unwrap())
        }
    }

    /// Retrieve all sections.
    pub fn sections(&mut self) -> ElfResult<Vec<Section>> {
        let header = self.header()?;
        let headers = self.section_headers()?.to_vec();

        // Read the raw string table data.
        let string_table = headers[header.section_name_string_table_index as usize];
        let mut strings = vec![0; string_table.size as usize];
        self.target.seek(SeekFrom::Start(string_table.offset))?;
        self.target.read_exact(&mut strings)?;

        // Build up the sections.
        let mut sections = Vec::with_capacity(headers.len());
        for table in headers {
            let mut data = vec![0; table.size as usize];
            self.target.seek(SeekFrom::Start(table.offset))?;
            self.target.read_exact(&mut data)?;

            let name = if table.name != 0 {
                let start = table.name as usize;
                let mut zero = start;
                while strings[zero] != 0 {
                    zero += 1;
                }

                let name_str = CStr::from_bytes_with_nul(&strings[start .. zero + 1]).unwrap();
                Some(name_str.to_string_lossy().into_owned())
            } else {
                None
            };

            let section = Section { name, data };
            sections.push(section);
        }

        Ok(sections)
    }

    /// Retrieve the section with a specific name if it is present.
    pub fn get_section(&mut self, name: &str) -> ElfResult<Section> {
        self.sections()?.into_iter().find(|s| match &s.name {
            Some(n) => n == name,
            None => false,
        }).ok_or_else(|| ElfError::MissingSection(name.to_owned()))
    }
}

impl<'a> ElfReader<Cursor<&'a [u8]>> {
    /// Create a new ELF reader working on a byte slice.
    pub fn from_slice(target: &'a [u8]) -> ElfReader<Cursor<&'a [u8]>> {
        ElfReader::new(Cursor::new(target))
    }
}

/// The result type for elf loading.
pub type ElfResult<T> = Result<T, ElfError>;

/// The error type for elf loading.
#[derive(Debug)]
pub enum ElfError {
    MissingSection(String),
    Io(io::Error),
}

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
