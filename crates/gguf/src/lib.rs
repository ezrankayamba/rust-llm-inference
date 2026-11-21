//! Memory-mapped GGUF (v2/v3) reader. Tensor data is never copied: views borrow the mapping.

pub mod writer;

use memmap2::Mmap;
use rli_core::{GgmlType, QMatrix};
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::path::Path;

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_DIMS: usize = 4;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Format(String),
    Missing(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format(m) => write!(f, "malformed GGUF: {m}"),
            Error::Missing(m) => write!(f, "missing from GGUF: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::I64(v) if v >= 0 => v as u64,
            _ => return None,
        })
    }
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => self.as_u64().map(|v| v as f32),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// GGML order: `dims[0]` is the innermost (contiguous) dimension.
    pub dims: Vec<u64>,
    pub type_id: u32,
    /// Absolute byte offset of the data within the file.
    pub offset: usize,
}

impl TensorInfo {
    pub fn ty(&self) -> Result<GgmlType> {
        GgmlType::from_id(self.type_id).ok_or_else(|| {
            Error::Format(format!(
                "tensor {} uses unsupported ggml type {}",
                self.name, self.type_id
            ))
        })
    }
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
    pub fn n_bytes(&self) -> Result<usize> {
        self.ty()?
            .checked_row_bytes(self.n_elements())
            .ok_or_else(|| Error::Format(format!("tensor {}: invalid size", self.name)))
    }

    /// Checks everything derived from `dims`, so later code can trust them: at least one
    /// dimension, none empty, whole blocks per row, no overflow. Unknown types pass, since they
    /// only fail if a model uses them.
    fn validate_shape(&self) -> Result<()> {
        let bad = |why: &str| Error::Format(format!("tensor {}: {why}", self.name));
        if self.dims.is_empty() || self.dims.len() > MAX_DIMS {
            return Err(bad("needs 1 to 4 dimensions"));
        }
        if self.dims.contains(&0) {
            return Err(bad("has an empty dimension"));
        }
        let n = self
            .dims
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| bad("element count overflows"))?;
        if let Ok(ty) = self.ty() {
            let row = usize::try_from(self.dims[0]).ok();
            if row.and_then(|r| ty.checked_row_bytes(r)).is_none()
                || ty.checked_row_bytes(n).is_none()
            {
                return Err(bad("size is not a whole number of blocks or overflows"));
            }
        }
        Ok(())
    }
}

pub struct Gguf {
    mmap: Mmap,
    pub version: u32,
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    by_name: HashMap<String, usize>,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let end = end.ok_or_else(|| Error::Format(format!("truncated at byte {}", self.pos)))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self
            .take(N)?
            .try_into()
            .expect("take returns exactly N bytes"))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn len(&mut self) -> Result<usize> {
        let n = self.u64()?;
        // Every element occupies at least one byte, so a count beyond the file is corrupt.
        if n > (self.buf.len() - self.pos) as u64 {
            return Err(Error::Format(format!("length {n} exceeds file size")));
        }
        Ok(n as usize)
    }
    fn string(&mut self) -> Result<String> {
        let n = self.len()?;
        let b = self.take(n)?;
        String::from_utf8(b.to_vec()).map_err(|_| Error::Format("non-UTF-8 string".into()))
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.arr::<1>()?[0]),
            1 => Value::I8(self.arr::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.arr()?)),
            3 => Value::I16(i16::from_le_bytes(self.arr()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.arr()?)),
            6 => Value::F32(f32::from_le_bytes(self.arr()?)),
            7 => Value::Bool(self.arr::<1>()?[0] != 0),
            8 => Value::String(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                // Models don't use these, and recursing would let a small file overflow the stack.
                if elem_ty == 9 {
                    return Err(Error::Format("nested metadata array".into()));
                }
                let n = self.len()?;
                Value::Array((0..n).map(|_| self.value(elem_ty)).collect::<Result<_>>()?)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.arr()?)),
            12 => Value::F64(f64::from_le_bytes(self.arr()?)),
            t => return Err(Error::Format(format!("unknown metadata value type {t}"))),
        })
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the mapping is read-only; mutating the file while it is open is unsupported,
        // as with every mmap-based model loader.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self> {
        let mut c = Cursor { buf: &mmap, pos: 0 };
        if c.take(4)? != MAGIC {
            return Err(Error::Format("bad magic (not a GGUF file)".into()));
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::Format(format!("unsupported GGUF version {version}")));
        }
        let n_tensors = c.len()?;
        let n_kv = c.len()?;

        // The counts are unverified, so nothing is allocated from them up front.
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            metadata.insert(key, c.value(ty)?);
        }

        let mut raw = Vec::new();
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()? as usize;
            let dims = (0..n_dims).map(|_| c.u64()).collect::<Result<Vec<_>>>()?;
            let type_id = c.u32()?;
            let rel = c.u64()?;
            raw.push((name, dims, type_id, rel));
        }

        let align = match metadata.get("general.alignment") {
            Some(v) => v
                .as_u64()
                // The spec requires a multiple of 8; GPU kernels read tensors as float and half.
                .filter(|a| a.is_power_of_two() && *a >= 8)
                .ok_or_else(|| {
                    Error::Format("general.alignment must be a power of two of at least 8".into())
                })?,
            None => DEFAULT_ALIGNMENT,
        };
        let data_start = (c.pos as u64).div_ceil(align) * align;

        let mut tensors = Vec::with_capacity(raw.len());
        let mut by_name = HashMap::with_capacity(raw.len());
        for (name, dims, type_id, rel) in raw {
            let offset = data_start
                .checked_add(rel)
                .filter(|_| rel.is_multiple_of(align))
                .and_then(|o| usize::try_from(o).ok())
                .ok_or_else(|| Error::Format(format!("tensor {name}: invalid offset {rel}")))?;
            let info = TensorInfo {
                name,
                dims,
                type_id,
                offset,
            };
            info.validate_shape()?;
            // Types we can't decode are allowed in the file; they error only if used.
            if let Ok(n) = info.n_bytes()
                && info
                    .offset
                    .checked_add(n)
                    .is_none_or(|end| end > mmap.len())
            {
                return Err(Error::Format(format!(
                    "tensor {} extends past end of file",
                    info.name
                )));
            }
            if by_name.insert(info.name.clone(), tensors.len()).is_some() {
                return Err(Error::Format(format!("duplicate tensor {}", info.name)));
            }
            tensors.push(info);
        }
        Ok(Self {
            mmap,
            version,
            metadata,
            tensors,
            by_name,
        })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    fn require(&self, key: &str) -> Result<&Value> {
        self.get(key)
            .ok_or_else(|| Error::Missing(format!("metadata key {key}")))
    }

    pub fn usize(&self, key: &str) -> Result<usize> {
        self.require(key)?
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| Error::Format(format!("{key} is not an unsigned integer")))
    }

    pub fn f32(&self, key: &str) -> Result<f32> {
        self.require(key)?
            .as_f32()
            .ok_or_else(|| Error::Format(format!("{key} is not a number")))
    }

    pub fn str(&self, key: &str) -> Result<&str> {
        self.require(key)?
            .as_str()
            .ok_or_else(|| Error::Format(format!("{key} is not a string")))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }

    pub fn tensor_data(&self, t: &TensorInfo) -> Result<&[u8]> {
        Ok(&self.mmap[t.offset..t.offset + t.n_bytes()?])
    }

    /// The whole mapped file, for backends that wrap it zero-copy (e.g. Metal shared buffers).
    pub fn mapped_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// A tensor viewed as a `rows x cols` matrix, flattening all dims past the first into rows.
    pub fn matrix(&self, name: &str) -> Result<QMatrix<'_>> {
        let t = self
            .tensor(name)
            .ok_or_else(|| Error::Missing(format!("tensor {name}")))?;
        let cols = t.dims[0] as usize;
        let rows = t.n_elements() / cols.max(1);
        Ok(QMatrix::new(t.ty()?, rows, cols, self.tensor_data(t)?))
    }

    /// An f32 copy of a small tensor (norm weights, biases).
    pub fn f32_vec(&self, name: &str) -> Result<Vec<f32>> {
        let m = self.matrix(name)?;
        let mut out = vec![0.0; m.rows * m.cols];
        for (r, chunk) in out.chunks_exact_mut(m.cols).enumerate() {
            m.dequantize_row_into(r, chunk);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::writer::Writer;
    use super::*;
    use rli_core::quant::quantize_q8_0;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("rli-gguf-{}-{name}.gguf", std::process::id()))
    }

    #[test]
    fn roundtrip_metadata_and_tensors() {
        let path = tmp_path("roundtrip");
        let w: Vec<f32> = (0..64).map(|i| i as f32 / 8.0 - 4.0).collect();
        let mut wr = Writer::new();
        wr.kv("general.architecture", Value::String("llama".into()));
        wr.kv("llama.block_count", Value::U32(2));
        wr.kv("llama.rope.freq_base", Value::F32(500000.0));
        wr.kv(
            "tokens",
            Value::Array(vec![Value::String("a".into()), Value::String("b".into())]),
        );
        wr.tensor_f32("norm.weight", &[4], &[1.0, 2.0, 3.0, 4.0]);
        wr.tensor("w.weight", &[32, 2], GgmlType::Q8_0, quantize_q8_0(&w));
        wr.write(&path).unwrap();

        let g = Gguf::open(&path).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.str("general.architecture").unwrap(), "llama");
        assert_eq!(g.usize("llama.block_count").unwrap(), 2);
        assert_eq!(g.f32("llama.rope.freq_base").unwrap(), 500000.0);
        assert_eq!(g.get("tokens").unwrap().as_array().unwrap().len(), 2);
        assert_eq!(g.f32_vec("norm.weight").unwrap(), [1.0, 2.0, 3.0, 4.0]);
        let m = g.matrix("w.weight").unwrap();
        assert_eq!((m.rows, m.cols, m.ty), (2, 32, GgmlType::Q8_0));
        assert_eq!(
            m.data.as_ptr() as usize % 32,
            g.mapped_bytes().as_ptr() as usize % 32
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_alignment_below_eight() {
        let path = tmp_path("align");
        let mut wr = Writer::new();
        wr.kv("general.alignment", Value::U32(4));
        wr.write(&path).unwrap();
        assert!(matches!(Gguf::open(&path), Err(Error::Format(_))));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        let path = tmp_path("bad");
        std::fs::write(&path, b"GGUX\x03\0\0\0").unwrap();
        assert!(matches!(Gguf::open(&path), Err(Error::Format(_))));
        std::fs::write(&path, b"GGUF\x03\0\0\0\x01").unwrap();
        assert!(matches!(Gguf::open(&path), Err(Error::Format(_))));
        std::fs::remove_file(path).unwrap();
    }

    const F32: u32 = 0;
    const Q8_0: u32 = GgmlType::Q8_0.id();

    /// A GGUF file from raw parts: `kv` is already encoded, `tensors` are (name, dims, type,
    /// relative offset), and `data` zero bytes follow the aligned header.
    fn raw(n_kv: u64, kv: &[u8], tensors: &[(&str, &[u64], u32, u64)], data: usize) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend((tensors.len() as u64).to_le_bytes());
        b.extend(n_kv.to_le_bytes());
        b.extend(kv);
        for (name, dims, ty, rel) in tensors {
            b.extend((name.len() as u64).to_le_bytes());
            b.extend(name.as_bytes());
            b.extend((dims.len() as u32).to_le_bytes());
            dims.iter().for_each(|d| b.extend(d.to_le_bytes()));
            b.extend(ty.to_le_bytes());
            b.extend(rel.to_le_bytes());
        }
        b.resize(b.len().div_ceil(32) * 32 + data, 0);
        b
    }

    fn assert_malformed(name: &str, bytes: &[u8]) {
        let path = tmp_path(name);
        std::fs::write(&path, bytes).unwrap();
        let r = Gguf::open(&path);
        std::fs::remove_file(path).unwrap();
        assert!(
            matches!(r, Err(Error::Format(_))),
            "{name} was not rejected"
        );
    }

    #[test]
    fn rejects_nested_metadata_arrays() {
        let mut kv = 1u64.to_le_bytes().to_vec();
        kv.extend(b"k");
        kv.extend(9u32.to_le_bytes());
        // Deep enough to overflow the stack if each level recurses.
        for _ in 0..500_000 {
            kv.extend(9u32.to_le_bytes());
            kv.extend(1u64.to_le_bytes());
        }
        kv.extend(0u32.to_le_bytes());
        kv.extend(0u64.to_le_bytes());
        assert_malformed("nested", &raw(1, &kv, &[], 0));
    }

    #[test]
    fn rejects_elements_not_in_whole_blocks() {
        let t: &[(&str, &[u64], u32, u64)] = &[("t", &[33], Q8_0, 0)];
        assert_malformed("blocks", &raw(0, &[], t, 64));
    }

    #[test]
    fn rejects_rows_not_in_whole_blocks() {
        let t: &[(&str, &[u64], u32, u64)] = &[("t", &[16, 2], Q8_0, 0)];
        assert_malformed("rows", &raw(0, &[], t, 64));
    }

    #[test]
    fn rejects_tensors_without_dimensions() {
        let t: &[(&str, &[u64], u32, u64)] = &[("t", &[], F32, 0)];
        assert_malformed("no-dims", &raw(0, &[], t, 64));
    }

    #[test]
    fn rejects_zero_sized_dimensions() {
        let t: &[(&str, &[u64], u32, u64)] = &[("t", &[0, 4], F32, 0)];
        assert_malformed("zero-dim", &raw(0, &[], t, 64));
    }

    #[test]
    fn rejects_too_many_dimensions() {
        let t: &[(&str, &[u64], u32, u64)] = &[("t", &[1, 1, 1, 1, 1], F32, 0)];
        assert_malformed("many-dims", &raw(0, &[], t, 64));
    }

    #[test]
    fn rejects_element_counts_that_overflow() {
        // 3 * this wraps around to 2 elements, which would fit in the file.
        let t: &[(&str, &[u64], u32, u64)] = &[("t", &[3, 6_148_914_691_236_517_206], F32, 0)];
        assert_malformed("elements", &raw(0, &[], t, 64));
    }

    #[test]
    fn rejects_byte_sizes_that_overflow() {
        // 4 bytes each wraps around to 0 bytes.
        let t: &[(&str, &[u64], u32, u64)] = &[("t", &[1 << 62], F32, 0)];
        assert_malformed("bytes", &raw(0, &[], t, 64));
    }

    #[test]
    fn rejects_offsets_that_overflow() {
        let t: &[(&str, &[u64], u32, u64)] = &[("t", &[4], F32, u64::MAX - 31)];
        assert_malformed("offset", &raw(0, &[], t, 64));
    }

    #[test]
    fn rejects_unaligned_offsets() {
        let t: &[(&str, &[u64], u32, u64)] = &[("t", &[4], F32, 4)];
        assert_malformed("unaligned", &raw(0, &[], t, 64));
    }

    #[test]
    fn rejects_duplicate_tensor_names() {
        let t: &[(&str, &[u64], u32, u64)] = &[("t", &[4], F32, 0), ("t", &[4], F32, 32)];
        assert_malformed("duplicate", &raw(0, &[], t, 64));
    }

    #[test]
    fn accepts_a_well_formed_raw_file() {
        let path = tmp_path("raw-ok");
        let t: &[(&str, &[u64], u32, u64)] = &[("a", &[4], F32, 0), ("b", &[32, 2], Q8_0, 32)];
        std::fs::write(&path, raw(0, &[], t, 32 + 68)).unwrap();
        let g = Gguf::open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(g.tensor("b").unwrap().n_bytes().unwrap(), 68);
    }

    #[test]
    fn unknown_tensor_types_load_and_fail_only_when_used() {
        let path = tmp_path("unknown-type");
        let t: &[(&str, &[u64], u32, u64)] = &[("t", &[33], 9999, 0)];
        std::fs::write(&path, raw(0, &[], t, 64)).unwrap();
        let g = Gguf::open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(g.tensor("t").unwrap().n_bytes().is_err());
    }
}
