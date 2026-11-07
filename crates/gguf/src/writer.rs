//! Minimal GGUF v3 writer, used to build synthetic models for tests.

use crate::Value;
use rli_core::GgmlType;
use std::path::Path;

const ALIGN: usize = 32;

#[derive(Default)]
pub struct Writer {
    kvs: Vec<(String, Value)>,
    tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)>,
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

fn type_id(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn put_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.push(*x as u8),
        Value::U16(x) => out.extend(x.to_le_bytes()),
        Value::I16(x) => out.extend(x.to_le_bytes()),
        Value::U32(x) => out.extend(x.to_le_bytes()),
        Value::I32(x) => out.extend(x.to_le_bytes()),
        Value::U64(x) => out.extend(x.to_le_bytes()),
        Value::I64(x) => out.extend(x.to_le_bytes()),
        Value::F32(x) => out.extend(x.to_le_bytes()),
        Value::F64(x) => out.extend(x.to_le_bytes()),
        Value::Bool(x) => out.push(*x as u8),
        Value::String(s) => put_str(out, s),
        Value::Array(items) => {
            out.extend(items.first().map_or(4, type_id).to_le_bytes());
            out.extend((items.len() as u64).to_le_bytes());
            items.iter().for_each(|i| put_value(out, i));
        }
    }
}

impl Writer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kv(&mut self, key: &str, v: Value) -> &mut Self {
        self.kvs.push((key.into(), v));
        self
    }

    pub fn tensor(&mut self, name: &str, dims: &[u64], ty: GgmlType, data: Vec<u8>) -> &mut Self {
        let n = dims.iter().product::<u64>() as usize;
        assert_eq!(
            data.len(),
            ty.row_bytes(n),
            "tensor {name}: data size mismatch"
        );
        self.tensors.push((name.into(), dims.to_vec(), ty, data));
        self
    }

    pub fn tensor_f32(&mut self, name: &str, dims: &[u64], data: &[f32]) -> &mut Self {
        self.tensor(
            name,
            dims,
            GgmlType::F32,
            data.iter().flat_map(|f| f.to_le_bytes()).collect(),
        )
    }

    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut out = Vec::new();
        out.extend(b"GGUF");
        out.extend(3u32.to_le_bytes());
        out.extend((self.tensors.len() as u64).to_le_bytes());
        out.extend((self.kvs.len() as u64).to_le_bytes());
        for (k, v) in &self.kvs {
            put_str(&mut out, k);
            out.extend(type_id(v).to_le_bytes());
            put_value(&mut out, v);
        }
        let mut offset = 0usize;
        for (name, dims, ty, data) in &self.tensors {
            put_str(&mut out, name);
            out.extend((dims.len() as u32).to_le_bytes());
            dims.iter().for_each(|d| out.extend(d.to_le_bytes()));
            out.extend(ty.id().to_le_bytes());
            out.extend((offset as u64).to_le_bytes());
            offset = (offset + data.len()).div_ceil(ALIGN) * ALIGN;
        }
        out.resize(out.len().div_ceil(ALIGN) * ALIGN, 0);
        for (_, _, _, data) in &self.tensors {
            out.extend(data);
            out.resize(out.len().div_ceil(ALIGN) * ALIGN, 0);
        }
        std::fs::write(path, out)
    }
}
