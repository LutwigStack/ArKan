use crate::layer::KanLayer;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Deserialize)]
pub(crate) struct LayerRecord {
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
    pub(crate) order: usize,
    pub(crate) grid_size: usize,
    pub(crate) global_basis_size: usize,
    pub(crate) local_basis_size: usize,
    pub(crate) basis_aligned: usize,
    pub(crate) grid_range: (f32, f32),
    pub(crate) mean: Vec<f32>,
    pub(crate) std: Vec<f32>,
    pub(crate) weights: Vec<f32>,
    pub(crate) bias: Vec<f32>,
    pub(crate) simd_width: usize,
}

#[derive(Serialize)]
#[serde(rename = "KanLayer")]
struct LayerRecordRef<'a> {
    in_dim: usize,
    out_dim: usize,
    order: usize,
    grid_size: usize,
    global_basis_size: usize,
    local_basis_size: usize,
    basis_aligned: usize,
    grid_range: (f32, f32),
    mean: &'a [f32],
    std: &'a [f32],
    weights: &'a [f32],
    bias: &'a [f32],
    simd_width: usize,
}

impl Serialize for KanLayer {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        LayerRecordRef {
            in_dim: self.in_dim,
            out_dim: self.out_dim,
            order: self.order,
            grid_size: self.grid_size,
            global_basis_size: self.global_basis_size,
            local_basis_size: self.local_basis_size,
            basis_aligned: self.basis_aligned,
            grid_range: self.grid_range,
            mean: &self.mean,
            std: &self.std,
            weights: &self.weights,
            bias: &self.bias,
            simd_width: self.wire_simd_width(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for KanLayer {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::from_record(LayerRecord::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}
