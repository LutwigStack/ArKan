use crate::baked::{BakedLayer, BakedModel};
use crate::config::KanConfig;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Deserialize)]
struct BakedV2 {
    config: KanConfig,
    layers: Vec<BakedLayer>,
    uncalibrated: bool,
}

#[derive(Serialize)]
#[serde(rename = "BakedModel")]
struct BakedV2Ref<'a> {
    config: &'a KanConfig,
    layers: &'a [BakedLayer],
    uncalibrated: bool,
}

impl Serialize for BakedModel {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        BakedV2Ref {
            config: &self.config,
            layers: &self.layers,
            uncalibrated: self.uncalibrated,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BakedModel {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let record = BakedV2::deserialize(deserializer)?;
        let model = Self {
            config: record.config,
            layers: record.layers,
            uncalibrated: record.uncalibrated,
        };
        model.validate().map_err(serde::de::Error::custom)?;
        Ok(model)
    }
}

impl BakedModel {
    /// Serializes the baked model to a self-describing byte vector.
    ///
    /// Layout:
    /// ```text
    /// [0..12]  magic   — MAGIC_BAKED (b"KAN_BAKED_v1")
    /// [12..16] version — u32 little-endian format version (currently 2)
    /// [16..]   body    — bincode-encoded BakedModel
    /// ```
    ///
    /// Requires the `serde` feature.
    #[cfg(feature = "serde")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        use crate::MAGIC_BAKED;

        let body_len = usize::try_from(bincode::serialized_size(self)?)
            .map_err(|_| bincode::Error::from(bincode::ErrorKind::SizeLimit))?;
        let total_len = (MAGIC_BAKED.len() + 4)
            .checked_add(body_len)
            .ok_or_else(|| bincode::Error::from(bincode::ErrorKind::SizeLimit))?;
        let mut out = Vec::with_capacity(total_len);
        out.extend_from_slice(MAGIC_BAKED);
        out.extend_from_slice(&Self::FORMAT_VERSION.to_le_bytes());
        bincode::serialize_into(&mut out, self)?;
        Ok(out)
    }

    /// Deserializes a baked model from bytes produced by [`BakedModel::to_bytes`].
    ///
    /// Returns a clear `Err` — never panics — for:
    /// - Input shorter than the 16-byte header.
    /// - Wrong magic bytes (not an ArKan baked-model file).
    /// - Wrong format version (produced by a different library version).
    /// - Corrupt bincode body or invalid executable shapes, scales or metadata.
    ///
    /// Requires the `serde` feature.
    #[cfg(feature = "serde")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        use crate::MAGIC_BAKED;

        let header_len = MAGIC_BAKED.len() + 4; // 12 + 4 = 16

        if bytes.len() < header_len {
            return Err(Box::new(bincode::ErrorKind::Custom(format!(
                "BakedModel::from_bytes: input too short ({} bytes, need at least {})",
                bytes.len(),
                header_len
            ))));
        }

        let (magic_bytes, rest) = bytes.split_at(MAGIC_BAKED.len());
        if magic_bytes != MAGIC_BAKED.as_ref() {
            return Err(Box::new(bincode::ErrorKind::Custom(format!(
                "BakedModel::from_bytes: wrong magic bytes (got {:?}, expected {:?}). \
                 Is this an ArKan baked-model file?",
                magic_bytes, MAGIC_BAKED
            ))));
        }

        let version = u32::from_le_bytes(rest[..4].try_into().unwrap());
        if version != Self::FORMAT_VERSION {
            return Err(Box::new(bincode::ErrorKind::Custom(format!(
                "BakedModel::from_bytes: format version mismatch (got {}, expected {}). \
                 Re-bake the model with the current library version.",
                version,
                Self::FORMAT_VERSION
            ))));
        }

        bincode::deserialize(&rest[4..])
    }
}
