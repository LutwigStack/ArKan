use crate::config::KanConfig;
use crate::layer::KanLayer;
use crate::network::{KanNetwork, TrainOptions};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Magic bytes for serialized network files.
///
/// Used to identify ArKan model files and distinguish from other formats.
#[cfg(feature = "serde")]
pub(crate) const SERIALIZATION_MAGIC: &[u8; 5] = b"ARKAN";

/// Current serialization format version.
///
/// Incremented when the format changes in a backwards-incompatible way.
/// - v1: Initial versioned format (ArKan 0.3.0+)
#[cfg(feature = "serde")]
pub(crate) const SERIALIZATION_VERSION: u32 = 1;

#[derive(Deserialize)]
struct NetworkV1 {
    config: KanConfig,
    layers: Vec<KanLayer>,
    layer_dims: Vec<usize>,
    layer_param_sizes: Vec<(usize, usize)>,
    default_train_options: TrainOptions,
}

#[derive(Serialize)]
#[serde(rename = "KanNetwork")]
struct NetworkV1Ref<'a> {
    config: &'a KanConfig,
    layers: &'a [KanLayer],
    layer_dims: &'a [usize],
    layer_param_sizes: &'a [(usize, usize)],
    default_train_options: &'a TrainOptions,
}

impl Serialize for KanNetwork {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        NetworkV1Ref {
            config: &self.config,
            layers: &self.layers,
            layer_dims: self.layout().layer_dims(),
            layer_param_sizes: self.layout().parameter_sizes(),
            default_train_options: &self.default_train_options,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for KanNetwork {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let record = NetworkV1::deserialize(deserializer)?;
        // Legacy caches are advisory. Checked reconstruction owns runtime geometry.
        let _ = (record.layer_dims, record.layer_param_sizes);
        Self::from_parts(record.config, record.layers, record.default_train_options)
            .map_err(serde::de::Error::custom)
    }
}

impl KanNetwork {
    /// Saves network to bytes using bincode with version header.
    ///
    /// The format includes:
    /// - Magic bytes: `ARKAN` (5 bytes)
    /// - Version: u32 (4 bytes)
    /// - Network data: bincode serialized
    ///
    /// Requires the `serde` feature.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let network = KanNetwork::new(KanConfig::preset());
    /// let bytes = network.to_bytes().unwrap();
    ///
    /// // Bytes start with magic header "ARKAN"
    /// assert_eq!(&bytes[..5], b"ARKAN");
    /// ```
    #[cfg(feature = "serde")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        let body_len = usize::try_from(bincode::serialized_size(self)?)
            .map_err(|_| bincode::Error::from(bincode::ErrorKind::SizeLimit))?;
        let total_len = (SERIALIZATION_MAGIC.len() + 4)
            .checked_add(body_len)
            .ok_or_else(|| bincode::Error::from(bincode::ErrorKind::SizeLimit))?;
        let mut bytes = Vec::with_capacity(total_len);
        bytes.extend_from_slice(SERIALIZATION_MAGIC);
        bytes.extend_from_slice(&SERIALIZATION_VERSION.to_le_bytes());
        bincode::serialize_into(&mut bytes, self)?;
        Ok(bytes)
    }

    /// Loads network from bytes with version validation.
    ///
    /// Requires the `serde` feature.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Magic bytes don't match
    /// - Version is incompatible
    /// - Deserialization fails
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// // Create network and serialize
    /// let original = KanNetwork::new(KanConfig::preset());
    /// let bytes = original.to_bytes().unwrap();
    ///
    /// // Deserialize
    /// let loaded = KanNetwork::from_bytes(&bytes).unwrap();
    /// assert_eq!(loaded.param_count(), original.param_count());
    /// ```
    #[cfg(feature = "serde")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        const HEADER_SIZE: usize = SERIALIZATION_MAGIC.len() + 4; // magic + version

        if bytes.len() < HEADER_SIZE {
            return Err(bincode::Error::from(bincode::ErrorKind::Custom(
                "Invalid file: too short for header".to_string(),
            )));
        }

        // Check magic bytes
        if &bytes[..SERIALIZATION_MAGIC.len()] != SERIALIZATION_MAGIC {
            return Err(bincode::Error::from(bincode::ErrorKind::Custom(
                "Invalid file: wrong magic bytes (not an ArKan model)".to_string(),
            )));
        }

        // Check version
        let version_bytes: [u8; 4] = bytes[SERIALIZATION_MAGIC.len()..HEADER_SIZE]
            .try_into()
            .unwrap();
        let version = u32::from_le_bytes(version_bytes);

        if version != SERIALIZATION_VERSION {
            return Err(bincode::Error::from(bincode::ErrorKind::Custom(format!(
                "Incompatible model version: expected {}, got {}",
                SERIALIZATION_VERSION, version
            ))));
        }

        // Deserialize network data
        bincode::deserialize(&bytes[HEADER_SIZE..])
    }

    /// Loads network from bytes without version check (legacy format).
    ///
    /// Use this to load models saved with ArKan < 0.3.0.
    ///
    /// # Warning
    ///
    /// This method is provided for backwards compatibility only.
    /// New code should use [`from_bytes`](Self::from_bytes).
    #[cfg(feature = "serde")]
    pub fn from_bytes_legacy(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}
