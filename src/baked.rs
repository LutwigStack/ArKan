//! Baked (quantized) model for deployment.
//!
//! This module provides a quantized, inference-only representation
//! of a trained KAN model for production deployment.
//!
//! # Status: Planned for v0.4.0
//!
//! This module is currently a stub. Full implementation is planned for v0.4.0.
//!
//! # Planned Features
//! - 8-bit quantized weights
//! - Fixed-point arithmetic
//! - No allocation inference
//! - Platform-specific SIMD paths
//!
//! For now, use [`KanNetwork`](crate::KanNetwork) directly for inference.

use crate::config::KanConfig;
use crate::network::KanNetwork;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Quantized KAN model for inference.
///
/// This is a stub - full implementation planned for v0.4.0.
/// Use [`KanNetwork`](crate::KanNetwork) for inference until then.
#[doc(hidden)]
#[deprecated(
    since = "0.2.0",
    note = "BakedModel is a stub, planned for v0.4.0. Use KanNetwork directly for inference."
)]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BakedModel {
    /// Original configuration for validation
    pub config: KanConfig,

    /// Quantized weights (TODO: implement proper quantization)
    /// For now, stores f32 weights compressed
    weights_data: Vec<u8>,

    /// Scale factors for dequantization
    scales: Vec<f32>,

    /// Bias terms (kept as f32 for precision)
    biases: Vec<f32>,
}

#[allow(deprecated)]
impl BakedModel {
    /// Bakes a trained KAN network into quantized form.
    ///
    /// # Arguments
    /// * `network` - Trained KAN network to bake
    ///
    /// # Returns
    /// Quantized model ready for inference
    pub fn from_network(network: &KanNetwork) -> Self {
        // TODO: Implement proper quantization
        // For now, just store compressed f32 weights

        let mut weights_data = Vec::new();
        let mut scales = Vec::new();
        let mut biases = Vec::new();

        for layer in &network.layers {
            // Find scale for this layer's weights
            let max_weight = layer.weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
            let scale = if max_weight > 0.0 {
                127.0 / max_weight
            } else {
                1.0
            };
            scales.push(scale);

            // Quantize weights to i8
            for &w in &layer.weights {
                let q = (w * scale).round() as i8;
                weights_data.push(q as u8);
            }

            // Keep biases as f32
            biases.extend(&layer.bias);
        }

        Self {
            config: network.config.clone(),
            weights_data,
            scales,
            biases,
        }
    }

    /// Runs inference on the baked model.
    ///
    /// # Arguments
    /// * `input` - Input features [input_dim]
    /// * `output` - Output buffer [output_dim]
    ///
    /// # Panics
    ///
    /// Always panics - BakedModel inference is not yet implemented.
    /// Use `KanNetwork::forward_single` or `KanNetwork::forward_batch` instead.
    #[allow(unused_variables)]
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        unimplemented!(
            "BakedModel::forward() is not implemented. \
             Use KanNetwork::forward_single() or KanNetwork::forward_batch() instead."
        );
    }

    /// Returns the size of the baked model in bytes.
    pub fn size_bytes(&self) -> usize {
        self.weights_data.len() + self.scales.len() * 4 + self.biases.len() * 4
    }

    /// Saves baked model to bytes.
    #[cfg(feature = "serde")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Loads baked model from bytes.
    #[cfg(feature = "serde")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn test_bake_model() {
        let config = KanConfig::default();
        let network = KanNetwork::new(config);

        let baked = BakedModel::from_network(&network);

        // Baked model should have compressed weights
        assert!(baked.size_bytes() > 0);
        assert!(!baked.weights_data.is_empty());
        assert!(!baked.scales.is_empty());
    }

    #[test]
    #[should_panic(expected = "BakedModel::forward() is not implemented")]
    fn test_baked_forward() {
        let config = KanConfig::default();
        let network = KanNetwork::new(config.clone());
        let baked = BakedModel::from_network(&network);

        let input = vec![0.5f32; config.input_dim];
        let mut output = vec![0.0f32; config.output_dim];

        // This should panic with unimplemented!
        baked.forward(&input, &mut output);
    }
}
