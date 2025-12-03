//! Baked (quantized) model for deployment.
//!
//! This module provides a quantized, inference-only representation
//! of a trained KAN model for production deployment.
//!
//! # Features (TODO)
//! - 8-bit quantized weights
//! - Fixed-point arithmetic
//! - No allocation inference
//! - Platform-specific SIMD paths

use crate::config::KanConfig;
use crate::network::KanNetwork;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Quantized KAN model for inference.
///
/// This is a stub for future implementation.
#[doc(hidden)]
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
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.config.input_dim);
        debug_assert_eq!(output.len(), self.config.output_dim);

        // TODO: Implement quantized inference
        // For now, this is a placeholder
        output.fill(0.0);

        // Placeholder: just pass through scaled input
        for (i, &x) in input.iter().enumerate().take(output.len()) {
            output[i] = x * 0.1;
        }
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
    fn test_baked_forward() {
        let config = KanConfig::default();
        let network = KanNetwork::new(config.clone());
        let baked = BakedModel::from_network(&network);

        let input = vec![0.5f32; config.input_dim];
        let mut output = vec![0.0f32; config.output_dim];

        baked.forward(&input, &mut output);

        // Output should be finite
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}
