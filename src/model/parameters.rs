use crate::error::ArkanResult;
use crate::layer::KanLayer;
use crate::network::KanNetwork;

/// Fixed-size mutable weights and bias of one checked layer.
pub struct LayerParametersMut<'a> {
    /// Spline coefficients in the existing `[output, input, basis]` order.
    pub weights: &'a mut [f32],
    /// Output biases.
    pub bias: &'a mut [f32],
}

/// Mutable parameters without access to topology, normalization or vector resizing.
pub struct ParametersMut<'a> {
    layers: &'a mut [KanLayer],
}

impl ParametersMut<'_> {
    /// Number of parameter-bearing layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }
    /// Whether there are no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
    /// Borrows each layer's weights and bias for preflight checks.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&[f32], &[f32])> {
        self.layers
            .iter()
            .map(|layer| (layer.weights.as_slice(), layer.bias.as_slice()))
    }
    /// Borrows each layer's fixed-size parameter slices for updating.
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = LayerParametersMut<'_>> {
        self.layers.iter_mut().map(|layer| LayerParametersMut {
            weights: &mut layer.weights,
            bias: &mut layer.bias,
        })
    }
}

impl KanNetwork {
    /// Validates the model, then borrows only its fixed-size parameter slices.
    ///
    /// Legacy public structural mutation is rejected before parameters are exposed.
    pub fn try_parameters_mut(&mut self) -> ArkanResult<ParametersMut<'_>> {
        self.checked_layout()?;
        Ok(ParametersMut {
            layers: &mut self.layers,
        })
    }
}
