use crate::config::{ConfigError, KanConfig};
use crate::error::{ArkanError, ArkanResult};
use crate::layer::KanLayer;

/// Borrowed, explicit per-feature statistics; identity is selected by the builder.
#[derive(Clone, Copy)]
pub(crate) struct Normalization<'a> {
    pub(crate) mean: &'a [f32],
    pub(crate) std: &'a [f32],
}

pub(crate) struct LayerSpec<'a> {
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
    pub(crate) order: usize,
    pub(crate) grid_size: usize,
    pub(crate) grid_range: (f32, f32),
    pub(crate) simd_width: usize,
    pub(crate) normalization: Option<Normalization<'a>>,
}

impl<'a> LayerSpec<'a> {
    pub(crate) fn new(
        in_dim: usize,
        out_dim: usize,
        config: &KanConfig,
        normalization: Option<Normalization<'a>>,
    ) -> ArkanResult<Self> {
        if in_dim == 0 || out_dim == 0 {
            return Err(
                ConfigError::InvalidDimension("layer dimensions must be positive".into()).into(),
            );
        }
        config.validate_layer_config()?;
        if let Some(stats) = normalization {
            if stats.mean.len() != in_dim || stats.std.len() != in_dim {
                return Err(ArkanError::cpu(
                    "Layer normalization dimensions do not match",
                ));
            }
        }
        Ok(Self {
            in_dim,
            out_dim,
            order: config.spline_order,
            grid_size: config.grid_size,
            grid_range: config.grid_range,
            simd_width: config.simd_width,
            normalization,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerLayout {
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
    pub(crate) order: usize,
    pub(crate) grid_size: usize,
    pub(crate) grid_range: (f32, f32),
    pub(crate) global_basis: usize,
    local_basis: usize,
    aligned_basis: usize,
}

impl LayerLayout {
    fn of(layer: &KanLayer) -> Self {
        Self {
            in_dim: layer.in_dim,
            out_dim: layer.out_dim,
            order: layer.order,
            grid_size: layer.grid_size,
            grid_range: layer.grid_range,
            global_basis: layer.global_basis_size,
            local_basis: layer.local_basis_size,
            aligned_basis: layer.basis_aligned,
        }
    }
}

/// Constructed once; legacy mutable metadata is checked against this snapshot.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ModelLayout {
    dimensions: Vec<usize>,
    parameter_sizes: Vec<(usize, usize)>,
    layers: Vec<LayerLayout>,
    simd_width: usize,
}

impl ModelLayout {
    pub(crate) fn from_layers(config: &KanConfig, layers: &[KanLayer]) -> ArkanResult<Self> {
        config.validate()?;
        let layout = Self {
            dimensions: config.layer_dims(),
            parameter_sizes: layers
                .iter()
                .map(|l| (l.weights.len(), l.bias.len()))
                .collect(),
            layers: layers.iter().map(LayerLayout::of).collect(),
            simd_width: config.simd_width,
        };
        layout.check_legacy(config, layers)?;
        Ok(layout)
    }

    pub(crate) fn check_legacy(&self, config: &KanConfig, layers: &[KanLayer]) -> ArkanResult<()> {
        config.validate()?;
        if layers.len() != self.layers.len()
            || layers.len() != config.hidden_dims.len() + 1
            || self.dimensions.len() != layers.len() + 1
            || self.dimensions.first() != Some(&config.input_dim)
            || self.dimensions.last() != Some(&config.output_dim)
            || self.dimensions[1..self.dimensions.len() - 1] != config.hidden_dims
            || self.simd_width != config.simd_width
        {
            return Err(ArkanError::cpu(
                "Network topology no longer matches its layout",
            ));
        }
        for (i, (layer, layout)) in layers.iter().zip(&self.layers).enumerate() {
            layer.validate_layout()?;
            if LayerLayout::of(layer) != *layout
                || layer.in_dim != self.dimensions[i]
                || layer.out_dim != self.dimensions[i + 1]
                || layer.order != config.spline_order
                || layer.grid_size != config.grid_size
                || layer.grid_range != config.grid_range
                || layer.basis_aligned
                    != layer.local_basis_size.div_ceil(config.simd_width) * config.simd_width
                || (layer.weights.len(), layer.bias.len()) != self.parameter_sizes[i]
            {
                return Err(ArkanError::cpu(
                    "Layer topology no longer matches its network",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn layer_dims(&self) -> &[usize] {
        &self.dimensions
    }
    pub(crate) fn parameter_sizes(&self) -> &[(usize, usize)] {
        &self.parameter_sizes
    }
    #[cfg(feature = "gpu")]
    pub(crate) fn layers(&self) -> &[LayerLayout] {
        &self.layers
    }
}
