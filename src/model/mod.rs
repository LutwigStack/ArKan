//! Checked model geometry and fixed-size parameter access.

pub mod config;
mod layout;
mod parameters;

pub(crate) use layout::{LayerSpec, ModelLayout, Normalization};
pub use parameters::{LayerParametersMut, ParametersMut};

use crate::cpu::KanLayer;
use crate::error::ArkanResult;
use crate::training::TrainOptions;
pub use config::*;

/// Complete KAN network with multiple layers.
///
/// This is the main struct for using ArKan. It holds the network configuration,
/// all layers, and provides methods for inference and training.
///
/// # Zero-Allocation Inference
///
/// Both `forward_single` and `forward_batch` perform zero allocations when
/// the [`crate::cpu::Workspace`] is properly sized. Create the workspace once with
/// [`create_workspace`](Self::create_workspace) and reuse it.
///
/// # Thread Safety
///
/// The network is `Send + Sync` (when `serde` feature is off). Multiple threads
/// can perform inference on the same network with separate workspaces.
pub struct KanNetwork {
    /// Network configuration.
    pub config: KanConfig,

    /// Layers: input→hidden\[0\]→...→hidden\[n\]→output.
    pub layers: Vec<KanLayer>,

    /// Immutable geometry checked against the legacy public fields.
    layout: ModelLayout,

    /// Default training options (gradient clipping, weight decay).
    pub default_train_options: TrainOptions,
}

impl KanNetwork {
    pub(crate) fn checked_layout(&self) -> ArkanResult<&ModelLayout> {
        self.layout.check_legacy(&self.config, &self.layers)?;
        Ok(&self.layout)
    }

    pub(crate) fn layout(&self) -> &ModelLayout {
        &self.layout
    }

    pub(crate) fn validate_layout(&self) -> ArkanResult<()> {
        self.checked_layout().map(|_| ())
    }

    pub(crate) fn from_parts(
        config: KanConfig,
        layers: Vec<KanLayer>,
        default_train_options: TrainOptions,
    ) -> ArkanResult<Self> {
        default_train_options.validate()?;
        let layout = ModelLayout::from_layers(&config, &layers)?;
        Ok(Self {
            config,
            layers,
            layout,
            default_train_options,
        })
    }

    /// Creates a new KAN network from configuration.
    ///
    /// Initializes all layers with random weights using Xavier initialization.
    /// Use [`KanConfig::init_seed`] for deterministic initialization.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let config = KanConfig::preset();
    /// let network = KanNetwork::new(config);
    ///
    /// assert_eq!(network.num_layers(), 3); // 2 hidden + 1 output
    /// ```
    #[must_use = "this creates a new network without modifying anything"]
    pub fn new(config: KanConfig) -> Self {
        Self::try_new(config).expect("KanNetwork::new failed")
    }

    /// Fallible constructor that validates config and size calculations.
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_new(config: KanConfig) -> ArkanResult<Self> {
        config.validate()?;

        let layer_dims = config.layer_dims();
        let mut layers = Vec::with_capacity(layer_dims.len() - 1);

        for i in 0..layer_dims.len() - 1 {
            let in_dim = layer_dims[i];
            let out_dim = layer_dims[i + 1];
            let normalization = (i == 0).then_some(Normalization {
                mean: &config.input_mean,
                std: &config.input_std,
            });
            let spec = LayerSpec::new(in_dim, out_dim, &config, normalization)?;
            layers.push(KanLayer::try_from_spec(spec, config.init_seed, i)?);
        }
        Self::from_parts(config, layers, TrainOptions::default())
    }

    /// Creates network from configuration (alias for [`new`](Self::new)).
    pub fn from_config(config: KanConfig) -> Self {
        Self::new(config)
    }

    /// Sets default training options for all subsequent `train_step` calls.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork, TrainOptions};
    ///
    /// let mut network = KanNetwork::new(KanConfig::preset());
    /// network.set_default_train_options(TrainOptions {
    ///     max_grad_norm: Some(1.0),
    ///     weight_decay: 0.01,
    /// });
    /// ```
    pub fn set_default_train_options(&mut self, opts: TrainOptions) {
        self.default_train_options = opts;
    }

    /// Returns the number of layers (hidden + output).
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns total number of trainable parameters (weights + biases).
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let network = KanNetwork::new(KanConfig::preset());
    /// println!("Parameters: {}", network.param_count()); // ~56K
    /// ```
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.param_count()).sum()
    }

    /// Sets input normalization statistics (mean and std per feature).
    ///
    /// Call this after computing statistics from your training data.
    /// Normalization is applied in the first layer only.
    ///
    /// # Arguments
    ///
    /// * `mean` - Per-feature mean values `[input_dim]`
    /// * `std` - Per-feature standard deviations `[input_dim]`
    pub fn set_input_normalization(&mut self, mean: &[f32], std: &[f32]) {
        if !self.layers.is_empty() {
            self.layers[0].set_normalization(mean, std);
        }
    }
}

impl Clone for KanNetwork {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            layers: self.layers.clone(),
            layout: self.layout.clone(),
            default_train_options: self.default_train_options,
        }
    }
}

#[cfg(test)]
mod tests;
