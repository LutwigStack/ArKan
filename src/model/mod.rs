//! Checked model geometry and fixed-size parameter access.

mod layout;
mod parameters;

pub(crate) use layout::{LayerSpec, ModelLayout, Normalization};
pub use parameters::{LayerParametersMut, ParametersMut};
