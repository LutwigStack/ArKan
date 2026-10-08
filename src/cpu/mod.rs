//! CPU kernels, inference orchestration and reusable execution scratch.

mod inference;
pub mod layer;
pub mod workspace;

pub use layer::KanLayer;
pub use workspace::{Workspace, WorkspaceGuard};
