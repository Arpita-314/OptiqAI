mod wave_propagation;
mod pattern_generation;
mod phase_mask;

pub use wave_propagation::*;
pub use pattern_generation::*;
pub use phase_mask::*;

#[pymodule]
fn fourierlab_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    wave_propagation::wave_propagation(_py, m)?;
    pattern_generation::pattern_generation(_py, m)?;
    phase_mask::phase_mask(_py, m)?;
    Ok(())
} 