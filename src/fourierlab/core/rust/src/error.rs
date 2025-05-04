use std::fmt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

#[derive(Debug)]
pub enum FourierError {
    InvalidInput(String),
    ComputationError(String),
    MemoryError(String),
    FFTError(String),
    PatternError(String),
    OptimizationError(String),
}

impl fmt::Display for FourierError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FourierError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            FourierError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            FourierError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            FourierError::FFTError(msg) => write!(f, "FFT error: {}", msg),
            FourierError::PatternError(msg) => write!(f, "Pattern generation error: {}", msg),
            FourierError::OptimizationError(msg) => write!(f, "Optimization error: {}", msg),
        }
    }
}

impl std::error::Error for FourierError {}

impl From<FourierError> for PyErr {
    fn from(err: FourierError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

// Helper macro for creating errors
#[macro_export]
macro_rules! fourier_err {
    ($kind:ident, $msg:expr) => {
        Err(FourierError::$kind($msg.to_string()))
    };
} 