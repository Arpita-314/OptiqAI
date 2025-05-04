use std::sync::{Arc, Mutex};
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use fftw::plan::{C2CPlan64, Plan};
use fftw::types::{c64, Sign};
use crate::error::{FourierError, Result};

/// Thread-safe FFT plan pool
pub struct FFTPlanPool {
    plans: Arc<Mutex<Vec<(usize, usize, C2CPlan64)>>>,
}

impl FFTPlanPool {
    pub fn new() -> Self {
        Self {
            plans: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get or create an FFT plan for the given dimensions
    pub fn get_plan(&self, height: usize, width: usize) -> Result<C2CPlan64> {
        let mut plans = self.plans.lock().map_err(|_| {
            FourierError::MemoryError("Failed to lock FFT plan pool".to_string())
        })?;

        // Try to find an existing plan
        if let Some((_, _, plan)) = plans.iter().find(|(h, w, _)| *h == height && *w == width) {
            return Ok(plan.clone());
        }

        // Create a new plan
        let input = Array2::<c64>::zeros((height, width));
        let output = Array2::<c64>::zeros((height, width));
        
        let plan = C2CPlan64::aligned(&input, &output, Sign::Forward)
            .map_err(|e| FourierError::FFTError(format!("Failed to create FFT plan: {}", e)))?;

        // Store the plan
        plans.push((height, width, plan.clone()));

        Ok(plan)
    }

    /// Get or create an inverse FFT plan
    pub fn get_inverse_plan(&self, height: usize, width: usize) -> Result<C2CPlan64> {
        let mut plans = self.plans.lock().map_err(|_| {
            FourierError::MemoryError("Failed to lock FFT plan pool".to_string())
        })?;

        // Try to find an existing plan
        if let Some((_, _, plan)) = plans.iter().find(|(h, w, _)| *h == height && *w == width) {
            return Ok(plan.clone());
        }

        // Create a new plan
        let input = Array2::<c64>::zeros((height, width));
        let output = Array2::<c64>::zeros((height, width));
        
        let plan = C2CPlan64::aligned(&input, &output, Sign::Backward)
            .map_err(|e| FourierError::FFTError(format!("Failed to create inverse FFT plan: {}", e)))?;

        // Store the plan
        plans.push((height, width, plan.clone()));

        Ok(plan)
    }
}

/// SIMD-optimized FFT operations
pub struct FFTProcessor {
    pool: FFTPlanPool,
}

impl FFTProcessor {
    pub fn new() -> Self {
        Self {
            pool: FFTPlanPool::new(),
        }
    }

    /// Perform forward FFT with SIMD optimization
    pub fn forward_fft(&self, input: &ArrayView2<Complex64>) -> Result<Array2<Complex64>> {
        let (height, width) = input.dim();
        let plan = self.pool.get_plan(height, width)?;
        
        let mut output = Array2::<c64>::zeros((height, width));
        plan.c2c(input.as_slice().unwrap(), output.as_slice_mut().unwrap())
            .map_err(|e| FourierError::FFTError(format!("FFT computation failed: {}", e)))?;

        Ok(output.mapv(|x| Complex64::new(x.re, x.im)))
    }

    /// Perform inverse FFT with SIMD optimization
    pub fn inverse_fft(&self, input: &ArrayView2<Complex64>) -> Result<Array2<Complex64>> {
        let (height, width) = input.dim();
        let plan = self.pool.get_inverse_plan(height, width)?;
        
        let mut output = Array2::<c64>::zeros((height, width));
        plan.c2c(input.as_slice().unwrap(), output.as_slice_mut().unwrap())
            .map_err(|e| FourierError::FFTError(format!("Inverse FFT computation failed: {}", e)))?;

        Ok(output.mapv(|x| Complex64::new(x.re, x.im)))
    }
}

// Thread-safe singleton instance
lazy_static::lazy_static! {
    pub static ref FFT_PROCESSOR: FFTProcessor = FFTProcessor::new();
} 