use ndarray::{Array2, ArrayView2, ArrayView3};
use num_complex::Complex64;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;
use ndarray_fft::{fft2, ifft2};
use crate::error::{FourierError, Result};
use crate::fft_pool::FFT_PROCESSOR;

/// Phase mask optimization with SIMD optimization and batch processing
pub struct PhaseMaskOptimizer {
    wavelength: f64,
    pixel_size: f64,
    learning_rate: f64,
    max_iterations: usize,
}

impl PhaseMaskOptimizer {
    pub fn new(wavelength: f64, pixel_size: f64, learning_rate: f64, max_iterations: usize) -> Result<Self> {
        if wavelength <= 0.0 || pixel_size <= 0.0 || learning_rate <= 0.0 {
            return Err(FourierError::InvalidInput(
                "Wavelength, pixel size, and learning rate must be positive".to_string(),
            ));
        }
        Ok(Self {
            wavelength,
            pixel_size,
            learning_rate,
            max_iterations,
        })
    }

    /// Optimize a single phase mask
    pub fn optimize_phase_mask(
        &self,
        target: &ArrayView2<Complex64>,
        initial_phase: Option<&ArrayView2<Complex64>>,
        distance: f64,
    ) -> Result<Array2<Complex64>> {
        if distance <= 0.0 {
            return Err(FourierError::InvalidInput("Distance must be positive".to_string()));
        }

        let (height, width) = target.dim();
        let mut phase = match initial_phase {
            Some(phase) => phase.to_owned(),
            None => Array2::<Complex64>::zeros((height, width)),
        };

        // Optimization loop
        for _ in 0..self.max_iterations {
            // Forward propagation
            let spectrum = FFT_PROCESSOR.forward_fft(&phase.view())?;
            
            // Calculate transfer function
            let k = 2.0 * std::f64::consts::PI / self.wavelength;
            let dx = self.pixel_size;
            let dy = self.pixel_size;
            
            let mut transfer = Array2::<Complex64>::zeros((height, width));
            for i in 0..height {
                for j in 0..width {
                    let fx = (j as f64 - width as f64 / 2.0) / (width as f64 * dx);
                    let fy = (i as f64 - height as f64 / 2.0) / (height as f64 * dy);
                    let f2 = fx * fx + fy * fy;
                    
                    if f2 < 1.0 / (self.wavelength * self.wavelength) {
                        let phase = k * distance * (1.0 - self.wavelength * self.wavelength * f2).sqrt();
                        transfer[[i, j]] = Complex64::new(0.0, phase).exp();
                    }
                }
            }

            // Apply transfer function
            let propagated = spectrum * transfer;
            let field = FFT_PROCESSOR.inverse_fft(&propagated.view())?;

            // Calculate error and gradient
            let error = self.calculate_error(&field, target)?;
            let gradient = self.calculate_gradient(&field, target, &propagated, &transfer)?;

            // Update phase
            phase -= self.learning_rate * gradient;

            // Normalize phase to [-π, π]
            phase.mapv_inplace(|x| {
                let phase = x.arg();
                Complex64::new(0.0, phase.max(-std::f64::consts::PI).min(std::f64::consts::PI)).exp()
            });
        }

        Ok(phase)
    }

    /// Optimize a batch of phase masks
    pub fn optimize_phase_mask_batch(
        &self,
        targets: &ArrayView3<Complex64>,
        initial_phases: Option<&ArrayView3<Complex64>>,
        distance: f64,
    ) -> Result<Array2<Complex64>> {
        if distance <= 0.0 {
            return Err(FourierError::InvalidInput("Distance must be positive".to_string()));
        }

        let (batch_size, height, width) = targets.dim();
        let phases: Vec<Array2<Complex64>> = targets
            .axis_iter(ndarray::Axis(0))
            .zip(initial_phases.map(|p| p.axis_iter(ndarray::Axis(0))).unwrap_or_else(|| {
                std::iter::repeat(ArrayView2::from_shape((height, width), &vec![Complex64::new(0.0, 0.0); height * width]).unwrap())
            }))
            .par_bridge()
            .map(|(target, initial_phase)| {
                self.optimize_phase_mask(&target, Some(&initial_phase), distance)
            })
            .collect::<Result<Vec<_>>>()?;

        // Combine phases
        let mut combined = Array2::<Complex64>::zeros((height, width));
        for phase in phases {
            combined += &phase;
        }
        combined /= batch_size as f64;

        Ok(combined)
    }

    fn calculate_error(&self, field: &Array2<Complex64>, target: &ArrayView2<Complex64>) -> Result<f64> {
        let diff = field - target;
        Ok(diff.mapv(|x| x.norm_sqr()).sum() / (field.dim().0 * field.dim().1) as f64)
    }

    fn calculate_gradient(
        &self,
        field: &Array2<Complex64>,
        target: &ArrayView2<Complex64>,
        spectrum: &Array2<Complex64>,
        transfer: &Array2<Complex64>,
    ) -> Result<Array2<Complex64>> {
        let (height, width) = field.dim();
        let mut gradient = Array2::<Complex64>::zeros((height, width));

        // Calculate gradient in frequency domain
        let error = field - target;
        let error_spectrum = FFT_PROCESSOR.forward_fft(&error.view())?;
        
        // Apply conjugate of transfer function
        let transfer_conj = transfer.mapv(|x| x.conj());
        let gradient_spectrum = error_spectrum * transfer_conj;

        // Transform back to spatial domain
        gradient = FFT_PROCESSOR.inverse_fft(&gradient_spectrum.view())?;

        // Scale gradient
        gradient *= 2.0 * Complex64::new(0.0, 1.0);

        Ok(gradient)
    }
}

/// Python bindings
#[pymodule]
fn phase_mask(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyclass]
    struct PyPhaseMaskOptimizer {
        inner: PhaseMaskOptimizer,
    }

    #[pymethods]
    impl PyPhaseMaskOptimizer {
        #[new]
        fn new(wavelength: f64, pixel_size: f64, learning_rate: f64, max_iterations: usize) -> Self {
            Self {
                inner: PhaseMaskOptimizer::new(wavelength, pixel_size, learning_rate, max_iterations).unwrap(),
            }
        }

        fn optimize_phase_mask<'py>(
            &self,
            py: Python<'py>,
            target_pattern: PyReadonlyArray2<Complex64>,
            initial_phase: Option<PyReadonlyArray2<Complex64>>,
            propagation_distance: f64,
        ) -> PyResult<&'py PyArray2<Complex64>> {
            let target = target_pattern.as_array();
            let initial = initial_phase.map(|p| p.as_array());
            let result = self.inner.optimize_phase_mask(&target, initial.as_ref(), propagation_distance).unwrap();
            Ok(PyArray2::from_array(py, &result))
        }
    }

    m.add_class::<PyPhaseMaskOptimizer>()?;
    Ok(())
} 