use ndarray::{Array2, ArrayView2, ArrayView3};
use num_complex::Complex64;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use crate::error::{FourierError, Result};
use crate::fft_pool::FFT_PROCESSOR;

/// Wave propagation methods with SIMD optimization and batch processing
pub struct WavePropagator {
    wavelength: f64,
    pixel_size: f64,
}

impl WavePropagator {
    pub fn new(wavelength: f64, pixel_size: f64) -> Result<Self> {
        if wavelength <= 0.0 || pixel_size <= 0.0 {
            return Err(FourierError::InvalidInput(
                "Wavelength and pixel size must be positive".to_string(),
            ));
        }
        Ok(Self {
            wavelength,
            pixel_size,
        })
    }

    /// Propagate a single field using angular spectrum method
    pub fn angular_spectrum(&self, field: &ArrayView2<Complex64>, distance: f64) -> Result<Array2<Complex64>> {
        if distance <= 0.0 {
            return Err(FourierError::InvalidInput("Distance must be positive".to_string()));
        }

        let (height, width) = field.dim();
        let spectrum = FFT_PROCESSOR.forward_fft(field)?;
        
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
        FFT_PROCESSOR.inverse_fft(&propagated.view())
    }

    /// Propagate a batch of fields using angular spectrum method
    pub fn angular_spectrum_batch(&self, fields: &ArrayView3<Complex64>, distance: f64) -> Result<Array2<Complex64>> {
        if distance <= 0.0 {
            return Err(FourierError::InvalidInput("Distance must be positive".to_string()));
        }

        let (batch_size, height, width) = fields.dim();
        let results: Vec<Array2<Complex64>> = fields
            .axis_iter(ndarray::Axis(0))
            .par_bridge()
            .map(|field| self.angular_spectrum(&field, distance))
            .collect::<Result<Vec<_>>>()?;

        // Combine results
        let mut combined = Array2::<Complex64>::zeros((height, width));
        for result in results {
            combined += &result;
        }
        combined /= batch_size as f64;

        Ok(combined)
    }

    /// Propagate a single field using Rayleigh-Sommerfeld method
    pub fn rayleigh_sommerfeld(&self, field: &ArrayView2<Complex64>, distance: f64) -> Result<Array2<Complex64>> {
        if distance <= 0.0 {
            return Err(FourierError::InvalidInput("Distance must be positive".to_string()));
        }

        let (height, width) = field.dim();
        let spectrum = FFT_PROCESSOR.forward_fft(field)?;
        
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
                    let amplitude = 1.0 / (1.0 - self.wavelength * self.wavelength * f2).sqrt();
                    transfer[[i, j]] = amplitude * Complex64::new(0.0, phase).exp();
                }
            }
        }

        // Apply transfer function
        let propagated = spectrum * transfer;
        FFT_PROCESSOR.inverse_fft(&propagated.view())
    }

    /// Propagate a batch of fields using Rayleigh-Sommerfeld method
    pub fn rayleigh_sommerfeld_batch(&self, fields: &ArrayView3<Complex64>, distance: f64) -> Result<Array2<Complex64>> {
        if distance <= 0.0 {
            return Err(FourierError::InvalidInput("Distance must be positive".to_string()));
        }

        let (batch_size, height, width) = fields.dim();
        let results: Vec<Array2<Complex64>> = fields
            .axis_iter(ndarray::Axis(0))
            .par_bridge()
            .map(|field| self.rayleigh_sommerfeld(&field, distance))
            .collect::<Result<Vec<_>>>()?;

        // Combine results
        let mut combined = Array2::<Complex64>::zeros((height, width));
        for result in results {
            combined += &result;
        }
        combined /= batch_size as f64;

        Ok(combined)
    }

    /// Propagate a single field using Fresnel method
    pub fn fresnel(&self, field: &ArrayView2<Complex64>, distance: f64) -> Result<Array2<Complex64>> {
        if distance <= 0.0 {
            return Err(FourierError::InvalidInput("Distance must be positive".to_string()));
        }

        let (height, width) = field.dim();
        let spectrum = FFT_PROCESSOR.forward_fft(field)?;
        
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
                
                let phase = -std::f64::consts::PI * self.wavelength * distance * f2;
                transfer[[i, j]] = Complex64::new(0.0, phase).exp();
            }
        }

        // Apply transfer function
        let propagated = spectrum * transfer;
        FFT_PROCESSOR.inverse_fft(&propagated.view())
    }

    /// Propagate a batch of fields using Fresnel method
    pub fn fresnel_batch(&self, fields: &ArrayView3<Complex64>, distance: f64) -> Result<Array2<Complex64>> {
        if distance <= 0.0 {
            return Err(FourierError::InvalidInput("Distance must be positive".to_string()));
        }

        let (batch_size, height, width) = fields.dim();
        let results: Vec<Array2<Complex64>> = fields
            .axis_iter(ndarray::Axis(0))
            .par_bridge()
            .map(|field| self.fresnel(&field, distance))
            .collect::<Result<Vec<_>>>()?;

        // Combine results
        let mut combined = Array2::<Complex64>::zeros((height, width));
        for result in results {
            combined += &result;
        }
        combined /= batch_size as f64;

        Ok(combined)
    }
}

/// Python bindings
#[pymodule]
fn wave_propagation(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyclass]
    struct PyWavePropagator {
        inner: WavePropagator,
    }

    #[pymethods]
    impl PyWavePropagator {
        #[new]
        fn new(wavelength: f64, pixel_size: f64) -> Self {
            Self {
                inner: WavePropagator::new(wavelength, pixel_size).unwrap(),
            }
        }

        fn angular_spectrum<'py>(
            &self,
            py: Python<'py>,
            field: PyReadonlyArray2<Complex64>,
            distance: f64,
        ) -> PyResult<&'py PyArray2<Complex64>> {
            let result = self.inner.angular_spectrum(&field.as_array(), distance).unwrap();
            Ok(PyArray2::from_array(py, &result))
        }

        fn rayleigh_sommerfeld<'py>(
            &self,
            py: Python<'py>,
            field: PyReadonlyArray2<Complex64>,
            distance: f64,
        ) -> PyResult<&'py PyArray2<Complex64>> {
            let result = self.inner.rayleigh_sommerfeld(&field.as_array(), distance).unwrap();
            Ok(PyArray2::from_array(py, &result))
        }

        fn fresnel<'py>(
            &self,
            py: Python<'py>,
            field: PyReadonlyArray2<Complex64>,
            distance: f64,
        ) -> PyResult<&'py PyArray2<Complex64>> {
            let result = self.inner.fresnel(&field.as_array(), distance).unwrap();
            Ok(PyArray2::from_array(py, &result))
        }
    }

    m.add_class::<PyWavePropagator>()?;
    Ok(())
} 