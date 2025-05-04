use ndarray::{Array2, ArrayView2, ArrayView3};
use num_complex::Complex64;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;
use special::Bessel;
use crate::error::{FourierError, Result};

/// Pattern generation with SIMD optimization and batch processing
pub struct PatternGenerator {
    wavelength: f64,
    pixel_size: f64,
}

impl PatternGenerator {
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

    /// Generate a single pattern
    pub fn generate_pattern(
        &self,
        pattern_type: &str,
        size: (usize, usize),
        width: f64,
        frequency: f64,
        order: Option<i32>,
    ) -> Result<Array2<Complex64>> {
        if width <= 0.0 || frequency <= 0.0 {
            return Err(FourierError::InvalidInput(
                "Width and frequency must be positive".to_string(),
            ));
        }

        let (height, width) = size;
        let mut pattern = Array2::<Complex64>::zeros((height, width));

        match pattern_type {
            "cross" => self.generate_cross(&mut pattern, width, frequency)?,
            "circle" => self.generate_circle(&mut pattern, width, frequency)?,
            "square" => self.generate_square(&mut pattern, width, frequency)?,
            "grating" => self.generate_grating(&mut pattern, width, frequency)?,
            "spiral" => self.generate_spiral(&mut pattern, width, frequency)?,
            "vortex" => {
                let order = order.ok_or_else(|| {
                    FourierError::InvalidInput("Order parameter required for vortex pattern".to_string())
                })?;
                self.generate_vortex(&mut pattern, width, frequency, order)?;
            }
            "bessel" => self.generate_bessel(&mut pattern, width, frequency)?,
            "hermite" => {
                let order = order.ok_or_else(|| {
                    FourierError::InvalidInput("Order parameter required for Hermite pattern".to_string())
                })?;
                self.generate_hermite(&mut pattern, width, frequency, order)?;
            }
            "laguerre" => {
                let order = order.ok_or_else(|| {
                    FourierError::InvalidInput("Order parameter required for Laguerre pattern".to_string())
                })?;
                self.generate_laguerre(&mut pattern, width, frequency, order)?;
            }
            _ => return Err(FourierError::PatternError(format!("Unknown pattern type: {}", pattern_type))),
        }

        Ok(pattern)
    }

    /// Generate a batch of patterns
    pub fn generate_pattern_batch(
        &self,
        pattern_type: &str,
        size: (usize, usize),
        width: f64,
        frequency: f64,
        order: Option<i32>,
        batch_size: usize,
    ) -> Result<Array2<Complex64>> {
        if width <= 0.0 || frequency <= 0.0 {
            return Err(FourierError::InvalidInput(
                "Width and frequency must be positive".to_string(),
            ));
        }

        let patterns: Vec<Array2<Complex64>> = (0..batch_size)
            .into_par_iter()
            .map(|_| self.generate_pattern(pattern_type, size, width, frequency, order))
            .collect::<Result<Vec<_>>>()?;

        // Combine patterns
        let mut combined = Array2::<Complex64>::zeros(size);
        for pattern in patterns {
            combined += &pattern;
        }
        combined /= batch_size as f64;

        Ok(combined)
    }

    fn generate_cross(&self, pattern: &mut Array2<Complex64>, width: f64, frequency: f64) -> Result<()> {
        let (height, width) = pattern.dim();
        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;

        pattern.par_iter_mut().enumerate().for_each(|(i, val)| {
            let y = (i / width) as f64 - center_y;
            let x = (i % width) as f64 - center_x;
            let r = (x * x + y * y).sqrt();
            *val = Complex64::new(0.0, 2.0 * std::f64::consts::PI * frequency * r).exp();
        });

        Ok(())
    }

    fn generate_circle(&self, pattern: &mut Array2<Complex64>, width: f64, frequency: f64) -> Result<()> {
        let (height, width) = pattern.dim();
        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;
        let radius = width.min(height) as f64 * width / 2.0;

        pattern.par_iter_mut().enumerate().for_each(|(i, val)| {
            let y = (i / width) as f64 - center_y;
            let x = (i % width) as f64 - center_x;
            let r = (x * x + y * y).sqrt();
            if r <= radius {
                *val = Complex64::new(0.0, 2.0 * std::f64::consts::PI * frequency * r).exp();
            }
        });

        Ok(())
    }

    fn generate_square(&self, pattern: &mut Array2<Complex64>, width: f64, frequency: f64) -> Result<()> {
        let (height, width) = pattern.dim();
        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;
        let size = width.min(height) as f64 * width / 2.0;

        pattern.par_iter_mut().enumerate().for_each(|(i, val)| {
            let y = (i / width) as f64 - center_y;
            let x = (i % width) as f64 - center_x;
            if x.abs() <= size && y.abs() <= size {
                *val = Complex64::new(0.0, 2.0 * std::f64::consts::PI * frequency * (x * x + y * y).sqrt()).exp();
            }
        });

        Ok(())
    }

    fn generate_grating(&self, pattern: &mut Array2<Complex64>, width: f64, frequency: f64) -> Result<()> {
        let (height, width) = pattern.dim();
        let center_x = width as f64 / 2.0;

        pattern.par_iter_mut().enumerate().for_each(|(i, val)| {
            let x = (i % width) as f64 - center_x;
            *val = Complex64::new(0.0, 2.0 * std::f64::consts::PI * frequency * x).exp();
        });

        Ok(())
    }

    fn generate_spiral(&self, pattern: &mut Array2<Complex64>, width: f64, frequency: f64) -> Result<()> {
        let (height, width) = pattern.dim();
        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;

        pattern.par_iter_mut().enumerate().for_each(|(i, val)| {
            let y = (i / width) as f64 - center_y;
            let x = (i % width) as f64 - center_x;
            let theta = y.atan2(x);
            let r = (x * x + y * y).sqrt();
            *val = Complex64::new(0.0, 2.0 * std::f64::consts::PI * frequency * (r + theta)).exp();
        });

        Ok(())
    }

    fn generate_vortex(&self, pattern: &mut Array2<Complex64>, width: f64, frequency: f64, order: i32) -> Result<()> {
        let (height, width) = pattern.dim();
        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;

        pattern.par_iter_mut().enumerate().for_each(|(i, val)| {
            let y = (i / width) as f64 - center_y;
            let x = (i % width) as f64 - center_x;
            let theta = y.atan2(x);
            let r = (x * x + y * y).sqrt();
            *val = Complex64::new(0.0, order as f64 * theta + 2.0 * std::f64::consts::PI * frequency * r).exp();
        });

        Ok(())
    }

    fn generate_bessel(&self, pattern: &mut Array2<Complex64>, width: f64, frequency: f64) -> Result<()> {
        let (height, width) = pattern.dim();
        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;
        let max_r = (width * width + height * height) as f64 / 4.0;

        pattern.par_iter_mut().enumerate().for_each(|(i, val)| {
            let y = (i / width) as f64 - center_y;
            let x = (i % width) as f64 - center_x;
            let r = (x * x + y * y).sqrt();
            if r <= max_r {
                let bessel = Bessel::j0(2.0 * std::f64::consts::PI * frequency * r);
                *val = Complex64::new(bessel, 0.0);
            }
        });

        Ok(())
    }

    fn generate_hermite(&self, pattern: &mut Array2<Complex64>, width: f64, frequency: f64, order: i32) -> Result<()> {
        let (height, width) = pattern.dim();
        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;
        let sigma = width as f64 * width / 4.0;

        pattern.par_iter_mut().enumerate().for_each(|(i, val)| {
            let y = (i / width) as f64 - center_y;
            let x = (i % width) as f64 - center_x;
            let r2 = (x * x + y * y) / (2.0 * sigma);
            let hermite = self.hermite_poly(order, x / sigma.sqrt());
            *val = Complex64::new(hermite * (-r2).exp(), 0.0);
        });

        Ok(())
    }

    fn generate_laguerre(&self, pattern: &mut Array2<Complex64>, width: f64, frequency: f64, order: i32) -> Result<()> {
        let (height, width) = pattern.dim();
        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;
        let sigma = width as f64 * width / 4.0;

        pattern.par_iter_mut().enumerate().for_each(|(i, val)| {
            let y = (i / width) as f64 - center_y;
            let x = (i % width) as f64 - center_x;
            let r2 = (x * x + y * y) / (2.0 * sigma);
            let laguerre = self.laguerre_poly(order, r2);
            *val = Complex64::new(laguerre * (-r2).exp(), 0.0);
        });

        Ok(())
    }

    fn hermite_poly(&self, n: i32, x: f64) -> f64 {
        match n {
            0 => 1.0,
            1 => 2.0 * x,
            _ => {
                let mut h0 = 1.0;
                let mut h1 = 2.0 * x;
                for i in 2..=n {
                    let h2 = 2.0 * x * h1 - 2.0 * (i - 1) as f64 * h0;
                    h0 = h1;
                    h1 = h2;
                }
                h1
            }
        }
    }

    fn laguerre_poly(&self, n: i32, x: f64) -> f64 {
        match n {
            0 => 1.0,
            1 => 1.0 - x,
            _ => {
                let mut l0 = 1.0;
                let mut l1 = 1.0 - x;
                for i in 2..=n {
                    let l2 = ((2 * i - 1 - x) as f64 * l1 - (i - 1) as f64 * l0) / i as f64;
                    l0 = l1;
                    l1 = l2;
                }
                l1
            }
        }
    }
}

/// Python bindings
#[pymodule]
fn pattern_generation(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyclass]
    struct PyPatternGenerator {
        inner: PatternGenerator,
    }

    #[pymethods]
    impl PyPatternGenerator {
        #[new]
        fn new(wavelength: f64, pixel_size: f64) -> Self {
            Self {
                inner: PatternGenerator::new(wavelength, pixel_size).unwrap(),
            }
        }

        fn generate_pattern<'py>(
            &self,
            py: Python<'py>,
            pattern_type: &str,
            size: (usize, usize),
            width: f64,
            frequency: f64,
            order: Option<i32>,
        ) -> PyResult<&'py PyArray2<Complex64>> {
            let result = self.inner.generate_pattern(pattern_type, size, width, frequency, order).unwrap();
            Ok(PyArray2::from_array(py, &result))
        }

        fn generate_pattern_batch<'py>(
            &self,
            py: Python<'py>,
            pattern_type: &str,
            size: (usize, usize),
            width: f64,
            frequency: f64,
            order: Option<i32>,
            batch_size: usize,
        ) -> PyResult<&'py PyArray2<Complex64>> {
            let result = self.inner.generate_pattern_batch(pattern_type, size, width, frequency, order, batch_size).unwrap();
            Ok(PyArray2::from_array(py, &result))
        }
    }

    m.add_class::<PyPatternGenerator>()?;
    Ok(())
} 