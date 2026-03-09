use crate::error::OxiPhotonError;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// ParamSweep — single-parameter sweep
// ---------------------------------------------------------------------------

/// Single-parameter sweep framework.
///
/// Holds a named parameter and a list of values to sweep over.  Results are
/// collected by calling [`ParamSweep::run`] with a closure.
///
/// # Examples
/// ```
/// use oxiphoton::fdtd::sweep::parameter::ParamSweep;
///
/// let sweep = ParamSweep::linspace("gap_nm", 50.0, 500.0, 10);
/// let results: Vec<f64> = sweep.run(|g| g * g);
/// assert_eq!(results.len(), 10);
/// ```
pub struct ParamSweep {
    /// Human-readable parameter name (for logging / output).
    pub param_name: String,
    /// Values to sweep over.
    pub values: Vec<f64>,
}

impl ParamSweep {
    /// Create a new sweep from an explicit list of values.
    pub fn new(name: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            param_name: name.into(),
            values,
        }
    }

    /// Create a linearly-spaced sweep: `n` points from `start` to `end` (inclusive).
    pub fn linspace(name: impl Into<String>, start: f64, end: f64, n: usize) -> Self {
        let values = linspace_vec(start, end, n);
        Self::new(name, values)
    }

    /// Create a logarithmically-spaced sweep.
    ///
    /// Points span `10^start_exp` to `10^end_exp` with `n` points (inclusive).
    pub fn logspace(name: impl Into<String>, start_exp: f64, end_exp: f64, n: usize) -> Self {
        let exps = linspace_vec(start_exp, end_exp, n);
        let values = exps.iter().map(|&e| 10.0_f64.powf(e)).collect();
        Self::new(name, values)
    }

    /// Run the sweep sequentially, collecting results.
    ///
    /// # Arguments
    /// * `f` — closure that maps a parameter value to a result
    ///
    /// # Returns
    /// Vec of results in the same order as `self.values`.
    pub fn run<F, R>(&self, f: F) -> Vec<R>
    where
        F: Fn(f64) -> R,
    {
        self.values.iter().map(|&v| f(v)).collect()
    }

    /// Run sweep in parallel using Rayon (requires the `parallel` feature).
    #[cfg(feature = "parallel")]
    pub fn run_parallel<F, R>(&self, f: F) -> Vec<R>
    where
        F: Fn(f64) -> R + Send + Sync,
        R: Send,
    {
        use rayon::prelude::*;
        self.values.par_iter().map(|&v| f(v)).collect()
    }

    /// Run sweep sequentially, passing both the index and the value to `f`.
    pub fn run_indexed<F, R>(&self, f: F) -> Vec<(f64, R)>
    where
        F: Fn(usize, f64) -> R,
    {
        self.values
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, f(i, v)))
            .collect()
    }

    /// Find the value that minimises an objective function.
    ///
    /// Evaluates `f` at every point in the sweep and returns
    /// `(minimising_value, minimum_result)`.
    ///
    /// Returns `(f64::NAN, f64::NAN)` if the sweep is empty.
    pub fn minimize<F>(&self, f: F) -> (f64, f64)
    where
        F: Fn(f64) -> f64,
    {
        self.values.iter().map(|&v| (v, f(v))).fold(
            (f64::NAN, f64::INFINITY),
            |(bv, br), (v, r)| {
                if r < br {
                    (v, r)
                } else {
                    (bv, br)
                }
            },
        )
    }

    /// Find the value that maximises an objective function.
    ///
    /// Returns `(maximising_value, maximum_result)`.
    ///
    /// Returns `(f64::NAN, f64::NEG_INFINITY)` if the sweep is empty.
    pub fn maximize<F>(&self, f: F) -> (f64, f64)
    where
        F: Fn(f64) -> f64,
    {
        self.values.iter().map(|&v| (v, f(v))).fold(
            (f64::NAN, f64::NEG_INFINITY),
            |(bv, br), (v, r)| {
                if r > br {
                    (v, r)
                } else {
                    (bv, br)
                }
            },
        )
    }
}

// ---------------------------------------------------------------------------
// ParamGrid — 2D parameter grid sweep
// ---------------------------------------------------------------------------

/// Two-dimensional parameter grid sweep.
///
/// Evaluates a closure at every point on a grid formed by the Cartesian
/// product of two parameter lists.  Results are returned as a
/// `Vec<Vec<R>>` where the outer index corresponds to `param1_values`
/// and the inner index to `param2_values`.
///
/// # Examples
/// ```
/// use oxiphoton::fdtd::sweep::parameter::ParamGrid;
///
/// let grid = ParamGrid::new(
///     "width_nm",  vec![100.0, 200.0, 300.0],
///     "height_nm", vec![50.0, 100.0],
/// );
/// let results = grid.run(|w, h| w * h);
/// assert_eq!(results.len(), 3);
/// assert_eq!(results[0].len(), 2);
/// ```
pub struct ParamGrid {
    /// Name of the first parameter.
    pub param1_name: String,
    /// Values of the first parameter.
    pub param1_values: Vec<f64>,
    /// Name of the second parameter.
    pub param2_name: String,
    /// Values of the second parameter.
    pub param2_values: Vec<f64>,
}

impl ParamGrid {
    /// Create a new 2D parameter grid.
    pub fn new(
        name1: impl Into<String>,
        values1: Vec<f64>,
        name2: impl Into<String>,
        values2: Vec<f64>,
    ) -> Self {
        Self {
            param1_name: name1.into(),
            param1_values: values1,
            param2_name: name2.into(),
            param2_values: values2,
        }
    }

    /// Run the grid sweep sequentially.
    ///
    /// # Returns
    /// `results[i][j]` = `f(param1_values[i], param2_values[j])`.
    pub fn run<F, R>(&self, f: F) -> Vec<Vec<R>>
    where
        F: Fn(f64, f64) -> R,
    {
        self.param1_values
            .iter()
            .map(|&v1| self.param2_values.iter().map(|&v2| f(v1, v2)).collect())
            .collect()
    }

    /// Run the grid sweep in parallel (requires the `parallel` feature).
    ///
    /// Row parallelism: each row of the grid (fixed `param1`) is computed
    /// in parallel across available threads.
    #[cfg(feature = "parallel")]
    pub fn run_parallel<F, R>(&self, f: F) -> Vec<Vec<R>>
    where
        F: Fn(f64, f64) -> R + Send + Sync,
        R: Send + Default + Clone,
    {
        use rayon::prelude::*;
        self.param1_values
            .par_iter()
            .map(|&v1| self.param2_values.iter().map(|&v2| f(v1, v2)).collect())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ConvergenceSweep — keeps doubling until convergence
// ---------------------------------------------------------------------------

/// Convergence test sweep.
///
/// Evaluates a function starting from `initial_value`, doubles the parameter
/// at each step, and stops when the relative change between successive results
/// is smaller than `tolerance`.  Useful for convergence studies in mesh
/// resolution, time-step count, etc.
///
/// # Examples
/// ```
/// use oxiphoton::fdtd::sweep::parameter::ConvergenceSweep;
///
/// // Riemann-sum approximation of ∫₁² 1/x² dx = 1 − 1/2 = 0.5.
/// // As n doubles the approximation converges to 0.5.
/// let sweep = ConvergenceSweep::new("n_steps", 16.0, 12, 1e-4);
/// let (converged_val, result) = sweep.run(|n| {
///     let n = n as usize;
///     let dx = 1.0 / n as f64;           // step size on [1, 2]
///     (0..n).map(|i| dx / (1.0 + i as f64 * dx).powi(2)).sum::<f64>()
/// }).expect("should converge");
/// assert!((result - 0.5).abs() < 1e-3);
/// ```
pub struct ConvergenceSweep {
    /// Human-readable parameter name.
    pub param_name: String,
    /// Starting value of the parameter.
    pub initial_value: f64,
    /// Maximum number of doublings before giving up.
    pub max_doublings: usize,
    /// Relative convergence tolerance: |Δr / r| < tolerance.
    pub tolerance: f64,
}

impl ConvergenceSweep {
    /// Create a new convergence sweep.
    pub fn new(name: impl Into<String>, initial: f64, max_doublings: usize, tol: f64) -> Self {
        Self {
            param_name: name.into(),
            initial_value: initial,
            max_doublings,
            tolerance: tol,
        }
    }

    /// Run the convergence sweep.
    ///
    /// # Returns
    /// `Ok((converged_param, result))` if convergence is achieved within
    /// `max_doublings` steps, or an `Err(OxiPhotonError::NumericalError)`
    /// if the maximum number of doublings is exceeded.
    pub fn run<F>(&self, f: F) -> Result<(f64, f64), OxiPhotonError>
    where
        F: Fn(f64) -> f64,
    {
        let mut value = self.initial_value;
        let mut prev_result = f(value);

        for _ in 0..self.max_doublings {
            value *= 2.0;
            let result = f(value);

            let rel_change = if prev_result.abs() > f64::EPSILON {
                (result - prev_result).abs() / prev_result.abs()
            } else {
                result.abs()
            };

            if rel_change < self.tolerance {
                return Ok((value, result));
            }

            prev_result = result;
        }

        Err(OxiPhotonError::NumericalError(format!(
            "ConvergenceSweep '{}': did not converge in {} doublings (tol={:.2e})",
            self.param_name, self.max_doublings, self.tolerance
        )))
    }
}

// ---------------------------------------------------------------------------
// WavelengthSweep — optical wavelength scan
// ---------------------------------------------------------------------------

/// Wavelength sweep for optical simulations.
///
/// Specifies a wavelength range in nanometres (for user convenience) but
/// stores and provides values in SI units (metres and Hz).
///
/// # Examples
/// ```
/// use oxiphoton::fdtd::sweep::parameter::WavelengthSweep;
///
/// let sweep = WavelengthSweep::new(1000.0, 1600.0, 61);
/// let lambdas_nm = sweep.wavelengths_nm();
/// assert!((lambdas_nm[0] - 1000.0).abs() < 1e-9);
/// assert!((lambdas_nm[60] - 1600.0).abs() < 1e-9);
/// ```
pub struct WavelengthSweep {
    /// Minimum wavelength in metres.
    pub lambda_min_m: f64,
    /// Maximum wavelength in metres.
    pub lambda_max_m: f64,
    /// Number of wavelength points.
    pub n_points: usize,
}

impl WavelengthSweep {
    /// Create a new wavelength sweep.
    ///
    /// # Arguments
    /// * `lambda_min_nm` — minimum wavelength in nanometres
    /// * `lambda_max_nm` — maximum wavelength in nanometres
    /// * `n_points`      — number of wavelength points (inclusive at both ends)
    pub fn new(lambda_min_nm: f64, lambda_max_nm: f64, n_points: usize) -> Self {
        Self {
            lambda_min_m: lambda_min_nm * 1e-9,
            lambda_max_m: lambda_max_nm * 1e-9,
            n_points,
        }
    }

    /// Wavelengths in metres (linearly spaced from λ_min to λ_max).
    pub fn wavelengths_m(&self) -> Vec<f64> {
        linspace_vec(self.lambda_min_m, self.lambda_max_m, self.n_points)
    }

    /// Wavelengths in nanometres.
    pub fn wavelengths_nm(&self) -> Vec<f64> {
        self.wavelengths_m().iter().map(|&l| l * 1e9).collect()
    }

    /// Frequencies in Hz: f = c / λ.
    ///
    /// Note: frequencies are *not* linearly spaced when wavelengths are.
    pub fn frequencies_hz(&self) -> Vec<f64> {
        use crate::units::conversion::SPEED_OF_LIGHT;
        self.wavelengths_m()
            .iter()
            .map(|&l| SPEED_OF_LIGHT / l)
            .collect()
    }

    /// Angular frequencies ω = 2πf in rad/s.
    pub fn angular_frequencies(&self) -> Vec<f64> {
        self.frequencies_hz()
            .iter()
            .map(|&f| 2.0 * PI * f)
            .collect()
    }

    /// Run the sweep sequentially.
    ///
    /// # Arguments
    /// * `f` — closure mapping wavelength in metres to a result
    ///
    /// # Returns
    /// Vec of `(wavelength_m, result)` pairs.
    pub fn run<F, R>(&self, f: F) -> Vec<(f64, R)>
    where
        F: Fn(f64) -> R,
    {
        self.wavelengths_m().iter().map(|&l| (l, f(l))).collect()
    }
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/// Generate `n` linearly-spaced points from `start` to `end` (inclusive).
fn linspace_vec(start: f64, end: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (end - start) / (n - 1) as f64;
            (0..n).map(|i| start + i as f64 * step).collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_param_sweep_linspace() {
        let sweep = ParamSweep::linspace("x", 1.0, 5.0, 5);
        assert_eq!(sweep.values.len(), 5);
        assert!((sweep.values[0] - 1.0).abs() < TOL);
        assert!((sweep.values[1] - 2.0).abs() < TOL);
        assert!((sweep.values[2] - 3.0).abs() < TOL);
        assert!((sweep.values[3] - 4.0).abs() < TOL);
        assert!((sweep.values[4] - 5.0).abs() < TOL);
    }

    #[test]
    fn test_param_sweep_run() {
        let sweep = ParamSweep::linspace("x", 1.0, 5.0, 5);
        let results: Vec<f64> = sweep.run(|x| x * x);
        let expected = [1.0, 4.0, 9.0, 16.0, 25.0];
        for (r, e) in results.iter().zip(expected.iter()) {
            assert!((r - e).abs() < TOL, "got {r}, expected {e}");
        }
    }

    #[test]
    fn test_param_grid_run() {
        let grid = ParamGrid::new("a", vec![1.0, 2.0, 3.0], "b", vec![10.0, 100.0, 1000.0]);
        let results = grid.run(|a, b| a * b);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].len(), 3);
        // results[i][j] = param1_values[i] * param2_values[j]
        assert!((results[0][0] - 10.0).abs() < TOL);
        assert!((results[0][1] - 100.0).abs() < TOL);
        assert!((results[1][0] - 20.0).abs() < TOL);
        assert!((results[2][2] - 3000.0).abs() < TOL);
    }

    #[test]
    fn test_wavelength_sweep() {
        let sweep = WavelengthSweep::new(1000.0, 1600.0, 61);
        let nm = sweep.wavelengths_nm();
        assert_eq!(nm.len(), 61);
        assert!((nm[0] - 1000.0).abs() < 1e-6, "min={}", nm[0]);
        assert!((nm[60] - 1600.0).abs() < 1e-6, "max={}", nm[60]);

        let m = sweep.wavelengths_m();
        assert!((m[0] - 1e-6).abs() < 1e-15);
        assert!((m[60] - 1.6e-6).abs() < 1e-15);

        // Frequencies should be c / lambda
        use crate::units::conversion::SPEED_OF_LIGHT;
        let freqs = sweep.frequencies_hz();
        let expected_f0 = SPEED_OF_LIGHT / (1000e-9);
        let expected_f1 = SPEED_OF_LIGHT / (1600e-9);
        assert!((freqs[0] - expected_f0).abs() / expected_f0 < 1e-10);
        assert!((freqs[60] - expected_f1).abs() / expected_f1 < 1e-10);
    }

    #[test]
    fn test_minimize() {
        // Minimum of f(x) = x² on the range [-1.0, 1.0] should be at x=0
        let sweep = ParamSweep::linspace("x", -1.0, 1.0, 201);
        let (best_x, best_val) = sweep.minimize(|x| x * x);
        assert!(best_val < 1e-4, "minimum value={best_val}");
        assert!(best_x.abs() < 0.02, "minimiser={best_x}");
    }

    #[test]
    fn test_convergence_sweep() {
        // Function whose value converges to π² / 6 as n → ∞  (Basel series)
        // f(n) ≈ sum_{k=1}^{n} 1/k²
        let sweep = ConvergenceSweep::new("n", 16.0, 20, 1e-3);
        let result = sweep.run(|n| {
            let n = n as usize;
            (1..=n).map(|k| 1.0 / (k as f64 * k as f64)).sum::<f64>()
        });
        assert!(result.is_ok(), "convergence failed: {:?}", result);
        let (_, val) = result.expect("convergence should succeed");
        // Basel series converges to π²/6 ≈ 1.6449
        let pi_sq_over_6 = PI * PI / 6.0;
        assert!(
            (val - pi_sq_over_6).abs() < 0.01,
            "val={val}, expected≈{pi_sq_over_6}"
        );
    }

    #[test]
    fn test_logspace() {
        let sweep = ParamSweep::logspace("freq", 0.0, 3.0, 4); // 1, 10, 100, 1000
        assert_eq!(sweep.values.len(), 4);
        assert!((sweep.values[0] - 1.0).abs() < TOL);
        assert!((sweep.values[1] - 10.0).abs() < 1e-9);
        assert!((sweep.values[2] - 100.0).abs() < 1e-7);
        assert!((sweep.values[3] - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_run_indexed() {
        let sweep = ParamSweep::linspace("t", 0.0, 2.0, 3);
        let indexed = sweep.run_indexed(|i, v| i as f64 + v);
        // values: 0.0, 1.0, 2.0; f(0,0.0)=0.0, f(1,1.0)=2.0, f(2,2.0)=4.0
        assert_eq!(indexed.len(), 3);
        assert!((indexed[0].1 - 0.0).abs() < TOL);
        assert!((indexed[1].1 - 2.0).abs() < TOL);
        assert!((indexed[2].1 - 4.0).abs() < TOL);
    }
}
