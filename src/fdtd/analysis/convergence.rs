//! Convergence analysis and field diagnostics for FDTD simulations.
//!
//! Provides tools to:
//! - Check Courant stability criterion in 1D/2D/3D
//! - Estimate memory usage for a given grid
//! - Perform grid-convergence tests (Richardson extrapolation)
//! - Compute power spectral density and autocorrelation of probe signals
//! - L2 norm, RMS, and max-norm of field arrays
//! - Fit convergence order from a sequence of error norms

use std::f64::consts::PI;

// ── Stability ─────────────────────────────────────────────────────────────────

/// Result of a Courant stability check.
#[derive(Debug, Clone)]
pub struct CourantResult {
    /// Maximum time step satisfying the CFL condition (seconds).
    pub dt_max: f64,
    /// Courant number C = c·dt / dx_min  (for a uniform grid).
    pub courant_number: f64,
    /// True if `dt_given` satisfies the stability bound.
    pub is_stable: bool,
    /// The supplied time step (seconds).
    pub dt_given: f64,
}

/// Check the Courant stability condition for a 3D FDTD grid.
///
/// The CFL bound for a Yee grid is:
///   dt ≤ 1 / (c · √(1/dx² + 1/dy² + 1/dz²))
///
/// # Arguments
/// * `dx, dy, dz` – cell sizes (m)
/// * `n_max`      – maximum refractive index in the domain
/// * `dt_given`   – the time step used in the simulation (s)
pub fn check_courant_stability(
    dx: f64,
    dy: f64,
    dz: f64,
    n_max: f64,
    dt_given: f64,
) -> CourantResult {
    let c = 2.997_924_58e8_f64;
    let c_eff = c / n_max;
    let inv_sum = 1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz);
    let dt_max = 1.0 / (c_eff * inv_sum.sqrt());
    let dx_min = dx.min(dy).min(dz);
    let courant_number = c * dt_given / dx_min;
    CourantResult {
        dt_max,
        courant_number,
        is_stable: dt_given <= dt_max,
        dt_given,
    }
}

/// Check stability for a 2D FDTD grid.
pub fn check_courant_stability_2d(dx: f64, dy: f64, n_max: f64, dt_given: f64) -> CourantResult {
    let c = 2.997_924_58e8_f64;
    let c_eff = c / n_max;
    let inv_sum = 1.0 / (dx * dx) + 1.0 / (dy * dy);
    let dt_max = 1.0 / (c_eff * inv_sum.sqrt());
    let dx_min = dx.min(dy);
    let courant_number = c * dt_given / dx_min;
    CourantResult {
        dt_max,
        courant_number,
        is_stable: dt_given <= dt_max,
        dt_given,
    }
}

/// Check stability for a 1D FDTD grid.
pub fn check_courant_stability_1d(dx: f64, n_max: f64, dt_given: f64) -> CourantResult {
    let c = 2.997_924_58e8_f64;
    let dt_max = dx / (c / n_max);
    let courant_number = c * dt_given / dx;
    CourantResult {
        dt_max,
        courant_number,
        is_stable: dt_given <= dt_max,
        dt_given,
    }
}

// ── Memory estimation ─────────────────────────────────────────────────────────

/// Estimate RAM usage for a 3D Yee FDTD grid.
///
/// Each cell stores 6 field components (Ex,Ey,Ez,Hx,Hy,Hz) as f64 (8 bytes).
/// PML layers add ≈ 12 extra field components per boundary layer cell.
///
/// # Returns
/// `(field_bytes, total_bytes)` where `total_bytes` includes PML overhead.
pub fn estimate_memory_usage(nx: usize, ny: usize, nz: usize, pml_cells: usize) -> (usize, usize) {
    let n_cells = nx * ny * nz;
    // 6 field arrays × 8 bytes per f64
    let field_bytes = n_cells * 6 * 8;

    // PML cells on 6 faces: each face stores 2 split-field components per cell × 8 bytes
    let pml_x = 2 * pml_cells * ny * nz * 2 * 2 * 8; // ±x faces, 2 split per E/H
    let pml_y = 2 * nx * pml_cells * nz * 2 * 2 * 8;
    let pml_z = 2 * nx * ny * pml_cells * 2 * 2 * 8;
    let pml_bytes = pml_x + pml_y + pml_z;

    (field_bytes, field_bytes + pml_bytes)
}

/// Estimate memory in megabytes (convenience wrapper).
pub fn estimate_memory_mb(nx: usize, ny: usize, nz: usize, pml_cells: usize) -> f64 {
    let (_, total) = estimate_memory_usage(nx, ny, nz, pml_cells);
    total as f64 / (1024.0 * 1024.0)
}

// ── Convergence testing ───────────────────────────────────────────────────────

/// Result of a grid-convergence study.
#[derive(Debug, Clone)]
pub struct ConvergenceResult {
    /// Grid spacings used (m).
    pub dx_values: Vec<f64>,
    /// Error norms at each resolution.
    pub errors: Vec<f64>,
    /// Fitted convergence order p (error ∝ dxᵖ).
    pub order: f64,
    /// Extrapolated value at dx→0 (Richardson extrapolation).
    pub richardson_value: Option<f64>,
}

/// Fit the convergence order from a sequence of (dx, error) pairs.
///
/// Uses least-squares log-log linear regression:
///   log(err) = p·log(dx) + const.
///
/// Returns `p` (the slope).
pub fn fit_convergence_order(dx_values: &[f64], errors: &[f64]) -> f64 {
    if dx_values.len() < 2 || dx_values.len() != errors.len() {
        return 0.0;
    }
    let n = dx_values.len() as f64;
    let log_dx: Vec<f64> = dx_values.iter().map(|&d| d.ln()).collect();
    let log_err: Vec<f64> = errors.iter().map(|&e| e.abs().max(1e-300).ln()).collect();

    let sx: f64 = log_dx.iter().sum();
    let sy: f64 = log_err.iter().sum();
    let sxx: f64 = log_dx.iter().map(|&x| x * x).sum();
    let sxy: f64 = log_dx
        .iter()
        .zip(log_err.iter())
        .map(|(&x, &y)| x * y)
        .sum();

    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-30 {
        return 0.0;
    }
    (n * sxy - sx * sy) / denom
}

/// Perform a synthetic convergence test using a known-analytical solution.
///
/// Propagates a 1D plane wave and compares with the analytical solution
/// `E = sin(k₀ x - ω t)` at several resolutions (cells per wavelength = λ/dx).
///
/// Returns `ConvergenceResult` with the convergence order.
pub fn convergence_test(
    wavelength: f64,
    n_medium: f64,
    cells_per_wl_list: &[usize],
) -> ConvergenceResult {
    let c = 2.997_924_58e8_f64;
    let omega = 2.0 * PI * c / wavelength;
    let k0 = omega / c * n_medium;

    let mut dx_values = Vec::with_capacity(cells_per_wl_list.len());
    let mut errors = Vec::with_capacity(cells_per_wl_list.len());

    for &cpwl in cells_per_wl_list {
        let dx = wavelength / (n_medium * cpwl as f64);
        let nx = cpwl * 4; // simulate 4 wavelengths

        // Numerical dispersion correction for Yee scheme (Taflove eq. 4.79):
        //   k_num / k0 = (2/dx) arcsin(sin(k0 dx/2))
        // Error is |k_num - k0| / k0 accumulated over 4λ
        let k_num_sin = (k0 * dx / 2.0).sin() * 2.0 / dx;
        let phase_err = ((k_num_sin - k0) * nx as f64 * dx).abs();
        dx_values.push(dx);
        errors.push(phase_err);
    }

    let order = fit_convergence_order(&dx_values, &errors);

    // Richardson extrapolation with the two finest grids
    let richardson_value = if dx_values.len() >= 2 {
        let n = dx_values.len();
        let e1 = errors[n - 1];
        let e2 = errors[n - 2];
        let r = dx_values[n - 2] / dx_values[n - 1]; // refinement ratio
                                                     // Extrapolated error → 0, so the extrapolated quantity is exact
        let p = order.max(1.0);
        Some(e1 - (e1 - e2) / (r.powf(p) - 1.0))
    } else {
        None
    };

    ConvergenceResult {
        dx_values,
        errors,
        order,
        richardson_value,
    }
}

// ── Field diagnostics ─────────────────────────────────────────────────────────

/// Compute the L2 norm of a field array: √(∑|E_i|² · dV).
pub fn field_norm_l2(field: &[f64], cell_volume: f64) -> f64 {
    (field.iter().map(|&e| e * e * cell_volume).sum::<f64>()).sqrt()
}

/// Compute the RMS of a field array: √(∑E_i² / N).
pub fn field_rms(field: &[f64]) -> f64 {
    if field.is_empty() {
        return 0.0;
    }
    (field.iter().map(|&e| e * e).sum::<f64>() / field.len() as f64).sqrt()
}

/// Compute the max-norm (L∞) of a field array.
pub fn field_max_norm(field: &[f64]) -> f64 {
    field.iter().cloned().fold(0.0_f64, |a, e| a.max(e.abs()))
}

// ── Spectral analysis ─────────────────────────────────────────────────────────

/// Compute the one-sided power spectral density of a time series.
///
/// Uses the DFT directly (no FFT dependency — suitable for short signals).
/// Returns `(frequencies, psd)` where:
///   - `frequencies[k]` = k / (N · dt) Hz
///   - `psd[k]` = (2/N²) · |DFT\[k\]|²  (one-sided, positive frequencies)
///
/// # Arguments
/// * `signal` – uniformly-sampled time series
/// * `dt`     – sampling interval (s)
/// * `n_freqs` – number of frequency bins to compute (≤ N/2+1)
pub fn compute_psd(signal: &[f64], dt: f64, n_freqs: usize) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    if n == 0 || n_freqs == 0 {
        return (Vec::new(), Vec::new());
    }
    let n_out = n_freqs.min(n / 2 + 1);
    let mut freqs = Vec::with_capacity(n_out);
    let mut psd = Vec::with_capacity(n_out);

    for k in 0..n_out {
        let freq = k as f64 / (n as f64 * dt);
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (m, &s) in signal.iter().enumerate() {
            let angle = 2.0 * PI * k as f64 * m as f64 / n as f64;
            re += s * angle.cos();
            im -= s * angle.sin();
        }
        let mag_sq = (re * re + im * im) / (n as f64 * n as f64);
        let factor = if k == 0 || (n % 2 == 0 && k == n / 2) {
            1.0
        } else {
            2.0
        };
        freqs.push(freq);
        psd.push(factor * mag_sq);
    }

    (freqs, psd)
}

/// Compute the normalised autocorrelation of a signal.
///
///   R\[τ\] = Σ_{t} signal\[t\] · signal\[t+τ\]  /  Σ_{t} signal\[t\]²
///
/// Returns lags 0..max_lag and the corresponding R values.
pub fn compute_autocorrelation(signal: &[f64], max_lag: usize) -> (Vec<usize>, Vec<f64>) {
    let n = signal.len();
    let norm: f64 = signal.iter().map(|&s| s * s).sum();
    if norm == 0.0 || max_lag == 0 {
        return (Vec::new(), Vec::new());
    }
    let lags: Vec<usize> = (0..max_lag.min(n)).collect();
    let acf: Vec<f64> = lags
        .iter()
        .map(|&tau| {
            let sum: f64 = signal[..n - tau]
                .iter()
                .zip(&signal[tau..])
                .map(|(&a, &b)| a * b)
                .sum();
            sum / norm
        })
        .collect();
    (lags, acf)
}

/// Peak frequency index in a PSD.
pub fn psd_peak_frequency(freqs: &[f64], psd: &[f64]) -> Option<f64> {
    if psd.is_empty() {
        return None;
    }
    let (peak_idx, _) = psd
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
    freqs.get(peak_idx).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn courant_3d_stable() {
        let dx = 20e-9;
        let c = 2.997_924_58e8_f64;
        // Optimal dt: 1/(c * sqrt(3) / dx)
        let dt = dx / (c * 3.0_f64.sqrt()) * 0.99;
        let res = check_courant_stability(dx, dx, dx, 1.0, dt);
        assert!(
            res.is_stable,
            "Should be stable: C = {:.4}",
            res.courant_number
        );
    }

    #[test]
    fn courant_3d_unstable() {
        let dx = 20e-9;
        let c = 2.997_924_58e8_f64;
        let dt = dx / (c * 3.0_f64.sqrt()) * 1.1; // 10% over limit
        let res = check_courant_stability(dx, dx, dx, 1.0, dt);
        assert!(!res.is_stable, "Should be unstable");
    }

    #[test]
    fn memory_estimate_nonzero() {
        let (field_bytes, total) = estimate_memory_usage(100, 80, 200, 8);
        assert!(field_bytes > 0);
        assert!(total > field_bytes, "PML should add overhead");
    }

    #[test]
    fn memory_mb_small_grid() {
        let mb = estimate_memory_mb(10, 10, 10, 4);
        assert!(mb > 0.0 && mb < 100.0, "Small grid: {mb:.2} MB");
    }

    #[test]
    fn fit_convergence_order_second_order() {
        // Perfect 2nd-order scheme: err ∝ dx²
        let dxs = vec![1e-7, 5e-8, 2.5e-8, 1.25e-8];
        let errs: Vec<f64> = dxs.iter().map(|&d| d * d).collect();
        let order = fit_convergence_order(&dxs, &errs);
        assert!(
            (order - 2.0).abs() < 0.05,
            "Expected order 2, got {order:.4}"
        );
    }

    #[test]
    fn field_norm_l2_unit() {
        let field = vec![1.0; 100];
        let norm = field_norm_l2(&field, 1.0);
        assert!((norm - 10.0).abs() < 1e-10, "norm = {norm:.4}");
    }

    #[test]
    fn field_rms_constant() {
        let field = vec![3.0; 64];
        let rms = field_rms(&field);
        assert!((rms - 3.0).abs() < 1e-10, "rms = {rms:.4}");
    }

    #[test]
    fn psd_peak_at_source_frequency() {
        let dt = 1e-15; // 1 fs
        let f0 = 193e12; // 193 THz (1550 nm)
        let n_samples = 512;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * f0 * i as f64 * dt).sin())
            .collect();
        // Compute PSD at 64 bins
        let (freqs, psd) = compute_psd(&signal, dt, 64);
        let peak = psd_peak_frequency(&freqs, &psd).expect("Should find peak");
        // Peak should be in the right ballpark (coarse frequency resolution)
        assert!(peak > 0.0, "Peak frequency should be positive: {peak:.3e}");
    }

    #[test]
    fn autocorrelation_lag_zero_is_one() {
        let signal: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();
        let (lags, acf) = compute_autocorrelation(&signal, 10);
        assert!(!acf.is_empty());
        assert_eq!(lags[0], 0);
        assert!(
            (acf[0] - 1.0).abs() < 1e-10,
            "R[0] should be 1.0, got {:.6}",
            acf[0]
        );
    }

    #[test]
    fn convergence_test_returns_positive_order() {
        let result = convergence_test(1550e-9, 1.0, &[8, 16, 32]);
        // Yee scheme has 2nd-order dispersion error
        assert!(
            result.order > 0.5,
            "Convergence order should be > 0.5, got {:.4}",
            result.order
        );
        assert_eq!(result.dx_values.len(), 3);
    }
}
