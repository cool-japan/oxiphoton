//! Adaptive optics control algorithms.
//!
//! Provides:
//! - [`IntegralController`]: Classic leaky integrator AO controller
//! - [`ModalController`]: Controller operating in Zernike modal space
//! - [`PredictiveController`]: Linear predictor for frozen-flow turbulence
//! - [`ClosedLoopMetrics`]: Performance metrics for the closed AO loop
//!
//! # Mathematical Background
//!
//! ## Interaction Matrix (IM)
//! The IM `D` maps actuator commands to sensor slopes:
//!   s = D * c
//!
//! ## Control Matrix (CM)
//! The CM `M†` is the pseudo-inverse of `D`, obtained by SVD truncation
//! of the smallest singular values to exclude poorly-controlled modes:
//!   M† = V * Σ†_trunc * U^T
//!
//! ## Integral Controller
//!   c(n+1) = leak * c(n) - gain * M† * s(n)
//!
//! ## Predictive Controller (Frozen Flow)
//!   c(n+1) = A * c(n)  where A is the learned AR(1) transition matrix.

use crate::error::OxiPhotonError;

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = 2.0 * PI;

// ─────────────────────────────────────────────────────────────────────────────
// SVD helpers (pure Rust, Golub-Reinsch bidiagonalisation)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the thin SVD of a matrix A (m×n, m >= n).
///
/// Returns `(U, sigma, Vt)` where
/// - `U` is `m × n` (column-major: column k = U\[k\], length m)
/// - `sigma` is n singular values, descending
/// - `Vt` is `n × n`, row-major (Vt\[k\] = k-th row = right singular vector)
///
/// Uses the one-sided Jacobi algorithm which is numerically stable for
/// small to medium matrices as used in AO control (~100×100).
#[allow(clippy::needless_range_loop)]
pub fn svd_thin(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let m = a.len();
    if m == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let n = a[0].len();

    // Work with column vectors of A: A = [a0 | a1 | … | a_{n-1}], each of length m.
    // One-sided Jacobi: orthogonalise the columns by Jacobi rotations on V.
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();
    // V starts as identity (n×n).
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0_f64; n];
            row[i] = 1.0;
            row
        })
        .collect();

    // Jacobi sweeps.
    let max_iter = 30 * n * n;
    for _ in 0..max_iter {
        let mut converged = true;
        for p in 0..n {
            for q in p + 1..n {
                // Compute inner products.
                let cpp: f64 = cols[p].iter().map(|&x| x * x).sum();
                let cqq: f64 = cols[q].iter().map(|&x| x * x).sum();
                let cpq: f64 = cols[p]
                    .iter()
                    .zip(cols[q].iter())
                    .map(|(&x, &y)| x * y)
                    .sum();

                let tol = 1e-14 * (cpp * cqq).sqrt();
                if cpq.abs() <= tol {
                    continue;
                }
                converged = false;

                // Jacobi rotation angle.
                let tau = (cqq - cpp) / (2.0 * cpq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let c_rot = 1.0 / (1.0 + t * t).sqrt();
                let s_rot = t * c_rot;

                // Rotate columns p and q of A.
                for i in 0..m {
                    let ap = cols[p][i];
                    let aq = cols[q][i];
                    cols[p][i] = c_rot * ap + s_rot * aq;
                    cols[q][i] = -s_rot * ap + c_rot * aq;
                }
                // Rotate columns p and q of V.
                for i in 0..n {
                    let vp = v[i][p];
                    let vq = v[i][q];
                    v[i][p] = c_rot * vp + s_rot * vq;
                    v[i][q] = -s_rot * vp + c_rot * vq;
                }
            }
        }
        if converged {
            break;
        }
    }

    // Extract singular values and normalise columns to get U.
    let mut sigma: Vec<f64> = (0..n)
        .map(|j| cols[j].iter().map(|&x| x * x).sum::<f64>().sqrt())
        .collect();

    let mut u: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            let s = sigma[j];
            if s > 1e-30 {
                cols[j].iter().map(|&x| x / s).collect()
            } else {
                vec![0.0; m]
            }
        })
        .collect();

    // Sort descending by singular value.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a_idx, &b_idx| {
        sigma[b_idx]
            .partial_cmp(&sigma[a_idx])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sigma_sorted: Vec<f64> = order.iter().map(|&i| sigma[i]).collect();
    let u_sorted: Vec<Vec<f64>> = order.iter().map(|&i| u[i].clone()).collect();
    // Vt[k] = v_k^T = row k of V^T = column k of V → v[][k].
    let vt_sorted: Vec<Vec<f64>> = order
        .iter()
        .map(|&i| (0..n).map(|j| v[j][i]).collect())
        .collect();

    sigma = sigma_sorted;
    u = u_sorted;

    (u, sigma, vt_sorted)
}

/// Compute the pseudo-inverse of A using SVD, keeping only `n_modes` modes.
///
/// Returns M† (n_act × n_slopes) = V * Σ†_trunc * U^T.
pub fn pseudo_inverse(a: &[Vec<f64>], n_modes: usize) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 {
        return Vec::new();
    }
    let n = a[0].len();

    let (u, sigma, vt) = svd_thin(a);
    let n_sv = sigma.len().min(n_modes);

    // M† = V * Σ†_trunc * U^T
    // V is n×n_sv (columns of Vt transposed), Σ† is n_sv diagonal,
    // U^T is n_sv×m.
    // Result is n×m.
    let mut pinv = vec![vec![0.0_f64; m]; n];
    for k in 0..n_sv {
        let s = sigma[k];
        if s < 1e-14 {
            break;
        }
        let inv_s = 1.0 / s;
        // V[:,k] = vt[k]^T (length n).
        // U[:,k] = u[k] (length m).
        for i in 0..n {
            for j in 0..m {
                pinv[i][j] += vt[k][i] * inv_s * u[k][j];
            }
        }
    }
    pinv
}

// ─────────────────────────────────────────────────────────────────────────────
// IntegralController
// ─────────────────────────────────────────────────────────────────────────────

/// Classic leaky integrator AO controller.
///
/// Implements:
///   c(n+1) = leak · c(n) − gain · M† · s(n)
///
/// where M† is the control matrix (pseudo-inverse of the interaction matrix).
#[derive(Debug, Clone)]
pub struct IntegralController {
    /// Loop gain (typically 0.1–0.5).
    pub gain: f64,
    /// Leak factor (typically 0.99–1.0; <1 prevents wind-up).
    pub leak: f64,
    /// Current actuator commands.
    pub commands: Vec<f64>,
    /// Interaction matrix `D` of shape `[n_slopes][n_actuators]`.
    pub interaction_matrix: Vec<Vec<f64>>,
    /// Control matrix `M†` of shape `[n_actuators][n_slopes]`.
    pub control_matrix: Vec<Vec<f64>>,
    /// Number of Zernike/SVD modes included in the control matrix.
    pub n_modes_corrected: usize,
    /// Number of actuators.
    pub n_actuators: usize,
    /// Number of slope measurements.
    pub n_slopes: usize,
}

impl IntegralController {
    /// Create a new integral controller.
    ///
    /// # Arguments
    /// * `gain` — loop gain
    /// * `n_actuators` — number of DM actuators
    /// * `n_slopes` — number of WFS slope measurements
    pub fn new(gain: f64, n_actuators: usize, n_slopes: usize) -> Self {
        Self {
            gain,
            leak: 0.99,
            commands: vec![0.0; n_actuators],
            interaction_matrix: vec![vec![0.0; n_actuators]; n_slopes],
            control_matrix: vec![vec![0.0; n_slopes]; n_actuators],
            n_modes_corrected: n_actuators.min(n_slopes),
            n_actuators,
            n_slopes,
        }
    }

    /// Set the interaction matrix `D` (shape `[n_slopes][n_actuators]`).
    ///
    /// This defines the DM-to-WFS response. After setting this, call
    /// `compute_control_matrix` to update M†.
    pub fn set_interaction_matrix(&mut self, mat: Vec<Vec<f64>>) -> Result<(), OxiPhotonError> {
        if mat.len() != self.n_slopes {
            return Err(OxiPhotonError::NumericalError(format!(
                "Interaction matrix has {} rows, expected {}",
                mat.len(),
                self.n_slopes
            )));
        }
        if !mat.is_empty() && mat[0].len() != self.n_actuators {
            return Err(OxiPhotonError::NumericalError(format!(
                "Interaction matrix has {} columns, expected {}",
                mat[0].len(),
                self.n_actuators
            )));
        }
        self.interaction_matrix = mat;
        Ok(())
    }

    /// Compute the control matrix M† via SVD, keeping `n_modes` singular vectors.
    ///
    /// Uses \[`pseudo_inverse`\] with SVD truncation.
    pub fn compute_control_matrix(&mut self, n_modes: usize) {
        self.n_modes_corrected = n_modes;
        self.control_matrix = pseudo_inverse(&self.interaction_matrix, n_modes);
    }

    /// Apply one integrator step given the current slope vector.
    ///
    /// # Arguments
    /// * `slopes` — slope measurements from the WFS, length `n_slopes`
    ///
    /// # Returns
    /// New actuator commands (also stored in `self.commands`).
    pub fn update(&mut self, slopes: &[f64]) -> Vec<f64> {
        let n_use = slopes.len().min(self.n_slopes);

        // Compute correction: delta_c = M† * s  (shape: n_actuators).
        let mut delta_c = vec![0.0_f64; self.n_actuators];
        for (i, delta_ci) in delta_c.iter_mut().enumerate().take(self.n_actuators) {
            if i < self.control_matrix.len() {
                for (j, &slope_j) in slopes.iter().enumerate().take(n_use) {
                    if j < self.control_matrix[i].len() {
                        *delta_ci += self.control_matrix[i][j] * slope_j;
                    }
                }
            }
        }

        // Update commands: c(n+1) = leak * c(n) - gain * delta_c.
        for (cmd, &dc) in self
            .commands
            .iter_mut()
            .zip(delta_c.iter())
            .take(self.n_actuators)
        {
            *cmd = self.leak * *cmd - self.gain * dc;
        }
        self.commands.clone()
    }

    /// Reset all commands to zero.
    pub fn reset(&mut self) {
        for c in self.commands.iter_mut() {
            *c = 0.0;
        }
    }

    /// Temporal bandwidth (−3 dB) of the integrator in units of loop frequency.
    ///
    /// For a leaky integrator with gain g and leak ρ:
    ///   f_{-3dB} ≈ g / (2π) (approximation for small gain).
    pub fn bandwidth_fraction(&self) -> f64 {
        self.gain / TWO_PI
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModalController
// ─────────────────────────────────────────────────────────────────────────────

/// Modal AO controller operating in the Zernike basis.
///
/// Converts slope measurements to modal coefficients using a modal
/// reconstruction matrix, then applies mode-dependent gains.
#[derive(Debug, Clone)]
pub struct ModalController {
    /// Number of Zernike modes controlled.
    pub n_modes: usize,
    /// Per-mode gains (can be set individually for modal optimisation).
    pub modal_gains: Vec<f64>,
    /// Leak per mode.
    pub modal_leaks: Vec<f64>,
    /// Current modal coefficients (Zernike amplitudes, metres).
    pub modal_commands: Vec<f64>,
    /// Modal reconstructor `R` of shape `[n_modes][n_slopes]`.
    pub modal_reconstructor: Vec<Vec<f64>>,
    /// Mode-to-actuator matrix `M` of shape `[n_actuators][n_modes]`.
    pub mode_to_actuator: Vec<Vec<f64>>,
    /// Number of slopes.
    pub n_slopes: usize,
    /// Number of actuators.
    pub n_actuators: usize,
}

impl ModalController {
    /// Create a modal controller with uniform gain.
    pub fn new(n_modes: usize, n_slopes: usize, n_actuators: usize, gain: f64) -> Self {
        Self {
            n_modes,
            modal_gains: vec![gain; n_modes],
            modal_leaks: vec![0.99; n_modes],
            modal_commands: vec![0.0; n_modes],
            modal_reconstructor: vec![vec![0.0; n_slopes]; n_modes],
            mode_to_actuator: vec![vec![0.0; n_modes]; n_actuators],
            n_slopes,
            n_actuators,
        }
    }

    /// Set the modal reconstructor matrix `R` (shape `[n_modes][n_slopes]`).
    pub fn set_modal_reconstructor(&mut self, r: Vec<Vec<f64>>) -> Result<(), OxiPhotonError> {
        if r.len() != self.n_modes {
            return Err(OxiPhotonError::NumericalError(format!(
                "Reconstructor has {} rows, expected {}",
                r.len(),
                self.n_modes
            )));
        }
        self.modal_reconstructor = r;
        Ok(())
    }

    /// Set the mode-to-actuator matrix `M` (shape `[n_actuators][n_modes]`).
    pub fn set_mode_to_actuator(&mut self, m: Vec<Vec<f64>>) -> Result<(), OxiPhotonError> {
        if m.len() != self.n_actuators {
            return Err(OxiPhotonError::NumericalError(format!(
                "Mode-to-actuator has {} rows, expected {}",
                m.len(),
                self.n_actuators
            )));
        }
        self.mode_to_actuator = m;
        Ok(())
    }

    /// Set per-mode gain (e.g., to suppress badly-sensed modes).
    pub fn set_modal_gain(&mut self, mode: usize, gain: f64) {
        if mode < self.n_modes {
            self.modal_gains[mode] = gain;
        }
    }

    /// Apply one modal controller step.
    ///
    /// Steps:
    /// 1. Reconstruct modal coefficients: a = R · s
    /// 2. Update modal commands: c_m(n+1) = leak · c_m(n) − g_m · a_m
    /// 3. Convert to actuator commands: c_act = M · c_m
    pub fn update(&mut self, slopes: &[f64]) -> Vec<f64> {
        let n_use = slopes.len().min(self.n_slopes);

        // Step 1: modal reconstruction.
        let mut modal_err = vec![0.0_f64; self.n_modes];
        for (m, modal_err_m) in modal_err.iter_mut().enumerate().take(self.n_modes) {
            for (j, &slope_j) in slopes.iter().enumerate().take(n_use) {
                if j < self.modal_reconstructor[m].len() {
                    *modal_err_m += self.modal_reconstructor[m][j] * slope_j;
                }
            }
        }

        // Step 2: integrator update per mode.
        for ((cmd, &err), (&leak, &gain)) in self
            .modal_commands
            .iter_mut()
            .zip(modal_err.iter())
            .zip(self.modal_leaks.iter().zip(self.modal_gains.iter()))
            .take(self.n_modes)
        {
            *cmd = leak * *cmd - gain * err;
        }

        // Step 3: project back to actuator space.
        let mut actuator_cmds = vec![0.0_f64; self.n_actuators];
        for (i, act_cmd) in actuator_cmds.iter_mut().enumerate().take(self.n_actuators) {
            for (&m2a, &mc) in self.mode_to_actuator[i]
                .iter()
                .zip(self.modal_commands.iter())
                .take(self.n_modes)
            {
                *act_cmd += m2a * mc;
            }
        }
        actuator_cmds
    }

    /// Reset all modal commands to zero.
    pub fn reset(&mut self) {
        for c in self.modal_commands.iter_mut() {
            *c = 0.0;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PredictiveController
// ─────────────────────────────────────────────────────────────────────────────

/// Linear predictive controller for Taylor frozen-flow turbulence.
///
/// Uses an AR(1) model: c(n+1) = A · c(n)
///
/// where A is the state transition matrix learned from a sequence of commands.
/// For frozen flow, A is a shift matrix corresponding to wind translation.
///
/// This is the simplest predictor; for better performance a higher-order
/// AR model or Kalman filter should be used.
#[derive(Debug, Clone)]
pub struct PredictiveController {
    /// Number of actuators.
    pub n_actuators: usize,
    /// AR(1) transition matrix `A` of shape `[n_actuators][n_actuators]`.
    pub transition_matrix: Vec<Vec<f64>>,
    /// Last actuator commands (state vector).
    pub state: Vec<f64>,
    /// Integral controller for residual correction.
    pub integral: IntegralController,
}

impl PredictiveController {
    /// Create a predictive controller.
    ///
    /// # Arguments
    /// * `n_actuators` — number of DM actuators
    /// * `n_slopes` — number of WFS slope measurements
    /// * `gain` — residual correction gain
    pub fn new(n_actuators: usize, n_slopes: usize, gain: f64) -> Self {
        // Default transition matrix: identity (persistence predictor).
        let transition_matrix = (0..n_actuators)
            .map(|i| {
                let mut row = vec![0.0_f64; n_actuators];
                if i < n_actuators {
                    row[i] = 0.95; // slightly damped persistence
                }
                row
            })
            .collect();

        Self {
            n_actuators,
            transition_matrix,
            state: vec![0.0; n_actuators],
            integral: IntegralController::new(gain, n_actuators, n_slopes),
        }
    }

    /// Set a custom transition matrix.
    pub fn set_transition_matrix(&mut self, a: Vec<Vec<f64>>) -> Result<(), OxiPhotonError> {
        if a.len() != self.n_actuators {
            return Err(OxiPhotonError::NumericalError(format!(
                "Transition matrix has {} rows, expected {}",
                a.len(),
                self.n_actuators
            )));
        }
        self.transition_matrix = a;
        Ok(())
    }

    /// Learn the transition matrix from a sequence of command vectors.
    ///
    /// Fits A by minimising ||C_{n+1} - A * C_n||² via the normal equations.
    /// `commands` should have at least 2 elements.
    pub fn learn_from_commands(&mut self, commands: &[Vec<f64>]) {
        if commands.len() < 2 {
            return;
        }
        let n = self.n_actuators;
        let n_samples = commands.len() - 1;

        // Compute A = C_next * C_prev^T * (C_prev * C_prev^T)^{-1}
        // Simple version: use the 1-step correlation estimator.
        // A[i][j] = (Σ c_{t+1}[i] * c_t[j]) / (Σ c_t[j]^2)

        let mut cross = vec![vec![0.0_f64; n]; n];
        let mut auto = vec![0.0_f64; n];

        for t in 0..n_samples {
            let c_prev = &commands[t];
            let c_next = &commands[t + 1];
            for i in 0..n.min(c_next.len()) {
                for j in 0..n.min(c_prev.len()) {
                    cross[i][j] += c_next[i] * c_prev[j];
                }
            }
            for j in 0..n.min(c_prev.len()) {
                auto[j] += c_prev[j] * c_prev[j];
            }
        }

        for (i, tm_row) in self.transition_matrix.iter_mut().enumerate().take(n) {
            for (j, tm_ij) in tm_row.iter_mut().enumerate().take(n) {
                *tm_ij = if auto[j] > 1e-30 {
                    cross[i][j] / auto[j]
                } else {
                    0.0
                };
            }
        }
    }

    /// Apply the predictive + integral correction.
    ///
    /// 1. Predict: c_pred = A · state
    /// 2. Correct residual: c_corr = integral.update(slopes)
    /// 3. New command: c_pred + c_corr
    pub fn update(&mut self, slopes: &[f64]) -> Vec<f64> {
        // Predict next state.
        let n = self.n_actuators;
        let mut predicted = vec![0.0_f64; n];
        for (i, pred_i) in predicted.iter_mut().enumerate().take(n) {
            for (j, &state_j) in self.state.iter().enumerate().take(n.min(self.state.len())) {
                *pred_i += self.transition_matrix[i][j] * state_j;
            }
        }

        // Residual integrator correction.
        let correction = self.integral.update(slopes);

        // Combine.
        let mut output = vec![0.0_f64; n];
        for (out_i, (&pred_i, &corr_i)) in output
            .iter_mut()
            .zip(predicted.iter().zip(correction.iter()))
            .take(n)
        {
            *out_i = pred_i + corr_i;
        }

        // Update state.
        self.state.clone_from(&output);
        output
    }

    /// Reset the controller state.
    pub fn reset(&mut self) {
        for s in self.state.iter_mut() {
            *s = 0.0;
        }
        self.integral.reset();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ClosedLoopMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Performance metrics for a closed AO loop.
///
/// Aggregates residual wavefront error, Strehl ratio, and bandwidth error
/// from a history of loop measurements.
#[derive(Debug, Clone, Default)]
pub struct ClosedLoopMetrics {
    /// History of residual RMS wavefront errors in metres.
    pub residual_rms_history: Vec<f64>,
    /// History of Strehl ratios (0–1).
    pub strehl_history: Vec<f64>,
    /// Loop gain used.
    pub loop_gain: f64,
    /// Loop frequency in Hz.
    pub loop_frequency: f64,
    /// Sensing wavelength in metres.
    pub wavelength: f64,
}

impl ClosedLoopMetrics {
    /// Create a new metrics tracker.
    pub fn new(loop_gain: f64, loop_frequency: f64, wavelength: f64) -> Self {
        Self {
            residual_rms_history: Vec::new(),
            strehl_history: Vec::new(),
            loop_gain,
            loop_frequency,
            wavelength,
        }
    }

    /// Record one frame of residual slopes.
    ///
    /// Computes the RMS slope error and its corresponding Strehl via Maréchal.
    pub fn record_frame(&mut self, residual_slopes: &[f64]) {
        let n = residual_slopes.len() as f64;
        if n < 1.0 {
            return;
        }
        let mean = residual_slopes.iter().sum::<f64>() / n;
        let rms = (residual_slopes
            .iter()
            .map(|&s| (s - mean) * (s - mean))
            .sum::<f64>()
            / n)
            .sqrt();
        // Convert slope RMS (rad) to OPD RMS (metres).
        let opd_rms = rms * self.wavelength / TWO_PI;
        self.residual_rms_history.push(opd_rms);

        // Maréchal Strehl: S = exp(−(2π σ/λ)²).
        let phase_rms = TWO_PI * opd_rms / self.wavelength;
        let strehl = (-phase_rms * phase_rms).exp();
        self.strehl_history.push(strehl);
    }

    /// Mean residual RMS wavefront error over the recorded history.
    pub fn mean_residual_rms(&self) -> f64 {
        if self.residual_rms_history.is_empty() {
            return 0.0;
        }
        let n = self.residual_rms_history.len() as f64;
        self.residual_rms_history.iter().sum::<f64>() / n
    }

    /// Mean Strehl ratio over the recorded history.
    pub fn mean_strehl(&self) -> f64 {
        if self.strehl_history.is_empty() {
            return 1.0;
        }
        let n = self.strehl_history.len() as f64;
        self.strehl_history.iter().sum::<f64>() / n
    }

    /// Bandwidth error variance (Greenwood 1977).
    ///
    /// For an integrator with gain g and frequency f_loop:
    ///   σ²_bw = (f_G / f_loop)^(5/3)
    ///
    /// where f_G is the Greenwood frequency.
    ///
    /// # Arguments
    /// * `greenwood_freq` — Greenwood frequency in Hz
    pub fn bandwidth_error_variance(&self, greenwood_freq: f64) -> f64 {
        let ratio = greenwood_freq / self.loop_frequency.max(1e-10);
        ratio.powf(5.0 / 3.0)
    }

    /// Noise error variance (Tyler 1984) for a Shack-Hartmann sensor.
    ///
    /// σ²_noise = (σ_θ / (2π/λ · d))²
    ///
    /// # Arguments
    /// * `slope_noise_rad` — slope noise RMS in radians
    /// * `subaperture_diameter` — lenslet pitch in metres
    pub fn noise_error_variance(&self, slope_noise_rad: f64, subaperture_diameter: f64) -> f64 {
        let kk = TWO_PI / self.wavelength;
        let denom = kk * subaperture_diameter;
        if denom < 1e-30 {
            return f64::INFINITY;
        }
        (slope_noise_rad / denom).powi(2)
    }

    /// Total AO residual variance: bandwidth + noise (uncorrelated, in m²).
    pub fn total_error_variance(
        &self,
        greenwood_freq: f64,
        slope_noise_rad: f64,
        subaperture_diameter: f64,
    ) -> f64 {
        // Convert dimensionless ratio variance to OPD variance (m²).
        let bw = self.bandwidth_error_variance(greenwood_freq);
        let noise = self.noise_error_variance(slope_noise_rad, subaperture_diameter);
        bw + noise
    }

    /// Clear all recorded history.
    pub fn clear(&mut self) {
        self.residual_rms_history.clear();
        self.strehl_history.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integral_controller_zero_slopes() {
        // With zero slopes and identity-like control matrix, commands stay zero.
        let mut ctrl = IntegralController::new(0.3, 4, 8);
        // Identity-like: M†[i][2i] = 1 etc (just test no panic).
        let slopes = vec![0.0_f64; 8];
        let cmds = ctrl.update(&slopes);
        assert_eq!(cmds.len(), 4);
        for c in &cmds {
            assert!(c.abs() < 1e-20, "Zero slopes → zero commands");
        }
    }

    #[test]
    fn test_integral_controller_reset() {
        let mut ctrl = IntegralController::new(0.3, 4, 4);
        ctrl.commands = vec![1.0, 2.0, 3.0, 4.0];
        ctrl.reset();
        for c in &ctrl.commands {
            assert_eq!(*c, 0.0);
        }
    }

    #[test]
    fn test_integral_controller_set_im_wrong_size() {
        let mut ctrl = IntegralController::new(0.3, 4, 8);
        let bad_im = vec![vec![0.0_f64; 4]; 5]; // should be 8 rows
        assert!(ctrl.set_interaction_matrix(bad_im).is_err());
    }

    #[test]
    fn test_integral_controller_gain_response() {
        // With non-zero control matrix and slopes, commands should be non-zero.
        let mut ctrl = IntegralController::new(0.5, 2, 2);
        // Set a simple identity control matrix.
        ctrl.control_matrix = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let slopes = vec![1.0, -1.0];
        let cmds = ctrl.update(&slopes);
        // c = 0 * 0.99 - 0.5 * [1, -1] = [-0.5, 0.5]
        assert!((cmds[0] - (-0.5)).abs() < 1e-12, "Command[0] = {}", cmds[0]);
        assert!((cmds[1] - 0.5).abs() < 1e-12, "Command[1] = {}", cmds[1]);
    }

    #[test]
    fn test_modal_controller_zero_slopes() {
        let mut ctrl = ModalController::new(6, 12, 10, 0.3);
        let slopes = vec![0.0_f64; 12];
        let cmds = ctrl.update(&slopes);
        assert_eq!(cmds.len(), 10);
        for c in &cmds {
            assert!(c.abs() < 1e-20);
        }
    }

    #[test]
    fn test_modal_controller_reset() {
        let mut ctrl = ModalController::new(6, 12, 10, 0.3);
        ctrl.modal_commands = vec![1.0; 6];
        ctrl.reset();
        for c in &ctrl.modal_commands {
            assert_eq!(*c, 0.0);
        }
    }

    #[test]
    fn test_modal_controller_set_reconstructor_wrong_size() {
        let mut ctrl = ModalController::new(6, 12, 10, 0.3);
        let bad = vec![vec![0.0_f64; 12]; 5]; // should be 6 rows
        assert!(ctrl.set_modal_reconstructor(bad).is_err());
    }

    #[test]
    fn test_predictive_controller_identity_transition() {
        let mut ctrl = PredictiveController::new(4, 8, 0.1);
        // State = [1, 2, 3, 4], A = damped identity.
        ctrl.state = vec![1.0, 2.0, 3.0, 4.0];
        let slopes = vec![0.0_f64; 8];
        let cmds = ctrl.update(&slopes);
        // Predicted = 0.95 * [1, 2, 3, 4], correction = 0.
        assert!((cmds[0] - 0.95).abs() < 1e-10, "Cmd[0] = {}", cmds[0]);
        assert!((cmds[1] - 1.9).abs() < 1e-10, "Cmd[1] = {}", cmds[1]);
    }

    #[test]
    fn test_predictive_controller_reset() {
        let mut ctrl = PredictiveController::new(4, 8, 0.1);
        ctrl.state = vec![1.0, 2.0, 3.0, 4.0];
        ctrl.reset();
        for s in &ctrl.state {
            assert_eq!(*s, 0.0);
        }
    }

    #[test]
    fn test_predictive_controller_learn_from_commands() {
        let mut ctrl = PredictiveController::new(2, 4, 0.1);
        // Commands: constant [1, 2] → transition should approach identity.
        let commands: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0, 2.0]).collect();
        ctrl.learn_from_commands(&commands);
        // A[0][0] ≈ 1, A[1][1] ≈ 1.
        assert!((ctrl.transition_matrix[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_svd_thin_rank1() {
        // 2×2 full-rank matrix with known singular values.
        // A = [[3,0],[4,0]] → σ₁ = 5, σ₂ ≈ 0 (rank-1 column-wise: both columns identical up to scale).
        // Use a clearly rank-1 matrix: col0 = [1,2], col1 = [2,4].
        // The SVD test: verify that reconstruction A ≈ Σᵢ σᵢ uᵢ vᵢᵀ works.
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let (u, sigma, vt) = svd_thin(&a);
        // The Frobenius norm of A is sqrt(1+4+4+16) = sqrt(25) = 5.
        // Sum of σᵢ² must equal ||A||²_F = 25.
        let sigma2_sum: f64 = sigma.iter().map(|&s| s * s).sum();
        assert!(
            (sigma2_sum - 25.0).abs() < 1e-6,
            "Sum of σᵢ² should equal ||A||²_F = 25, got {}",
            sigma2_sum
        );
        // Reconstruct A from all singular values and verify.
        let m = 2;
        let n_sv = sigma.len();
        let mut recon = vec![vec![0.0_f64; 2]; 2];
        for k in 0..n_sv {
            for i in 0..m {
                for j in 0..2 {
                    recon[i][j] += sigma[k] * u[k][i] * vt[k][j];
                }
            }
        }
        assert!(
            (recon[0][0] - 1.0).abs() < 1e-6,
            "A[0][0] recon = {}",
            recon[0][0]
        );
        assert!(
            (recon[0][1] - 2.0).abs() < 1e-6,
            "A[0][1] recon = {}",
            recon[0][1]
        );
        assert!(
            (recon[1][0] - 2.0).abs() < 1e-6,
            "A[1][0] recon = {}",
            recon[1][0]
        );
        assert!(
            (recon[1][1] - 4.0).abs() < 1e-6,
            "A[1][1] recon = {}",
            recon[1][1]
        );
    }

    #[test]
    fn test_closed_loop_metrics_empty() {
        let m = ClosedLoopMetrics::new(0.3, 1000.0, 633e-9);
        assert_eq!(m.mean_residual_rms(), 0.0);
        assert_eq!(m.mean_strehl(), 1.0);
    }

    #[test]
    fn test_closed_loop_metrics_record_and_retrieve() {
        let mut m = ClosedLoopMetrics::new(0.3, 1000.0, 633e-9);
        let slopes = vec![0.01_f64; 100];
        m.record_frame(&slopes);
        assert_eq!(m.residual_rms_history.len(), 1);
        assert_eq!(m.strehl_history.len(), 1);
        let s = m.strehl_history[0];
        assert!(s > 0.0 && s <= 1.0, "Strehl should be in (0, 1], got {}", s);
    }

    #[test]
    fn test_closed_loop_metrics_bandwidth_error() {
        let m = ClosedLoopMetrics::new(0.3, 1000.0, 633e-9);
        let bw_err = m.bandwidth_error_variance(100.0);
        // (100/1000)^(5/3) = 0.1^(5/3) ≈ 0.0464.
        let expected = 0.1_f64.powf(5.0 / 3.0);
        assert!((bw_err - expected).abs() < 1e-10, "bw_err = {}", bw_err);
    }

    #[test]
    fn test_integral_controller_bandwidth() {
        let ctrl = IntegralController::new(0.3, 4, 8);
        let bw = ctrl.bandwidth_fraction();
        let expected = 0.3 / TWO_PI;
        assert!((bw - expected).abs() < 1e-12);
    }
}
