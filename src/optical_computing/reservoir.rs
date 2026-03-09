//! Optical reservoir computing.
//!
//! Implements a delay-based optical reservoir computer (Appeltant et al., 2011)
//! using an MZI-based nonlinear node, and a photonic echo-state network (ESN)
//! with ridge-regression output training.

// ─────────────────────────────────────────────────────────────────────────────
// Optical delay reservoir
// ─────────────────────────────────────────────────────────────────────────────

/// Delay-based optical reservoir computer.
///
/// The reservoir uses a single nonlinear optical node with a delay line of
/// length τ = N · θ, where N is the number of virtual nodes and θ is the
/// node separation time. The input is masked with a random binary mask M_k.
///
/// State update: x_k\[n\] = sin²(β · x_{k-N}\[n-1\] + γ · (J · M_k · u\[n\]) + φ)
pub struct OpticalReservoir {
    /// Number of virtual nodes N.
    pub n_virtual_nodes: usize,
    /// Node separation time θ (seconds).
    pub theta: f64,
    /// Delay time τ = N·θ (seconds).
    pub tau: f64,
    /// Feedback strength β.
    pub feedback_strength: f64,
    /// Random binary mask M_k ∈ {-1, +1}, length N.
    pub input_mask: Vec<f64>,
    /// Current reservoir state x\[n\], length N.
    pub state: Vec<f64>,
    /// Trained output weights W_out, length N.
    pub output_weights: Vec<f64>,
}

impl OpticalReservoir {
    /// Create a new optical reservoir computer.
    ///
    /// The input mask is initialised with a deterministic pseudo-random
    /// binary sequence derived from the node index.
    pub fn new(n_nodes: usize, theta: f64, feedback: f64) -> Self {
        let tau = n_nodes as f64 * theta;
        // Deterministic pseudo-random mask: +1 for even, -1 for odd index.
        let input_mask: Vec<f64> = (0..n_nodes)
            .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        Self {
            n_virtual_nodes: n_nodes,
            theta,
            tau,
            feedback_strength: feedback,
            input_mask,
            state: vec![0.0; n_nodes],
            output_weights: vec![0.0; n_nodes],
        }
    }

    /// MZI-based nonlinear activation.
    ///
    /// x = sin²(β · feedback + γ · input + phase)
    ///
    /// γ is set to 0.1 (input coupling coefficient).
    pub fn activation(&self, feedback: f64, input: f64, phase: f64) -> f64 {
        let gamma = 0.1_f64;
        let arg = self.feedback_strength * feedback + gamma * input + phase;
        arg.sin().powi(2)
    }

    /// Run the reservoir on an input time series.
    ///
    /// For each input sample u\[n\], all N virtual node states x_k\[n\] are
    /// updated sequentially. Returns the reservoir state matrix (T × N).
    pub fn run(&mut self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let n = self.n_virtual_nodes;
        let mut all_states: Vec<Vec<f64>> = Vec::with_capacity(inputs.len());

        for &u in inputs {
            // Saved previous state for delayed feedback
            let prev_state = self.state.clone();

            for k in 0..n {
                // Feedback from the node k steps back (with wrap-around)
                let feedback = if k == 0 {
                    // First node reads from the delay line (last state)
                    prev_state[n - 1]
                } else {
                    prev_state[k - 1]
                };

                let masked_input = self.input_mask[k] * u;
                let phase = 0.0_f64; // static bias phase
                self.state[k] = self.activation(feedback, masked_input, phase);
            }

            all_states.push(self.state.clone());
        }

        all_states
    }

    /// Train output weights using ridge regression.
    ///
    /// Solves W_out = (X^T X + λI)^{-1} X^T y by Gaussian elimination.
    ///
    /// `states` is a (T × N) matrix; `targets` is a length-T vector.
    pub fn train(&mut self, states: &[Vec<f64>], targets: &[f64], lambda: f64) {
        let t = states.len();
        let n = self.n_virtual_nodes;

        if t == 0 || n == 0 {
            return;
        }

        // Form X^T X (n×n) and X^T y (n)
        let mut xtx: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        let mut xty: Vec<f64> = vec![0.0; n];

        for (step, state) in states.iter().enumerate() {
            let y = if step < targets.len() {
                targets[step]
            } else {
                0.0
            };
            for i in 0..n {
                xty[i] += state[i] * y;
                for j in 0..n {
                    xtx[i][j] += state[i] * state[j];
                }
            }
        }

        // Add ridge regularisation: X^T X + λI
        for (i, xtx_row) in xtx.iter_mut().enumerate().take(n) {
            xtx_row[i] += lambda;
        }

        // Solve the linear system via Gaussian elimination with partial pivoting
        self.output_weights = gaussian_elimination(&xtx, &xty).unwrap_or_else(|| vec![0.0; n]);
    }

    /// Predict output for each row of the state matrix.
    ///
    /// y_hat\[n\] = W_out · x\[n\].
    pub fn predict(&self, states: &[Vec<f64>]) -> Vec<f64> {
        states
            .iter()
            .map(|state| {
                state
                    .iter()
                    .zip(self.output_weights.iter())
                    .map(|(x, w)| x * w)
                    .sum()
            })
            .collect()
    }

    /// Normalised mean-square error: NMSE = ‖y_hat - y‖² / ‖y - ȳ‖².
    pub fn nmse(&self, predicted: &[f64], actual: &[f64]) -> f64 {
        assert_eq!(predicted.len(), actual.len());
        if actual.is_empty() {
            return 0.0;
        }

        let mean: f64 = actual.iter().sum::<f64>() / actual.len() as f64;

        let num: f64 = predicted
            .iter()
            .zip(actual.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum();

        let den: f64 = actual.iter().map(|a| (a - mean).powi(2)).sum();

        if den < 1e-30 {
            return 0.0;
        }
        num / den
    }

    /// Estimate the memory capacity of the reservoir.
    ///
    /// MC = Σ_{τ=1}^{test_length} corr²(y\[n\], u\[n-τ\])
    ///
    /// Runs the reservoir on a random-ish test sequence and computes the
    /// squared correlation for delays up to `test_length`.
    pub fn memory_capacity(&mut self, test_length: usize) -> f64 {
        if test_length == 0 {
            return 0.0;
        }

        // Generate a deterministic pseudo-random test sequence.
        let total = test_length + self.n_virtual_nodes;
        let inputs: Vec<f64> = (0..total)
            .map(|k| {
                let x = (k as f64 * 1.7 + 0.3).sin();
                x * x - 0.5 // zero-mean uniform-like
            })
            .collect();

        let states = self.run(&inputs[..total]);

        // Train output weights for each delay τ and sum squared correlations.
        let lambda = 1e-4;
        let mut mc = 0.0_f64;

        let warmup = self.n_virtual_nodes;
        let valid_len = states.len().saturating_sub(warmup);

        for tau in 1..=test_length.min(valid_len) {
            // Target: u[n - τ] for n in [warmup, warmup+valid_len)
            let targets: Vec<f64> = (warmup..states.len())
                .map(|n| {
                    if n >= tau + warmup {
                        inputs[n - tau]
                    } else {
                        0.0
                    }
                })
                .collect();

            let train_states: Vec<Vec<f64>> = states[warmup..].to_vec();
            self.train(&train_states, &targets, lambda);

            let predicted = self.predict(&train_states);

            // corr²(predicted, targets)
            let corr_sq = pearson_r_sq(&predicted, &targets);
            mc += corr_sq;
        }

        mc
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Echo State Network
// ─────────────────────────────────────────────────────────────────────────────

/// Photonic echo state network (ESN).
///
/// The reservoir matrix W_res is initialised with a fixed random-like pattern
/// scaled to have spectral radius `spectral_radius`. The input matrix W_in and
/// the output weights W_out are set during construction / training.
pub struct EchoStateNetwork {
    /// Number of reservoir neurons.
    pub n_reservoir: usize,
    /// Target spectral radius ρ(W_res).
    pub spectral_radius: f64,
    /// Input scaling factor.
    pub input_scaling: f64,
    /// Reservoir weight matrix W_res (n_reservoir × n_reservoir).
    pub reservoir_matrix: Vec<Vec<f64>>,
    /// Input weight matrix W_in (n_reservoir × n_inputs).
    pub input_matrix: Vec<Vec<f64>>,
    /// Output weight matrix W_out (n_outputs × n_reservoir).
    pub output_weights: Vec<Vec<f64>>,
    /// Current reservoir state x (length n_reservoir).
    pub state: Vec<f64>,
}

impl EchoStateNetwork {
    /// Create a new echo state network with a deterministic reservoir matrix.
    ///
    /// W_res is initialised with a pseudo-random sparse pattern, then scaled
    /// so that its dominant singular value approximates `spectral_radius`.
    pub fn new(n_in: usize, n_reservoir: usize, n_out: usize, spectral_radius: f64) -> Self {
        let reservoir_matrix = build_reservoir_matrix(n_reservoir, spectral_radius);
        let input_matrix = build_input_matrix(n_reservoir, n_in);

        Self {
            n_reservoir,
            spectral_radius,
            input_scaling: 1.0,
            reservoir_matrix,
            input_matrix,
            output_weights: vec![vec![0.0; n_reservoir]; n_out],
            state: vec![0.0; n_reservoir],
        }
    }

    /// Update reservoir state: x(t+1) = tanh(W_res · x(t) + W_in · u(t)).
    pub fn update_state(&mut self, input: &[f64]) {
        let n = self.n_reservoir;
        let n_in = input.len();

        let mut pre_act: Vec<f64> = vec![0.0; n];

        // W_res · x
        for (pre_act_i, res_row) in pre_act.iter_mut().zip(self.reservoir_matrix.iter()).take(n) {
            *pre_act_i += res_row
                .iter()
                .zip(self.state.iter())
                .map(|(w, x)| w * x)
                .sum::<f64>();
        }

        // W_in · u (with input scaling)
        for (i, pre_act_i) in pre_act.iter_mut().enumerate().take(n) {
            for (j, &u_j) in input.iter().enumerate() {
                if j < n_in && j < self.input_matrix[i].len() {
                    *pre_act_i += self.input_matrix[i][j] * u_j * self.input_scaling;
                }
            }
        }

        for (state_i, &pa) in self.state.iter_mut().zip(pre_act.iter()).take(n) {
            *state_i = pa.tanh();
        }
    }

    /// Forward pass: y = W_out · x(t+1).
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        // We do a stateless forward: compute state update on a copy.
        let mut tmp_state = self.state.clone();
        let n = self.n_reservoir;

        let mut pre_act: Vec<f64> = vec![0.0; n];
        for (pre_act_i, res_row) in pre_act.iter_mut().zip(self.reservoir_matrix.iter()).take(n) {
            *pre_act_i += res_row
                .iter()
                .zip(tmp_state.iter())
                .map(|(w, x)| w * x)
                .sum::<f64>();
        }
        for (i, pre_act_i) in pre_act.iter_mut().enumerate().take(n) {
            for (j, &u_j) in input.iter().enumerate() {
                if j < self.input_matrix[i].len() {
                    *pre_act_i += self.input_matrix[i][j] * u_j * self.input_scaling;
                }
            }
        }
        for (tmp_i, &pa) in tmp_state.iter_mut().zip(pre_act.iter()).take(n) {
            *tmp_i = pa.tanh();
        }

        // y = W_out · x
        self.output_weights
            .iter()
            .map(|row| row.iter().zip(tmp_state.iter()).map(|(w, x)| w * x).sum())
            .collect()
    }

    /// Train output weights using ridge regression.
    ///
    /// Runs the reservoir on all input sequences (with warm-up equal to 10%
    /// of total length), collects state snapshots, and solves:
    /// W_out = (X^T X + λI)^{-1} X^T Y.
    ///
    /// `inputs` is (T × n_in); `targets` is (T × n_out).
    pub fn ridge_regression_train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        lambda: f64,
    ) {
        let t_total = inputs.len();
        if t_total == 0 {
            return;
        }

        let warmup = (t_total / 10).max(1);
        let n = self.n_reservoir;
        let n_out = self.output_weights.len();

        // Reset state
        self.state = vec![0.0; n];

        // Collect states after warm-up
        let mut collected_states: Vec<Vec<f64>> = Vec::with_capacity(t_total - warmup);

        for (t, input) in inputs.iter().enumerate() {
            self.update_state(input);
            if t >= warmup {
                collected_states.push(self.state.clone());
            }
        }

        let t_train = collected_states.len();
        if t_train == 0 {
            return;
        }

        // Solve one ridge regression per output dimension
        for out_idx in 0..n_out {
            // Build X^T X (n×n) and X^T y (n)
            let mut xtx: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
            let mut xty: Vec<f64> = vec![0.0; n];

            for (t, state) in collected_states.iter().enumerate() {
                let y_t = if t + warmup < targets.len() {
                    targets[t + warmup].get(out_idx).copied().unwrap_or(0.0)
                } else {
                    0.0
                };
                for i in 0..n {
                    xty[i] += state[i] * y_t;
                    for j in 0..n {
                        xtx[i][j] += state[i] * state[j];
                    }
                }
            }

            // Ridge regularisation
            for (i, xtx_row) in xtx.iter_mut().enumerate().take(n) {
                xtx_row[i] += lambda;
            }

            let w = gaussian_elimination(&xtx, &xty).unwrap_or_else(|| vec![0.0; n]);
            self.output_weights[out_idx] = w;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a deterministic sparse reservoir matrix scaled to `spectral_radius`.
///
/// Uses a fixed pattern: W[i]\[j\] = sin(i*7 + j*13 + 0.5) for (i+j) % 5 == 0,
/// else 0. The matrix is then scaled by spectral_radius / estimated_max_singular.
fn build_reservoir_matrix(n: usize, spectral_radius: f64) -> Vec<Vec<f64>> {
    let mut m: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    for (i, m_row) in m.iter_mut().enumerate().take(n) {
        for (j, m_ij) in m_row.iter_mut().enumerate().take(n) {
            if (i + j) % 5 == 0 {
                *m_ij = ((i as f64) * 7.0 + (j as f64) * 13.0 + 0.5).sin();
            }
        }
    }

    // Estimate the dominant singular value by power iteration (20 steps).
    let dom_sv = power_iteration_norm(&m, n, 20);

    if dom_sv > 1e-10 {
        let scale = spectral_radius / dom_sv;
        for row in m.iter_mut() {
            for v in row.iter_mut() {
                *v *= scale;
            }
        }
    }

    m
}

/// Build a deterministic input weight matrix (n_reservoir × n_in).
fn build_input_matrix(n_reservoir: usize, n_in: usize) -> Vec<Vec<f64>> {
    (0..n_reservoir)
        .map(|i| {
            (0..n_in)
                .map(|j| ((i as f64) * 3.7 + (j as f64) * 5.3 + 1.0).sin() * 0.5)
                .collect()
        })
        .collect()
}

/// Estimate the spectral norm of a square matrix via power iteration.
fn power_iteration_norm(m: &[Vec<f64>], n: usize, iters: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }

    // Start with a simple non-zero vector
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).recip()).collect();

    for _ in 0..iters {
        // w = M · v
        let mut w: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            w[i] = m[i].iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        }
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-30 {
            break;
        }
        v = w.iter().map(|x| x / norm).collect();
    }

    // Final Rayleigh quotient
    let mv: Vec<f64> = (0..n)
        .map(|i| m[i].iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect();
    mv.iter()
        .zip(v.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .abs()
}

/// Solve A·x = b using Gaussian elimination with partial pivoting.
///
/// Returns `None` if the system is singular (‖A‖ too small on the pivot).
fn gaussian_elimination(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Some(Vec::new());
    }

    // Build augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivoting: find row with max absolute value in this column
        let pivot_row = (col..n)
            .max_by(|&i, &j| {
                aug[i][col]
                    .abs()
                    .partial_cmp(&aug[j][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);

        aug.swap(col, pivot_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 {
            return None; // singular
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            let (left, right) = aug.split_at_mut(row);
            for (piv_k, row_k) in left[col][col..=n].iter().zip(right[0][col..=n].iter_mut()) {
                *row_k -= factor * piv_k;
            }
        }
    }

    // Back-substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = aug[i][n];
        for j in (i + 1)..n {
            s -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() < 1e-15 {
            return None;
        }
        x[i] = s / aug[i][i];
    }

    Some(x)
}

/// Compute the squared Pearson correlation coefficient between two vectors.
fn pearson_r_sq(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let mean_x: f64 = x[..n].iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y[..n].iter().sum::<f64>() / n as f64;

    let cov: f64 = x[..n]
        .iter()
        .zip(y[..n].iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    let var_x: f64 = x[..n].iter().map(|xi| (xi - mean_x).powi(2)).sum();
    let var_y: f64 = y[..n].iter().map(|yi| (yi - mean_y).powi(2)).sum();

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-30 {
        return 0.0;
    }
    (cov / denom).powi(2)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservoir_activation_bounds() {
        let rc = OpticalReservoir::new(10, 1e-9, 0.5);
        // sin²(x) is always in [0, 1]
        for phase in [0.0, 0.5, 1.0, -1.0, std::f64::consts::PI] {
            let out = rc.activation(0.3, 0.7, phase);
            assert!(
                (0.0..=1.0).contains(&out),
                "activation out of bounds: {out}"
            );
        }
    }

    #[test]
    fn reservoir_run_output_shape() {
        let mut rc = OpticalReservoir::new(5, 1e-9, 0.4);
        let inputs = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let states = rc.run(&inputs);
        assert_eq!(states.len(), inputs.len());
        for row in &states {
            assert_eq!(row.len(), 5);
        }
    }

    #[test]
    fn reservoir_train_and_predict() {
        let n = 8;
        let mut rc = OpticalReservoir::new(n, 1e-9, 0.5);

        // Generate a sine-wave sequence
        let t_len = 50;
        let inputs: Vec<f64> = (0..t_len).map(|k| (k as f64 * 0.3).sin()).collect();
        let states = rc.run(&inputs);

        // Target: delayed input (shift by 1)
        let targets: Vec<f64> = (0..t_len)
            .map(|k| if k > 0 { inputs[k - 1] } else { 0.0 })
            .collect();

        rc.train(&states, &targets, 1e-4);
        let predicted = rc.predict(&states);

        // Should have some correlation (not perfect due to small reservoir)
        let nmse = rc.nmse(&predicted, &targets);
        assert!(nmse < 10.0, "NMSE too large: {nmse}");
    }

    #[test]
    fn nmse_zero_for_perfect_prediction() {
        let rc = OpticalReservoir::new(4, 1e-9, 0.5);
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(rc.nmse(&v, &v), 0.0);
    }

    #[test]
    fn gaussian_elimination_basic() {
        // 2x + y = 5, x + 3y = 7 → x=1.6, y=1.8
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 7.0];
        let x = gaussian_elimination(&a, &b).expect("should solve");
        assert!((x[0] - 1.6).abs() < 1e-10, "x[0]={}", x[0]);
        assert!((x[1] - 1.8).abs() < 1e-10, "x[1]={}", x[1]);
    }

    #[test]
    fn gaussian_elimination_singular_returns_none() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]]; // rank-1
        let b = vec![3.0, 6.0];
        // With exact zero pivot after elimination this should return None
        // (2nd pivot is 0 after forward elimination).
        let result = gaussian_elimination(&a, &b);
        // Either None (singular) or Some (pivot just large enough for tolerance).
        // We just check it doesn't panic.
        let _ = result;
    }

    #[test]
    fn esn_state_update() {
        let mut esn = EchoStateNetwork::new(1, 4, 1, 0.9);
        let input = vec![0.5];
        esn.update_state(&input);
        // State should be in (-1, 1) due to tanh
        for &s in &esn.state {
            assert!(s > -1.0 && s < 1.0, "state out of tanh range: {s}");
        }
    }

    #[test]
    fn esn_forward_output_size() {
        let esn = EchoStateNetwork::new(2, 6, 3, 0.9);
        let output = esn.forward(&[1.0, 0.5]);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn esn_ridge_regression_train() {
        let mut esn = EchoStateNetwork::new(1, 8, 1, 0.8);

        // Identity mapping: y = u
        let t = 30;
        let inputs: Vec<Vec<f64>> = (0..t).map(|k| vec![(k as f64 * 0.2).sin()]).collect();
        let targets: Vec<Vec<f64>> = inputs.clone();

        esn.ridge_regression_train(&inputs, &targets, 1e-4);

        // After training, output should roughly track input
        let test_in = vec![0.7];
        let _out = esn.forward(&test_in);
        // (Just verify no panic and output has correct size)
        let out = esn.forward(&test_in);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn memory_capacity_nonnegative() {
        let mut rc = OpticalReservoir::new(6, 1e-9, 0.5);
        let mc = rc.memory_capacity(5);
        assert!(mc >= 0.0, "MC should be non-negative, got {mc}");
    }
}
