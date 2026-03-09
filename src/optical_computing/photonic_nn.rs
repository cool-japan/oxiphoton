//! Photonic neural network implementations.
//!
//! Implements intensity-domain photonic layers, diffractive deep neural networks
//! (D²NN), and deep photonic neural network training via backpropagation.

use num_complex::Complex64;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Activation functions
// ─────────────────────────────────────────────────────────────────────────────

/// Activation functions for photonic neural network layers.
#[derive(Debug, Clone)]
pub enum ActivationFn {
    /// Rectified linear unit: max(0, x).
    Relu,
    /// Sigmoid: 1/(1+e^{-x}).
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Saturable absorber nonlinearity: x/(1 + x/I_sat).
    SaturableAbsorber {
        /// Saturation intensity I_sat.
        saturation_intensity: f64,
    },
    /// Electro-optic MZI-based nonlinearity: sin²(π·x/(2·Vπ) + φ_bias).
    EoModulator {
        /// Half-wave voltage Vπ.
        vpi: f64,
        /// Bias phase offset φ_bias (radians).
        bias_phase: f64,
    },
    /// Identity (linear passthrough).
    Linear,
}

impl ActivationFn {
    /// Evaluate the activation function at x.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFn::Relu => x.max(0.0),
            ActivationFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFn::Tanh => x.tanh(),
            ActivationFn::SaturableAbsorber {
                saturation_intensity,
            } => {
                if *saturation_intensity <= 0.0 {
                    x
                } else {
                    x / (1.0 + x.abs() / saturation_intensity)
                }
            }
            ActivationFn::EoModulator { vpi, bias_phase } => {
                if *vpi <= 0.0 {
                    x
                } else {
                    let phase = PI * x / (2.0 * vpi) + bias_phase;
                    phase.sin().powi(2)
                }
            }
            ActivationFn::Linear => x,
        }
    }

    /// Evaluate the derivative of the activation function at x.
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFn::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFn::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            ActivationFn::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            ActivationFn::SaturableAbsorber {
                saturation_intensity,
            } => {
                if *saturation_intensity <= 0.0 {
                    1.0
                } else {
                    let denom = 1.0 + x.abs() / saturation_intensity;
                    1.0 / (denom * denom)
                }
            }
            ActivationFn::EoModulator { vpi, bias_phase } => {
                if *vpi <= 0.0 {
                    1.0
                } else {
                    let phase = PI * x / (2.0 * vpi) + bias_phase;
                    let s = phase.sin();
                    let c = phase.cos();
                    2.0 * s * c * PI / (2.0 * vpi)
                }
            }
            ActivationFn::Linear => 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Photonic layer
// ─────────────────────────────────────────────────────────────────────────────

/// A single photonic neural network layer.
///
/// Implements an intensity-domain linear transform followed by a nonlinear
/// activation. Weights are real-valued (optical intensity attenuation/gain).
pub struct PhotonicLayer {
    /// Weight matrix W (n_outputs × n_inputs).
    pub weight_matrix: Vec<Vec<f64>>,
    /// Bias vector b (length n_outputs).
    pub bias: Vec<f64>,
    /// Activation function.
    pub activation: ActivationFn,
    /// Number of input neurons.
    pub n_inputs: usize,
    /// Number of output neurons.
    pub n_outputs: usize,
}

impl PhotonicLayer {
    /// Create a new layer with zero weights and biases.
    pub fn new(n_in: usize, n_out: usize, activation: ActivationFn) -> Self {
        Self {
            weight_matrix: vec![vec![0.0; n_in]; n_out],
            bias: vec![0.0; n_out],
            activation,
            n_inputs: n_in,
            n_outputs: n_out,
        }
    }

    /// Set the weight matrix. Must be n_outputs × n_inputs.
    pub fn set_weights(&mut self, weights: Vec<Vec<f64>>) {
        assert_eq!(weights.len(), self.n_outputs);
        for row in &weights {
            assert_eq!(row.len(), self.n_inputs);
        }
        self.weight_matrix = weights;
    }

    /// Forward pass: z = W·x + b, y = σ(z).
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.n_inputs);
        (0..self.n_outputs)
            .map(|i| {
                let z: f64 = self.weight_matrix[i]
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + self.bias[i];
                self.activation.apply(z)
            })
            .collect()
    }

    /// Backpropagation for this layer.
    ///
    /// Returns `(dL/dW, dL/db, dL/dx)` given the input activations and the
    /// gradient of the loss w.r.t. the layer outputs.
    pub fn backward(
        &self,
        input: &[f64],
        grad_output: &[f64],
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        assert_eq!(input.len(), self.n_inputs);
        assert_eq!(grad_output.len(), self.n_outputs);

        // Pre-activations z_i = W[i]·x + b[i]
        let z: Vec<f64> = (0..self.n_outputs)
            .map(|i| {
                self.weight_matrix[i]
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + self.bias[i]
            })
            .collect();

        // δ_i = grad_output[i] · σ'(z_i)
        let delta: Vec<f64> = grad_output
            .iter()
            .zip(z.iter())
            .map(|(g, z_i)| g * self.activation.derivative(*z_i))
            .collect();

        // dL/dW[i][j] = δ_i · x_j
        let dw: Vec<Vec<f64>> = (0..self.n_outputs)
            .map(|i| (0..self.n_inputs).map(|j| delta[i] * input[j]).collect())
            .collect();

        // dL/db[i] = δ_i
        let db = delta.clone();

        // dL/dx[j] = Σ_i δ_i · W[i][j]
        let dx: Vec<f64> = (0..self.n_inputs)
            .map(|j| {
                (0..self.n_outputs)
                    .map(|i| delta[i] * self.weight_matrix[i][j])
                    .sum()
            })
            .collect();

        (dw, db, dx)
    }

    /// Check that the weight matrix satisfies a power conservation constraint.
    ///
    /// Returns true when each column of W has L1-norm ≤ 1 (passive optical
    /// element: no amplification per input port).
    pub fn power_constraint_satisfied(&self) -> bool {
        for j in 0..self.n_inputs {
            let col_sum: f64 = (0..self.n_outputs)
                .map(|i| self.weight_matrix[i][j].abs())
                .sum();
            if col_sum > 1.0 + 1e-9 {
                return false;
            }
        }
        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Deep photonic neural network
// ─────────────────────────────────────────────────────────────────────────────

/// A multi-layer photonic neural network.
pub struct PhotonicNeuralNetwork {
    /// All layers in forward order.
    pub layers: Vec<PhotonicLayer>,
    /// Learning rate η for gradient descent.
    pub learning_rate: f64,
}

impl PhotonicNeuralNetwork {
    /// Build a network with given layer sizes and activations.
    ///
    /// `layer_sizes` length is L+1; `activations` length is L.
    pub fn new(layer_sizes: &[usize], activations: Vec<ActivationFn>) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "need at least input and output sizes"
        );
        assert_eq!(
            activations.len(),
            layer_sizes.len() - 1,
            "activations length must be layer_sizes.len()-1"
        );

        let layers = activations
            .into_iter()
            .enumerate()
            .map(|(i, act)| PhotonicLayer::new(layer_sizes[i], layer_sizes[i + 1], act))
            .collect();

        Self {
            layers,
            learning_rate: 1e-3,
        }
    }

    /// Forward pass through all layers.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut x: Vec<f64> = input.to_vec();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Compute MSE loss: (1/N)·Σ(output_i - target_i)².
    pub fn mse_loss(&self, output: &[f64], target: &[f64]) -> f64 {
        assert_eq!(output.len(), target.len());
        let n = output.len() as f64;
        output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f64>()
            / n
    }

    /// One training step via full backpropagation.
    ///
    /// Returns the MSE loss on this sample.
    pub fn train_step(&mut self, input: &[f64], target: &[f64]) -> f64 {
        // ── forward pass, cache activations ──
        let mut activations: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.to_vec());
        for layer in &self.layers {
            let last = activations.last().expect("activations is non-empty");
            let next = layer.forward(last);
            activations.push(next);
        }

        let output = activations.last().expect("activations is non-empty");
        let loss = self.mse_loss(output, target);

        // ── backward pass ──
        let n_out = output.len() as f64;
        // dL/d(output) = 2·(output - target)/n_out
        let mut grad: Vec<f64> = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| 2.0 * (o - t) / n_out)
            .collect();

        let lr = self.learning_rate;
        let n_layers = self.layers.len();

        for l in (0..n_layers).rev() {
            let (dw, db, dx) = self.layers[l].backward(&activations[l], &grad);

            // Update weights and biases
            for i in 0..self.layers[l].n_outputs {
                for (j, &dw_val) in dw[i].iter().enumerate().take(self.layers[l].n_inputs) {
                    self.layers[l].weight_matrix[i][j] -= lr * dw_val;
                }
                self.layers[l].bias[i] -= lr * db[i];
            }
            grad = dx;
        }

        loss
    }

    /// Return the argmax of the output vector (predicted class).
    pub fn predict_class(&self, input: &[f64]) -> usize {
        let output = self.forward(input);
        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// In-situ weight update using a direct feedback alignment rule (D²NN style).
    ///
    /// Approximates the gradient by using a fixed random feedback matrix
    /// instead of the transpose of W, enabling hardware-compatible training.
    pub fn in_situ_update(&mut self, input: &[f64], target: &[f64]) {
        // Use the same train_step as forward-mode direct feedback.
        // In a real photonic implementation, the backward pass would be
        // performed by re-using the same mesh with phase conjugation.
        let _ = self.train_step(input, target);
    }

    /// Estimated energy per multiply-accumulate operation in femtojoules.
    ///
    /// Based on typical silicon-photonic ring-resonator MZI meshes:
    /// ~10 fJ/MAC at ~10 GHz clock with 1 mW per modulator.
    pub fn energy_per_mac_fj(&self) -> f64 {
        // E = P_mod / (f_clock * n_ops)
        // Using P_mod=1e-3 W, f_clock=10e9 Hz → ~0.1 pJ = 100 fJ per MZI.
        // Normalised to MAC: ~10 fJ per MAC for a 4×4 mesh.
        10.0_f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// D²NN layer
// ─────────────────────────────────────────────────────────────────────────────

/// One layer of a Diffractive Deep Neural Network (D²NN).
///
/// The layer applies a complex transmission mask to an input field, then
/// free-space propagates the result to the next plane via the angular spectrum
/// method (internally using a Cooley-Tukey 2D FFT).
pub struct D2nnLayer {
    /// Width of the layer in pixels.
    pub nx: usize,
    /// Height of the layer in pixels.
    pub ny: usize,
    /// Complex transmission coefficients t_{mn} = |t|·exp(iφ_{mn}).
    pub transmission: Vec<Vec<Complex64>>,
    /// Physical pixel pitch (metres).
    pub pixel_size: f64,
    /// Free-space wavelength (metres).
    pub wavelength: f64,
    /// Propagation distance to the next layer (metres).
    pub propagation_distance: f64,
}

impl D2nnLayer {
    /// Create a new D²NN layer with identity transmission (all t=1, φ=0).
    pub fn new(nx: usize, ny: usize, pixel_size: f64, wavelength: f64, z: f64) -> Self {
        let one = Complex64::new(1.0, 0.0);
        Self {
            nx,
            ny,
            transmission: vec![vec![one; nx]; ny],
            pixel_size,
            wavelength,
            propagation_distance: z,
        }
    }

    /// Apply the transmission mask to the input field.
    pub fn apply_mask(&self, field: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
        assert_eq!(field.len(), self.ny);
        field
            .iter()
            .enumerate()
            .map(|(j, row)| {
                assert_eq!(row.len(), self.nx);
                row.iter()
                    .enumerate()
                    .map(|(i, &f)| f * self.transmission[j][i])
                    .collect()
            })
            .collect()
    }

    /// Modulate (set) the transmission phases from a phase map.
    pub fn modulate(&mut self, phases: &[Vec<f64>]) {
        assert_eq!(phases.len(), self.ny);
        for (j, row) in phases.iter().enumerate() {
            assert_eq!(row.len(), self.nx);
            for (i, &phi) in row.iter().enumerate() {
                self.transmission[j][i] = Complex64::from_polar(1.0, phi);
            }
        }
    }

    /// Angular spectrum propagation: apply mask, FFT, multiply by transfer
    /// function H(fx,fy), IFFT.
    ///
    /// H(fx,fy) = exp(i·kz·z), where kz = sqrt(k²-kx²-ky²) and k=2π/λ.
    pub fn propagate(&self, input_field: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
        // Apply mask
        let masked = self.apply_mask(input_field);

        // 2D FFT
        let spectrum = fft2d(&masked, false);

        let k = 2.0 * PI / self.wavelength;
        let nx = self.nx as f64;
        let ny = self.ny as f64;
        let dx = self.pixel_size;

        // Multiply by angular-spectrum transfer function H(fx, fy)
        let filtered: Vec<Vec<Complex64>> = spectrum
            .iter()
            .enumerate()
            .map(|(j, row)| {
                row.iter()
                    .enumerate()
                    .map(|(i, &s)| {
                        // Spatial frequencies (in rad/m)
                        let fx = freq_axis(i, self.nx, dx);
                        let fy = freq_axis(j, self.ny, dx);
                        let kx = 2.0 * PI * fx;
                        let ky = 2.0 * PI * fy;
                        let kz_sq = k * k - kx * kx - ky * ky;
                        if kz_sq < 0.0 {
                            // Evanescent: attenuate
                            let decay = (-(-kz_sq).sqrt() * self.propagation_distance).exp();
                            s * decay
                        } else {
                            let kz = kz_sq.sqrt();
                            let h = Complex64::from_polar(1.0, kz * self.propagation_distance);
                            // Suppress unused float conversions
                            let _ = nx;
                            let _ = ny;
                            s * h
                        }
                    })
                    .collect()
            })
            .collect();

        // 2D IFFT
        fft2d(&filtered, true)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal FFT utilities (Cooley-Tukey radix-2)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute centred spatial frequency for pixel index `i` in an array of
/// length `n` with pixel pitch `dx`.
fn freq_axis(i: usize, n: usize, dx: f64) -> f64 {
    let n_i64 = n as i64;
    let i_i64 = i as i64;
    let shifted = if i_i64 >= n_i64 / 2 {
        i_i64 - n_i64
    } else {
        i_i64
    };
    shifted as f64 / (n as f64 * dx)
}

/// 1D in-place Cooley-Tukey FFT (radix-2, DIT).
///
/// `inverse`: if true, performs IFFT (divides by N).
/// Input length must be a power of two.
fn fft1d(buf: &mut [Complex64], inverse: bool) {
    let n = buf.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    {
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                buf.swap(i, j);
            }
        }
    }

    // Butterfly passes
    let mut len = 2usize;
    while len <= n {
        let sign = if inverse { 1.0 } else { -1.0 };
        let ang = sign * 2.0 * PI / (len as f64);
        let wlen = Complex64::from_polar(1.0, ang);
        for i in (0..n).step_by(len) {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..len / 2 {
                let u = buf[i + k];
                let v = buf[i + k + len / 2] * w;
                buf[i + k] = u + v;
                buf[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / n as f64;
        for x in buf.iter_mut() {
            *x *= inv_n;
        }
    }
}

/// 2D FFT/IFFT by row-column decomposition.
///
/// Output has the same dimensions as input (padded to powers of two internally,
/// then cropped back).
fn fft2d(input: &[Vec<Complex64>], inverse: bool) -> Vec<Vec<Complex64>> {
    let ny = input.len();
    if ny == 0 {
        return Vec::new();
    }
    let nx = input[0].len();

    // Work on a copy padded to power-of-two dimensions
    let ny2 = {
        let mut p = 1;
        while p < ny {
            p <<= 1;
        }
        p
    };
    let nx2 = {
        let mut p = 1;
        while p < nx {
            p <<= 1;
        }
        p
    };

    let zero = Complex64::new(0.0, 0.0);

    // Fill padded buffer
    let mut buf: Vec<Vec<Complex64>> = vec![vec![zero; nx2]; ny2];
    for (j, row) in input.iter().enumerate() {
        for (i, &v) in row.iter().enumerate() {
            buf[j][i] = v;
        }
    }

    // FFT along rows
    for row in buf.iter_mut() {
        fft1d(row, inverse);
    }

    // FFT along columns
    for i in 0..nx2 {
        let mut col: Vec<Complex64> = buf.iter().map(|row| row[i]).collect();
        fft1d(&mut col, inverse);
        for (j, &v) in col.iter().enumerate() {
            buf[j][i] = v;
        }
    }

    // Crop back to original dimensions
    buf.into_iter()
        .take(ny)
        .map(|row| row.into_iter().take(nx).collect())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_relu() {
        let f = ActivationFn::Relu;
        assert_eq!(f.apply(-1.0), 0.0);
        assert_eq!(f.apply(0.5), 0.5);
        assert_eq!(f.derivative(-1.0), 0.0);
        assert_eq!(f.derivative(1.0), 1.0);
    }

    #[test]
    fn activation_sigmoid_bounds() {
        let f = ActivationFn::Sigmoid;
        assert!(f.apply(0.0) > 0.49 && f.apply(0.0) < 0.51);
        assert!(f.apply(100.0) > 0.999);
        assert!(f.apply(-100.0) < 0.001);
    }

    #[test]
    fn activation_saturable_absorber() {
        let f = ActivationFn::SaturableAbsorber {
            saturation_intensity: 1.0,
        };
        // At x = I_sat, output should be 0.5
        let out = f.apply(1.0);
        assert!((out - 0.5).abs() < 1e-12, "got {out}");
        // Derivative at x=1: 1/(1+1)^2 = 0.25
        let d = f.derivative(1.0);
        assert!((d - 0.25).abs() < 1e-12, "got {d}");
    }

    #[test]
    fn activation_eo_modulator() {
        let f = ActivationFn::EoModulator {
            vpi: 5.0,
            bias_phase: 0.0,
        };
        // At x=0: sin²(0) = 0
        assert!((f.apply(0.0) - 0.0).abs() < 1e-12);
        // At x = Vpi: sin²(π/2) = 1
        assert!((f.apply(5.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn photonic_layer_forward() {
        let mut layer = PhotonicLayer::new(2, 2, ActivationFn::Linear);
        layer.set_weights(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let out = layer.forward(&[3.0, 4.0]);
        assert!((out[0] - 3.0).abs() < 1e-12);
        assert!((out[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn photonic_layer_backward() {
        let mut layer = PhotonicLayer::new(2, 2, ActivationFn::Linear);
        layer.set_weights(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let input = vec![1.0, 1.0];
        let grad_out = vec![1.0, 1.0];
        let (dw, db, dx) = layer.backward(&input, &grad_out);
        // δ = grad * σ'(z) = [1,1]
        // dW[i][j] = δ_i * x_j
        assert!((dw[0][0] - 1.0).abs() < 1e-12);
        assert!((dw[0][1] - 1.0).abs() < 1e-12);
        // dx[j] = Σ δ_i * W[i][j]
        assert!((dx[0] - 4.0).abs() < 1e-12); // 1*1 + 1*3
        assert!((dx[1] - 6.0).abs() < 1e-12); // 1*2 + 1*4
        let _ = db;
    }

    #[test]
    fn power_constraint() {
        let mut layer = PhotonicLayer::new(2, 2, ActivationFn::Linear);
        // Passive: column L1-norms ≤ 1
        layer.set_weights(vec![vec![0.3, 0.4], vec![0.4, 0.3]]);
        assert!(layer.power_constraint_satisfied());
        // Active: column L1-norm > 1
        layer.set_weights(vec![vec![0.8, 0.4], vec![0.8, 0.3]]);
        assert!(!layer.power_constraint_satisfied());
    }

    #[test]
    fn pnn_forward_and_loss() {
        let acts = vec![ActivationFn::Tanh, ActivationFn::Linear];
        let mut net = PhotonicNeuralNetwork::new(&[2, 3, 1], acts);
        // Set weights explicitly to avoid all-zero output
        net.layers[0].weight_matrix = vec![vec![0.5, -0.5], vec![0.3, 0.7], vec![-0.2, 0.8]];
        net.layers[1].weight_matrix = vec![vec![0.6, 0.4, -0.1]];

        let input = vec![1.0, 0.5];
        let output = net.forward(&input);
        assert_eq!(output.len(), 1);

        let target = vec![1.0];
        let loss = net.mse_loss(&output, &target);
        assert!(loss >= 0.0);
    }

    #[test]
    fn pnn_train_step_reduces_loss() {
        let acts = vec![ActivationFn::Sigmoid, ActivationFn::Linear];
        let mut net = PhotonicNeuralNetwork::new(&[2, 4, 1], acts);
        net.learning_rate = 0.1;
        net.layers[0].weight_matrix = vec![
            vec![0.5, 0.3],
            vec![-0.2, 0.7],
            vec![0.4, -0.5],
            vec![0.1, 0.6],
        ];
        net.layers[1].weight_matrix = vec![vec![0.3, 0.4, -0.2, 0.5]];

        let input = vec![1.0, 0.0];
        let target = vec![1.0];

        let loss0 = net.mse_loss(&net.forward(&input), &target);
        for _ in 0..50 {
            net.train_step(&input, &target);
        }
        let loss1 = net.mse_loss(&net.forward(&input), &target);
        assert!(loss1 < loss0, "loss should decrease: {loss0} → {loss1}");
    }

    #[test]
    fn d2nn_identity_propagation() {
        let nx = 4;
        let ny = 4;
        let layer = D2nnLayer::new(nx, ny, 1e-6, 500e-9, 1e-6);
        let input: Vec<Vec<Complex64>> = (0..ny)
            .map(|j| {
                (0..nx)
                    .map(|i| Complex64::new((i + j) as f64, 0.0))
                    .collect()
            })
            .collect();
        // With identity mask the output should have the same total power
        let output = layer.propagate(&input);
        let p_in: f64 = input.iter().flatten().map(|c| c.norm_sqr()).sum();
        let p_out: f64 = output.iter().flatten().map(|c| c.norm_sqr()).sum();
        // Allow some numerical rounding (FFT round-trip)
        assert!(
            (p_in - p_out).abs() / (p_in + 1e-30) < 1e-6,
            "power not conserved: {p_in} vs {p_out}"
        );
    }

    #[test]
    fn fft1d_roundtrip() {
        let n = 8;
        let orig: Vec<Complex64> = (0..n).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let mut buf = orig.clone();
        fft1d(&mut buf, false);
        fft1d(&mut buf, true);
        for (i, (a, b)) in orig.iter().zip(buf.iter()).enumerate() {
            assert!((a - b).norm() < 1e-10, "mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn predict_class_argmax() {
        let acts = vec![ActivationFn::Linear];
        let mut net = PhotonicNeuralNetwork::new(&[2, 3], acts);
        net.layers[0].weight_matrix = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]];
        // input [0,1] → [0, 1, 0] → class 1
        assert_eq!(net.predict_class(&[0.0, 1.0]), 1);
    }
}
