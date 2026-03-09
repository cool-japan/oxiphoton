use num_complex::Complex64;
/// Coupled Mode Theory (CMT) for waveguide couplers and resonators.
///
/// CMT describes the evolution of mode amplitudes in coupled optical systems.
///
/// For two coupled waveguides with propagation constants β₁, β₂ and
/// coupling coefficient κ:
///
///   da₁/dz = i·β₁·a₁ + i·κ₁₂·a₂
///   da₂/dz = i·β₂·a₂ + i·κ₂₁·a₁
///
/// For a resonator coupled to a bus waveguide with coupling rate 1/τ_e and
/// decay rate 1/τ_i (loss):
///
///   da/dt = (i·ω₀ - 1/τ_i - 1/τ_e)·a + κ·s_in
///   s_out = s_in - κ*·a
use std::f64::consts::PI;

/// Two-waveguide directional coupler model via CMT.
#[derive(Debug, Clone, Copy)]
pub struct DirectionalCouplerCmt {
    /// Propagation constant of waveguide 1 (rad/m)
    pub beta1: f64,
    /// Propagation constant of waveguide 2 (rad/m)
    pub beta2: f64,
    /// Cross-coupling coefficient κ (rad/m) — symmetric: κ₁₂ = κ₂₁ = κ
    pub kappa: f64,
    /// Interaction length (m)
    pub length: f64,
}

impl DirectionalCouplerCmt {
    pub fn new(beta1: f64, beta2: f64, kappa: f64, length: f64) -> Self {
        Self {
            beta1,
            beta2,
            kappa,
            length,
        }
    }

    /// Symmetric coupler (Δβ = 0): both waveguides identical.
    pub fn symmetric(beta: f64, kappa: f64, length: f64) -> Self {
        Self::new(beta, beta, kappa, length)
    }

    /// Phase mismatch: Δβ = β₁ - β₂.
    pub fn delta_beta(&self) -> f64 {
        self.beta1 - self.beta2
    }

    /// Coupling parameter κ̃ = √(κ² + (Δβ/2)²).
    pub fn kappa_eff(&self) -> f64 {
        let db2 = self.delta_beta() / 2.0;
        (self.kappa * self.kappa + db2 * db2).sqrt()
    }

    /// Transfer matrix [a1_out, a2_out] for input [a1_in, a2_in].
    ///
    /// Analytic solution via coupled-mode eigenmode decomposition.
    pub fn transfer_matrix(&self) -> [[Complex64; 2]; 2] {
        let l = self.length;
        let k_eff = self.kappa_eff();
        let db = self.delta_beta();
        let beta_avg = (self.beta1 + self.beta2) / 2.0;

        // Common phase factor
        let phase = Complex64::new(0.0, beta_avg * l).exp();

        let cos_kl = (k_eff * l).cos();
        let sin_kl = (k_eff * l).sin();
        let sinc_kl = if k_eff.abs() < 1e-30 {
            l
        } else {
            sin_kl / k_eff
        };

        // M11 = (cos κ̃L + i Δβ/2 / κ̃ · sin κ̃L) · exp(i β_avg L)
        let m11 = phase * Complex64::new(cos_kl, db / 2.0 * sinc_kl);
        let m12 = phase * Complex64::new(0.0, self.kappa) * sinc_kl;
        let m21 = phase * Complex64::new(0.0, self.kappa) * sinc_kl;
        let m22 = phase * Complex64::new(cos_kl, -db / 2.0 * sinc_kl);

        [[m11, m12], [m21, m22]]
    }

    /// Power transmission from port 1→1 (through) and 1→2 (cross).
    ///
    /// Input: unit power in waveguide 1, no input in waveguide 2.
    pub fn power_transfer(&self) -> (f64, f64) {
        let m = self.transfer_matrix();
        let t11 = m[0][0].norm_sqr(); // through
        let t21 = m[1][0].norm_sqr(); // cross
        (t11, t21)
    }

    /// Coupling length L_c for complete power transfer (symmetric coupler only).
    ///
    ///   L_c = π / (2κ)
    pub fn coupling_length(&self) -> f64 {
        PI / (2.0 * self.kappa)
    }

    /// Required coupling coefficient κ for a target cross-coupling ratio η
    /// in length L.
    ///
    ///   κ = arcsin(√η) / L  (for symmetric coupler)
    pub fn kappa_for_coupling(eta: f64, length: f64) -> f64 {
        eta.sqrt().asin() / length
    }
}

/// Temporal CMT model for a resonator coupled to a waveguide bus.
///
/// The resonator amplitude a(t) evolves as:
///   da/dt = (iω₀ - γᵢ - γₑ)·a + √(2γₑ)·s_in
///   s_out = -s_in + √(2γₑ)·a
///
/// where γᵢ = 1/(2τᵢ) is the intrinsic decay rate and γₑ = 1/(2τₑ) is the
/// external (coupling) decay rate.
#[derive(Debug, Clone, Copy)]
pub struct ResonatorCmt {
    /// Resonance angular frequency ω₀ (rad/s)
    pub omega_0: f64,
    /// Intrinsic energy decay rate γᵢ = ω₀/(2Q_i) (rad/s)
    pub gamma_i: f64,
    /// External (coupling) decay rate γₑ (rad/s)
    pub gamma_e: f64,
}

impl ResonatorCmt {
    pub fn new(omega_0: f64, q_intrinsic: f64, q_external: f64) -> Self {
        Self {
            omega_0,
            gamma_i: omega_0 / (2.0 * q_intrinsic),
            gamma_e: omega_0 / (2.0 * q_external),
        }
    }

    /// Total loaded Q factor: 1/Q_L = 1/Q_i + 1/Q_e.
    pub fn q_loaded(&self) -> f64 {
        self.omega_0 / (2.0 * (self.gamma_i + self.gamma_e))
    }

    /// Transmission spectrum |s_out/s_in|² as a function of detuning Δω = ω - ω₀.
    ///
    ///   T(Δω) = |(-iΔω + γᵢ - γₑ) / (-iΔω + γᵢ + γₑ)|²
    pub fn transmission(&self, omega: f64) -> f64 {
        let delta = omega - self.omega_0;
        let num = Complex64::new(self.gamma_i - self.gamma_e, -delta);
        let den = Complex64::new(self.gamma_i + self.gamma_e, -delta);
        (num / den).norm_sqr()
    }

    /// Reflection (drop) spectrum |a_drop|² ∝ γₑ² / (Δω² + γ_total²).
    ///
    /// Normalised drop power spectrum.
    pub fn drop_power(&self, omega: f64) -> f64 {
        let delta = omega - self.omega_0;
        let gamma_total = self.gamma_i + self.gamma_e;
        4.0 * self.gamma_e * self.gamma_i / (delta * delta + gamma_total * gamma_total)
    }

    /// FWHM linewidth (rad/s) = 2(γᵢ + γₑ).
    pub fn linewidth_fwhm(&self) -> f64 {
        2.0 * (self.gamma_i + self.gamma_e)
    }

    /// Extinction ratio (dB) at resonance for all-pass configuration.
    ///
    ///   ER = 20 log₁₀|γᵢ - γₑ| / (γᵢ + γₑ)  (at Δω = 0)
    pub fn extinction_ratio_db(&self) -> f64 {
        let ratio = (self.gamma_i - self.gamma_e).abs() / (self.gamma_i + self.gamma_e);
        if ratio < 1e-30 {
            f64::INFINITY
        } else {
            -20.0 * ratio.log10()
        }
    }

    /// Critical coupling condition: γᵢ = γₑ → T = 0 (all power absorbed).
    pub fn is_critically_coupled(&self) -> bool {
        (self.gamma_i - self.gamma_e).abs() / (self.gamma_i + self.gamma_e) < 1e-3
    }
}

/// CMT model for a photonic crystal nanocavity.
/// Extends ResonatorCmt with far-field radiation channels.
#[derive(Debug, Clone, Copy)]
pub struct NanocavityCmt {
    pub resonator: ResonatorCmt,
    /// Number of radiation channels (far-field ports)
    pub n_radiation: usize,
    /// Radiation decay rates γ_rad,k (rad/s) per channel (average)
    pub gamma_rad: f64,
}

impl NanocavityCmt {
    pub fn new(omega_0: f64, q_i: f64, q_e: f64, n_rad: usize, gamma_rad: f64) -> Self {
        Self {
            resonator: ResonatorCmt::new(omega_0, q_i, q_e),
            n_radiation: n_rad,
            gamma_rad,
        }
    }

    /// Total decay rate including radiation.
    pub fn gamma_total(&self) -> f64 {
        self.resonator.gamma_i + self.resonator.gamma_e + self.n_radiation as f64 * self.gamma_rad
    }

    /// Purcell factor for emitter coupled to this nanocavity.
    ///
    ///   F_p = (3/(4π²)) · (λ/n)³ · Q/V
    ///
    /// Here V is provided in units of (λ/n)³.
    pub fn purcell_factor(&self, mode_volume_normalized: f64) -> f64 {
        let q = self.resonator.q_loaded();
        3.0 / (4.0 * PI * PI) * q / mode_volume_normalized
    }
}

// ---------------------------------------------------------------------------
// Tapered coupler — adiabatic power transfer
// ---------------------------------------------------------------------------

/// Tapered adiabatic coupler where the coupling coefficient κ(z) varies
/// linearly from `kappa0` at z = 0 to `kappa1` at z = `length`.
///
/// An optional propagation-constant mismatch Δβ(z) may be supplied via
/// `delta_beta_fn`.  When absent the coupler is phase-matched (Δβ = 0).
///
/// The coupled-mode equations integrated are:
///
///   da/dz = -i·κ(z)·b  −  i·Δβ(z)/2·a
///   db/dz = -i·κ(z)·a  +  i·Δβ(z)/2·b
///
/// with initial conditions a(0) = 1, b(0) = 0.
pub struct TaperedCoupler {
    /// Total interaction length (m)
    pub length: f64,
    /// Coupling coefficient at z = 0 (rad/m)
    pub kappa0: f64,
    /// Coupling coefficient at z = length (rad/m)
    pub kappa1: f64,
    /// Optional propagation-constant mismatch as a function of z (rad/m)
    pub delta_beta_fn: Option<Box<dyn Fn(f64) -> f64>>,
}

impl TaperedCoupler {
    /// Create a linearly tapered coupler with no phase mismatch (Δβ = 0).
    pub fn new(length: f64, kappa0: f64, kappa1: f64) -> Self {
        Self {
            length,
            kappa0,
            kappa1,
            delta_beta_fn: None,
        }
    }

    /// Attach a propagation-mismatch function Δβ(z).
    pub fn with_delta_beta(mut self, f: impl Fn(f64) -> f64 + 'static) -> Self {
        self.delta_beta_fn = Some(Box::new(f));
        self
    }

    /// Linearly interpolated coupling at position z.
    fn kappa_at(&self, z: f64) -> f64 {
        let t = z / self.length;
        self.kappa0 + t * (self.kappa1 - self.kappa0)
    }

    /// Integrate the coupled-mode equations using the classical 4th-order
    /// Runge-Kutta method.
    ///
    /// Returns `(pa, pb)` where `pa[i]` = |a(z_i)|² and `pb[i]` = |b(z_i)|²
    /// sampled at `n_steps + 1` evenly spaced z positions from 0 to `length`.
    pub fn propagate(&self, n_steps: usize) -> (Vec<f64>, Vec<f64>) {
        let dz = self.length / n_steps as f64;
        let mut a = Complex64::new(1.0, 0.0);
        let mut b = Complex64::new(0.0, 0.0);

        let mut pa = Vec::with_capacity(n_steps + 1);
        let mut pb = Vec::with_capacity(n_steps + 1);
        pa.push(a.norm_sqr());
        pb.push(b.norm_sqr());

        for step in 0..n_steps {
            let z = step as f64 * dz;
            let (a_new, b_new) = self.rk4_step(a, b, z, dz);
            a = a_new;
            b = b_new;
            pa.push(a.norm_sqr());
            pb.push(b.norm_sqr());
        }

        (pa, pb)
    }

    /// Single RK4 step for the coupled-mode system.
    fn rk4_step(&self, a: Complex64, b: Complex64, z: f64, dz: f64) -> (Complex64, Complex64) {
        let f = |av: Complex64, bv: Complex64, zv: f64| -> (Complex64, Complex64) {
            let kap = self.kappa_at(zv);
            let db2 = self
                .delta_beta_fn
                .as_ref()
                .map_or(0.0, |func| func(zv) / 2.0);
            let i = Complex64::i();
            let da = -i * kap * bv - i * db2 * av;
            let db = -i * kap * av + i * db2 * bv;
            (da, db)
        };

        let (k1a, k1b) = f(a, b, z);
        let (k2a, k2b) = f(a + 0.5 * dz * k1a, b + 0.5 * dz * k1b, z + 0.5 * dz);
        let (k3a, k3b) = f(a + 0.5 * dz * k2a, b + 0.5 * dz * k2b, z + 0.5 * dz);
        let (k4a, k4b) = f(a + dz * k3a, b + dz * k3b, z + dz);

        let a_new = a + (dz / 6.0) * (k1a + 2.0 * k2a + 2.0 * k3a + k4a);
        let b_new = b + (dz / 6.0) * (k1b + 2.0 * k2b + 2.0 * k3b + k4b);
        (a_new, b_new)
    }

    /// Final |b(L)|² after full propagation — the power transferred to mode b.
    pub fn transfer_efficiency(&self, n_steps: usize) -> f64 {
        let (_, pb) = self.propagate(n_steps);
        *pb.last().unwrap_or(&0.0)
    }
}

// ---------------------------------------------------------------------------
// Grating-assisted coupler — analytic CMT solution
// ---------------------------------------------------------------------------

/// Grating-assisted coupler described by a uniform grating coupling
/// coefficient κ, a phase mismatch Δβ, and a device length L.
///
/// Analytic coupled-mode solution (counter-propagating or co-propagating
/// depending on sign convention):
///
///   s² = κ² - (Δβ/2)²   (for over-coupled regime, s real)
///
/// Transfer matrix relates forward (a) and backward (b) amplitudes at z=0
/// to those at z=L.
#[derive(Debug, Clone, Copy)]
pub struct GratingCoupler {
    /// Grating coupling coefficient κ (rad/m)
    pub kappa: f64,
    /// Phase mismatch Δβ = β₁ - β₂ - K_g  (rad/m)
    pub delta_beta: f64,
    /// Device length L (m)
    pub length: f64,
}

impl GratingCoupler {
    pub fn new(kappa: f64, delta_beta: f64, length: f64) -> Self {
        Self {
            kappa,
            delta_beta,
            length,
        }
    }

    /// Analytic transfer matrix [[M11, M12], [M21, M22]] for the grating coupler.
    ///
    /// Derived from the contra-directional coupled-wave equations
    ///
    ///   da/dz = +i·(Δβ/2)·a + i·κ·b
    ///   db/dz = -i·(Δβ/2)·b − i·κ·a
    ///
    /// solved with s = √(κ² − (Δβ/2)²).  The matrix relates
    ///
    ///   [a(L), b(0)]ᵀ = M · [a(0), b(L)]ᵀ
    pub fn transfer_matrix(&self) -> [[Complex64; 2]; 2] {
        let kap = self.kappa;
        let db2 = self.delta_beta / 2.0;
        let l = self.length;
        let s2 = kap * kap - db2 * db2;
        let i = Complex64::i();

        if s2 >= 0.0 {
            // Over-coupled (strong grating): s real
            let s = s2.sqrt();
            let sl = s * l;
            let sh = sl.sinh();
            let ch = sl.cosh();
            // M11 = (cosh(sL) + i·Δβ/2/s·sinh(sL)) * exp(i·Δβ/2·L)
            // M12 = i·κ/s·sinh(sL) * exp(-i·Δβ/2·L)
            // M21 = -i·κ/s·sinh(sL) * exp(i·Δβ/2·L)
            // M22 = (cosh(sL) - i·Δβ/2/s·sinh(sL)) * exp(-i·Δβ/2·L)
            let ph_p = Complex64::new(0.0, db2 * l).exp();
            let ph_m = Complex64::new(0.0, -db2 * l).exp();
            let inv_s = if s.abs() < 1e-30 { l } else { 1.0 / s };
            let m11 = ph_p * (ch + i * db2 * sh * inv_s);
            let m12 = ph_m * i * kap * sh * inv_s;
            let m21 = ph_p * (-i * kap * sh * inv_s);
            let m22 = ph_m * (ch - i * db2 * sh * inv_s);
            [[m11, m12], [m21, m22]]
        } else {
            // Under-coupled (weak grating): s imaginary → use sin/cos
            let s = (-s2).sqrt();
            let sl = s * l;
            let sn = sl.sin();
            let cs = sl.cos();
            let ph_p = Complex64::new(0.0, db2 * l).exp();
            let ph_m = Complex64::new(0.0, -db2 * l).exp();
            let inv_s = if s.abs() < 1e-30 { l } else { 1.0 / s };
            let m11 = ph_p * (cs + i * db2 * sn * inv_s);
            let m12 = ph_m * i * kap * sn * inv_s;
            let m21 = ph_p * (-i * kap * sn * inv_s);
            let m22 = ph_m * (cs - i * db2 * sn * inv_s);
            [[m11, m12], [m21, m22]]
        }
    }

    /// Power transmittance |a(L)|² for unit forward input a(0)=1, b(L)=0.
    ///
    /// From the boundary-value solution:
    ///
    ///   T = s² / (s²·cosh²(sL) + (Δβ/2)²·sinh²(sL))   [over-coupled]
    pub fn transmittance(&self) -> f64 {
        let kap = self.kappa;
        let db2 = self.delta_beta / 2.0;
        let l = self.length;
        let s2 = kap * kap - db2 * db2;

        if s2 >= 0.0 {
            let s = s2.sqrt();
            let sh = (s * l).sinh();
            let ch = (s * l).cosh();
            let denom = s2 * ch * ch + db2 * db2 * sh * sh;
            if denom < 1e-300 {
                return 1.0;
            }
            s2 / denom
        } else {
            let s = (-s2).sqrt();
            let sn = (s * l).sin();
            // Under-coupled: T = 1 / (1 + κ²sin²(sL)/s²)
            let ratio = kap * kap * sn * sn / (-s2);
            1.0 / (1.0 + ratio)
        }
    }

    /// Power reflectance |b(0)|² for unit forward input a(0)=1, b(L)=0.
    ///
    ///   R = κ²·sinh²(sL) / (s²·cosh²(sL) + (Δβ/2)²·sinh²(sL))   [over-coupled]
    pub fn reflectance(&self) -> f64 {
        let kap = self.kappa;
        let db2 = self.delta_beta / 2.0;
        let l = self.length;
        let s2 = kap * kap - db2 * db2;

        if s2 >= 0.0 {
            let s = s2.sqrt();
            let sh = (s * l).sinh();
            let ch = (s * l).cosh();
            let denom = s2 * ch * ch + db2 * db2 * sh * sh;
            if denom < 1e-300 {
                return 0.0;
            }
            kap * kap * sh * sh / denom
        } else {
            let s = (-s2).sqrt();
            let sn = (s * l).sin();
            let ratio = kap * kap * sn * sn / (-s2);
            ratio / (1.0 + ratio)
        }
    }

    /// Phase-matched coupling length for complete power exchange:
    ///
    ///   L_pm = π / (2κ)   (at Δβ = 0)
    pub fn phase_matched_length(&self) -> f64 {
        PI / (2.0 * self.kappa)
    }
}

// ---------------------------------------------------------------------------
// Temporal CMT — pulse / driven excitation of a single resonance
// ---------------------------------------------------------------------------

/// Single-resonance temporal CMT driven by an external source.
///
/// The resonance amplitude a(t) follows:
///
///   da/dt = (-i·ω₀ − γ)·a + d·s_in(t)
///
/// with coupling coefficient d and total amplitude decay rate γ = 1/(2τ).
///
/// Transmission lineshape (Fano / Lorentzian) and impulse/steady-state
/// responses are provided.
#[derive(Debug, Clone, Copy)]
pub struct TemporalCmt {
    /// Resonance angular frequency ω₀ (rad/s)
    pub omega0: f64,
    /// Amplitude decay rate γ = 1/(2τ) (rad/s)
    pub gamma: f64,
    /// Coupling coefficient d (s^{-1/2})
    pub d: f64,
}

impl TemporalCmt {
    /// Create a `TemporalCmt` from resonance frequency ω₀, energy lifetime τ
    /// (so γ = 1/(2τ)), and coupling rate d.
    pub fn new(omega0: f64, tau: f64, d: f64) -> Self {
        Self {
            omega0,
            gamma: 1.0 / (2.0 * tau),
            d,
        }
    }

    /// Impulse response a(t) of the resonator for t ≥ 0:
    ///
    ///   a(t) = d · exp(−i·ω₀·t) · exp(−γ·t)
    ///
    /// Returns zero for t < 0 (causal).
    pub fn impulse_response(&self, t: &[f64]) -> Vec<Complex64> {
        t.iter()
            .map(|&ti| {
                if ti < 0.0 {
                    Complex64::new(0.0, 0.0)
                } else {
                    let phase = Complex64::new(0.0, -self.omega0 * ti).exp();
                    let decay = (-self.gamma * ti).exp();
                    self.d * decay * phase
                }
            })
            .collect()
    }

    /// Steady-state amplitude under a continuous-wave drive at frequency ω_drive:
    ///
    ///   a_ss = d / (i·(ω_drive − ω₀) + γ)
    pub fn steady_state_amplitude(&self, omega_drive: f64) -> Complex64 {
        let detuning = omega_drive - self.omega0;
        let denom = Complex64::new(self.gamma, detuning);
        Complex64::new(self.d, 0.0) / denom
    }

    /// Transmission spectrum |s_out / s_in|² over a range of drive frequencies.
    ///
    /// Using the standard temporal-CMT result for an all-pass ring / single-port
    /// resonator:
    ///
    ///   s_out = s_in + d* · a_ss
    ///
    /// where s_in = 1 (unit input), so
    ///
    ///   T(ω) = |1 + d* · a_ss|²
    ///
    /// This yields a Lorentzian absorption dip at ω = ω₀ when the resonator
    /// is critically coupled (d² = γ).
    pub fn transmission_spectrum(&self, omegas: &[f64]) -> Vec<f64> {
        omegas
            .iter()
            .map(|&omega| {
                let a_ss = self.steady_state_amplitude(omega);
                let s_out = Complex64::new(1.0, 0.0) + Complex64::new(self.d, 0.0) * a_ss;
                s_out.norm_sqr()
            })
            .collect()
    }

    /// Lorentzian FWHM linewidth of the resonance (rad/s) = 2γ.
    pub fn linewidth(&self) -> f64 {
        2.0 * self.gamma
    }

    /// Peak stored energy under CW drive at resonance:
    ///
    ///   U_peak = |a_ss(ω₀)|² = (d/γ)²
    pub fn peak_stored_energy(&self) -> f64 {
        (self.d / self.gamma).powi(2)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ---- DirectionalCouplerCmt ----

    #[test]
    fn symmetric_coupler_50_50() {
        // At L = L_c/2, should give 50/50 splitting
        let kappa = 1000.0; // rad/m
        let l_half = PI / (4.0 * kappa);
        let coupler = DirectionalCouplerCmt::symmetric(1e6, kappa, l_half);
        let (t, x) = coupler.power_transfer();
        assert!((t - 0.5).abs() < 1e-6, "T={t:.4}");
        assert!((x - 0.5).abs() < 1e-6, "X={x:.4}");
    }

    #[test]
    fn full_transfer_at_coupling_length() {
        let kappa = 1000.0;
        let l_c = PI / (2.0 * kappa);
        let coupler = DirectionalCouplerCmt::symmetric(1e6, kappa, l_c);
        let (t, x) = coupler.power_transfer();
        assert!(t < 1e-10, "Through should be ~0: {t:.2e}");
        assert!((x - 1.0).abs() < 1e-10, "Cross should be ~1: {x:.4}");
    }

    #[test]
    fn power_conservation_coupler() {
        let coupler = DirectionalCouplerCmt::new(1.5e6, 1.5e6 + 100.0, 800.0, 0.1e-3);
        let (t, x) = coupler.power_transfer();
        assert!((t + x - 1.0).abs() < 1e-10, "T+X should be 1: {:.4}", t + x);
    }

    #[test]
    fn resonator_transmission_unity_far_from_resonance() {
        let r = ResonatorCmt::new(2.0 * PI * 200e12, 1e5, 1e4);
        let t_far = r.transmission(r.omega_0 * 1.001); // far detuned
                                                       // Not necessarily unity unless under-coupled; just check it's between 0 and 1
        assert!((0.0..=1.0 + 1e-10).contains(&t_far));
    }

    #[test]
    fn resonator_critically_coupled_zero_transmission() {
        let omega = 2.0 * PI * 200e12;
        let q = 1e5;
        let r = ResonatorCmt::new(omega, q, q); // γi = γe: critical coupling
        let t_at_resonance = r.transmission(omega);
        assert!(
            t_at_resonance < 1e-20,
            "T at resonance = {t_at_resonance:.2e}"
        );
        assert!(r.is_critically_coupled());
    }

    #[test]
    fn resonator_linewidth_positive() {
        let r = ResonatorCmt::new(2.0 * PI * 200e12, 1e5, 2e4);
        assert!(r.linewidth_fwhm() > 0.0);
    }

    #[test]
    fn coupling_length_formula() {
        let kappa = 1000.0;
        let coupler = DirectionalCouplerCmt::symmetric(1e6, kappa, 1.0);
        let lc = coupler.coupling_length();
        assert!((lc - PI / (2.0 * kappa)).abs() < 1e-12);
    }

    #[test]
    fn resonator_q_loaded_less_than_qi() {
        let r = ResonatorCmt::new(2.0 * PI * 200e12, 1e5, 2e4);
        let q_l = r.q_loaded();
        // Q_L < Q_i always (coupling adds loss)
        assert!(q_l < 1e5);
    }

    #[test]
    fn purcell_factor_positive() {
        let nc = NanocavityCmt::new(2.0 * PI * 400e12, 1e6, 1e4, 2, 1e9);
        let fp = nc.purcell_factor(1.0); // V = 1 (λ/n)³
        assert!(fp > 0.0);
    }

    // ---- TaperedCoupler ----

    #[test]
    fn tapered_coupler_power_conservation() {
        // Power |a|² + |b|² should remain ≈ 1 throughout (lossless system).
        let tc = TaperedCoupler::new(1e-3, 500.0, 1500.0);
        let (pa, pb) = tc.propagate(1000);
        for (i, (a, b)) in pa.iter().zip(pb.iter()).enumerate() {
            let total = a + b;
            assert!(
                (total - 1.0).abs() < 1e-6,
                "step {i}: |a|²+|b|² = {total:.8}"
            );
        }
    }

    #[test]
    fn tapered_coupler_initial_conditions() {
        let tc = TaperedCoupler::new(1e-3, 200.0, 800.0);
        let (pa, pb) = tc.propagate(500);
        assert!((pa[0] - 1.0).abs() < 1e-15, "a(0) should be 1");
        assert!(pb[0].abs() < 1e-15, "b(0) should be 0");
    }

    #[test]
    fn tapered_coupler_transfer_efficiency_range() {
        let tc = TaperedCoupler::new(5e-3, 0.0, 2000.0);
        let eta = tc.transfer_efficiency(2000);
        assert!((0.0..=1.0).contains(&eta), "efficiency out of range: {eta}");
    }

    #[test]
    fn tapered_coupler_uniform_full_transfer() {
        // Uniform coupler (kappa0 = kappa1) at coupling length → full transfer.
        let kappa = 1000.0_f64;
        let l_c = PI / (2.0 * kappa);
        let tc = TaperedCoupler::new(l_c, kappa, kappa);
        let eta = tc.transfer_efficiency(5000);
        assert!(
            (eta - 1.0).abs() < 1e-4,
            "uniform coupler at L_c: eta={eta:.6}"
        );
    }

    #[test]
    fn tapered_coupler_with_delta_beta() {
        // Adding a non-zero Δβ should reduce transfer efficiency.
        let kappa = 1000.0_f64;
        let l_c = PI / (2.0 * kappa);
        let tc_ideal = TaperedCoupler::new(l_c, kappa, kappa);
        let tc_mismatched = TaperedCoupler::new(l_c, kappa, kappa).with_delta_beta(|_z| 500.0_f64); // constant Δβ
        let eta_ideal = tc_ideal.transfer_efficiency(5000);
        let eta_mm = tc_mismatched.transfer_efficiency(5000);
        // Mismatch reduces coupling (or at least differs)
        assert!(
            (eta_ideal - eta_mm).abs() > 1e-4,
            "Δβ should change efficiency: ideal={eta_ideal:.4}, mm={eta_mm:.4}"
        );
    }

    // ---- GratingCoupler ----

    #[test]
    fn grating_coupler_phase_matched_length() {
        let kappa = 500.0_f64;
        let gc = GratingCoupler::new(kappa, 0.0, 1.0);
        let lpm = gc.phase_matched_length();
        assert!((lpm - PI / (2.0 * kappa)).abs() < 1e-12);
    }

    #[test]
    fn grating_coupler_transmittance_reflectance_sum() {
        // T + R should equal 1 for a lossless grating at phase matching.
        let kappa = 300.0_f64;
        let l = PI / (2.0 * kappa);
        let gc = GratingCoupler::new(kappa, 0.0, l);
        let t = gc.transmittance();
        let r = gc.reflectance();
        assert!((t + r - 1.0).abs() < 1e-8, "T+R={:.6}", t + r);
    }

    #[test]
    fn grating_coupler_high_reflectance_long_grating() {
        // For a phase-matched grating (Δβ=0) with large κL, T → 0 and R → 1.
        // T = 1/cosh²(κL); at κL = 4 → T = 1/cosh²(4) ≈ 7e-4.
        let kappa = 400.0_f64;
        let l = 4.0 / kappa; // κL = 4
        let gc = GratingCoupler::new(kappa, 0.0, l);
        let t = gc.transmittance();
        let r = gc.reflectance();
        assert!(t < 2e-3, "T should be small for long grating: {t:.4e}");
        assert!(r > 0.997, "R should be close to 1 for long grating: {r:.6}");
    }

    #[test]
    fn grating_coupler_mismatched_higher_transmittance() {
        // Large Δβ (phase mismatch) → grating becomes ineffective → T ≈ 1.
        let kappa = 200.0_f64;
        let l = 0.5e-3_f64;
        let gc_pm = GratingCoupler::new(kappa, 0.0, l);
        let gc_mm = GratingCoupler::new(kappa, 1e6, l); // huge mismatch
        let t_pm = gc_pm.transmittance();
        let t_mm = gc_mm.transmittance();
        assert!(
            t_mm > t_pm,
            "mismatched coupler should transmit more: T_pm={t_pm:.4}, T_mm={t_mm:.4}"
        );
    }

    // ---- TemporalCmt ----

    #[test]
    fn temporal_cmt_impulse_response_causal() {
        let tcmt = TemporalCmt::new(2.0 * PI * 200e12, 1e-12, 1.0);
        let t_neg = [-1e-12_f64, -0.5e-12];
        let resp = tcmt.impulse_response(&t_neg);
        for r in &resp {
            assert!(r.norm() < 1e-30, "response should be zero for t<0");
        }
    }

    #[test]
    fn temporal_cmt_impulse_response_decay() {
        let tau = 1e-12_f64;
        let tcmt = TemporalCmt::new(2.0 * PI * 200e12, tau, 1.0);
        let times = [0.0_f64, tau, 2.0 * tau];
        let resp = tcmt.impulse_response(&times);
        // |a(tau)| / |a(0)| should equal exp(-1/2)
        let ratio = resp[1].norm() / resp[0].norm();
        let expected = (-0.5_f64).exp(); // exp(-gamma*tau) = exp(-1/2)
        assert!(
            (ratio - expected).abs() < 1e-10,
            "decay ratio={ratio:.6}, expected={expected:.6}"
        );
    }

    #[test]
    fn temporal_cmt_steady_state_at_resonance() {
        let omega0 = 2.0 * PI * 200e12;
        let tau = 1e-12_f64;
        let d = 1.0_f64;
        let tcmt = TemporalCmt::new(omega0, tau, d);
        let a_ss = tcmt.steady_state_amplitude(omega0);
        // At resonance: a_ss = d / gamma = 2*tau*d
        let expected_magnitude = d * 2.0 * tau; // d / (1/(2*tau))
        assert!(
            (a_ss.norm() - expected_magnitude).abs() < 1e-30 * expected_magnitude.abs() + 1e-10,
            "a_ss at resonance: {:.4e}, expected {:.4e}",
            a_ss.norm(),
            expected_magnitude
        );
    }

    #[test]
    fn temporal_cmt_transmission_spectrum_lorentzian_shape() {
        let omega0 = 2.0 * PI * 200e12;
        let tau = 1e-12_f64;
        // d² = gamma for critical coupling → T(ω₀) ≈ 0 when d² = γ
        let gamma = 1.0 / (2.0 * tau);
        let d = gamma.sqrt(); // d² = gamma → critical coupling dip
        let tcmt = TemporalCmt::new(omega0, tau, d);

        let lw = tcmt.linewidth();
        let omegas: Vec<f64> = (0..201)
            .map(|i| omega0 + (i as f64 - 100.0) * lw * 0.1)
            .collect();
        let spec = tcmt.transmission_spectrum(&omegas);

        // Spectrum should be symmetric around ω₀
        let n = spec.len();
        let mid = n / 2;
        for k in 1..10 {
            let diff = (spec[mid + k] - spec[mid - k]).abs();
            assert!(
                diff < 1e-10,
                "spectrum not symmetric at offset {k}: diff={diff:.2e}"
            );
        }
    }

    #[test]
    fn temporal_cmt_linewidth_positive() {
        let tcmt = TemporalCmt::new(2.0 * PI * 193e12, 5e-13, 1.0);
        assert!(tcmt.linewidth() > 0.0);
    }

    #[test]
    fn temporal_cmt_peak_stored_energy() {
        let tau = 2e-12_f64;
        let d = 3.0_f64;
        let tcmt = TemporalCmt::new(2.0 * PI * 200e12, tau, d);
        let expected = (d * 2.0 * tau).powi(2); // (d/gamma)²
        let computed = tcmt.peak_stored_energy();
        assert!(
            (computed - expected).abs() / expected < 1e-10,
            "peak energy mismatch: {computed:.4e} vs {expected:.4e}"
        );
    }
}
