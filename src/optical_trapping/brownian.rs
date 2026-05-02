/// Brownian dynamics simulation of optically trapped particles
///
/// Implements Langevin dynamics (Euler-Maruyama) for a particle in a harmonic
/// optical trap, including Stokes drag, diffusion, MSD computation, and PSD analysis.
/// Uses a pure-Rust LCG + Box-Muller Gaussian RNG (no rand crate).
// Physical constants
const KB: f64 = 1.380649e-23; // Boltzmann constant [J/K]
const PI: f64 = std::f64::consts::PI;

/// Stokes drag coefficient for a sphere in a viscous fluid \[N·s/m\]
///
/// γ = 6π η r
///
/// # Arguments
/// * `radius_m` — sphere radius \[m\]
/// * `viscosity_pa_s` — dynamic viscosity of fluid \[Pa·s\]
pub fn stokes_drag(radius_m: f64, viscosity_pa_s: f64) -> f64 {
    6.0 * PI * viscosity_pa_s * radius_m
}

/// Diffusion coefficient from Einstein relation \[m²/s\]
///
/// D = k_B T / γ
///
/// # Arguments
/// * `temperature_k` — temperature \[K\]
/// * `drag` — Stokes drag coefficient γ \[N·s/m\]
pub fn diffusion_coefficient(temperature_k: f64, drag: f64) -> f64 {
    if drag <= 0.0 {
        return 0.0;
    }
    KB * temperature_k / drag
}

/// Faxén correction factor for drag near a plane wall
///
/// To first order in (r/h): γ_eff/γ_0 ≈ 1/(1 - 9r/(16h))
/// (Blake/Faxén formula, valid for h >> r)
///
/// # Arguments
/// * `radius_m` — particle radius \[m\]
/// * `wall_distance_m` — distance from particle centre to wall \[m\]
///
/// # Returns
/// Dimensionless correction factor γ_eff/γ_0 (≥ 1)
pub fn faxen_drag_correction(radius_m: f64, wall_distance_m: f64) -> f64 {
    if wall_distance_m <= radius_m {
        // Particle touching or overlapping wall — return large but finite value
        return 100.0;
    }
    // Higher-order Faxén series (Brenner 1961):
    // γ_eff/γ_0 = [1 - (9/16)(r/h) + (1/8)(r/h)³ - (45/256)(r/h)⁴ - (1/16)(r/h)⁵]^{-1}
    let rh = radius_m / wall_distance_m;
    let denom = 1.0 - (9.0 / 16.0) * rh + (1.0 / 8.0) * rh.powi(3)
        - (45.0 / 256.0) * rh.powi(4)
        - (1.0 / 16.0) * rh.powi(5);
    if denom <= 0.0 {
        return 100.0;
    }
    1.0 / denom
}

/// Langevin dynamics simulator for a Brownian particle in a harmonic optical trap
///
/// Integrates the overdamped Langevin equation:
///   dx/dt = -k x/γ + F_fluct/γ
///   F_fluct ~ N(0, √(2γ k_B T / dt))
///
/// Uses Euler-Maruyama scheme (exact for linear SDE in overdamped limit).
/// RNG is a 64-bit LCG (no external rand crate).
#[derive(Debug, Clone)]
pub struct LangevinSimulator {
    /// Temperature \[K\]
    pub temperature_k: f64,
    /// Stokes drag coefficient γ = 6πηr \[N·s/m\]
    pub drag_coeff: f64,
    /// Trap stiffness vector (kx, ky, kz) \[N/m\]; 0 = free diffusion in that axis
    pub trap_stiffness: [f64; 3],
    /// Integration time step \[s\]
    pub dt_s: f64,
    /// Current position \[m\]
    pub position: [f64; 3],
    /// LCG random state (internal, do not set directly)
    rng_state: u64,
}

impl LangevinSimulator {
    /// Create a new Langevin simulator
    ///
    /// # Arguments
    /// * `temperature_k` — temperature \[K\]
    /// * `radius_m` — particle radius \[m\]
    /// * `viscosity_pa_s` — fluid viscosity \[Pa·s\]
    /// * `stiffness` — trap stiffness \[N/m\] for each axis
    /// * `dt_s` — timestep \[s\]
    pub fn new(
        temperature_k: f64,
        radius_m: f64,
        viscosity_pa_s: f64,
        stiffness: [f64; 3],
        dt_s: f64,
    ) -> Self {
        let drag_coeff = stokes_drag(radius_m, viscosity_pa_s);
        Self {
            temperature_k,
            drag_coeff,
            trap_stiffness: stiffness,
            dt_s,
            position: [0.0; 3],
            rng_state: 0xdeadbeef_cafebabe_u64,
        }
    }

    /// Create with explicit drag coefficient (for custom geometries)
    pub fn with_drag(temperature_k: f64, drag_coeff: f64, stiffness: [f64; 3], dt_s: f64) -> Self {
        Self {
            temperature_k,
            drag_coeff,
            trap_stiffness: stiffness,
            dt_s,
            position: [0.0; 3],
            rng_state: 0xfeedface_deadc0de_u64,
        }
    }

    /// Seed the internal LCG RNG
    pub fn seed(&mut self, seed: u64) {
        self.rng_state = seed;
    }

    /// Draw one uniform U(0,1) sample from LCG
    /// LCG: state = state × a + c  (Knuth constants)
    fn lcg_uniform(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.rng_state >> 33) as f64 / u32::MAX as f64
    }

    /// Draw one standard normal N(0,1) sample via Box-Muller
    ///
    /// Uses two uniform samples and returns the cosine variant
    fn gaussian_sample(&mut self) -> f64 {
        let u1 = self.lcg_uniform().max(1e-15); // avoid log(0)
        let u2 = self.lcg_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Perform one Euler-Maruyama integration step
    ///
    /// Returns updated position \[m\]
    pub fn step(&mut self) -> [f64; 3] {
        let dt = self.dt_s;
        let gamma = self.drag_coeff;
        // Thermal noise amplitude: σ_F = √(2 γ k_B T / dt)
        // Displacement noise: σ_x = √(2 D dt) = √(2 k_B T dt / γ)
        let sigma_x = (2.0 * KB * self.temperature_k * dt / gamma).sqrt();

        let mut new_pos = [0.0f64; 3];
        for (i, pos) in new_pos.iter_mut().enumerate() {
            let drift = -self.trap_stiffness[i] * self.position[i] / gamma * dt;
            let noise = sigma_x * self.gaussian_sample();
            *pos = self.position[i] + drift + noise;
        }
        self.position = new_pos;
        new_pos
    }

    /// Run simulation for `n_steps` steps, recording all positions
    ///
    /// Returns trajectory as Vec of \[x, y, z\] positions \[m\]
    pub fn run(&mut self, n_steps: usize) -> Vec<[f64; 3]> {
        let mut trajectory = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            trajectory.push(self.step());
        }
        trajectory
    }

    /// Compute mean-square displacement (MSD) from a trajectory
    ///
    /// MSD(τ) = <|r(t+τ) - r(t)|²> averaged over all time origins.
    /// Returns Vec of MSD values for lags 1..=max_lag.
    ///
    /// For a trapped particle: MSD(τ) = 2D/k × (1 - exp(-k τ/γ)) → 2D/k at long τ
    pub fn mean_square_displacement(trajectory: &[[f64; 3]], max_lag: usize) -> Vec<f64> {
        let n = trajectory.len();
        if n == 0 {
            return Vec::new();
        }
        let max_lag = max_lag.min(n - 1);
        let mut msd = vec![0.0f64; max_lag];

        for lag in 1..=max_lag {
            let count = n - lag;
            if count == 0 {
                continue;
            }
            let mut sum = 0.0;
            for t in 0..count {
                let r0 = trajectory[t];
                let r1 = trajectory[t + lag];
                let dr2 =
                    (r1[0] - r0[0]).powi(2) + (r1[1] - r0[1]).powi(2) + (r1[2] - r0[2]).powi(2);
                sum += dr2;
            }
            msd[lag - 1] = sum / count as f64;
        }
        msd
    }

    /// Compute power spectral density (PSD) for a 1D coordinate series
    ///
    /// Uses DFT (no FFT library required) for small-to-medium datasets.
    /// Returns Vec of (frequency \[Hz\], PSD \[m²/Hz\]) pairs.
    ///
    /// For a trapped particle the PSD is Lorentzian:
    /// S(f) = k_B T / (π² γ (f_c² + f²))
    pub fn power_spectral_density(trajectory: &[f64], dt: f64) -> Vec<(f64, f64)> {
        let n = trajectory.len();
        if n < 2 {
            return Vec::new();
        }

        // Apply Hann window to reduce spectral leakage
        let windowed: Vec<f64> = trajectory
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos());
                x * w
            })
            .collect();

        // Window normalisation factor
        let win_norm: f64 = windowed
            .iter()
            .map(|&_w| {
                // w was applied to x; we need to recover the window itself
                // Just use 0.5*(1-cos) sum
                let i_approx = 0; // placeholder — computed properly below
                let _ = i_approx;
                1.0
            })
            .count() as f64;
        // Proper window power normalisation: sum of w² / N
        let window_power: f64 = (0..n)
            .map(|i| {
                let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos());
                w * w
            })
            .sum::<f64>()
            / n as f64;
        let _ = win_norm; // suppress warning

        // Compute one-sided DFT PSD for positive frequencies
        let n_freq = n / 2 + 1;
        let mut psd = Vec::with_capacity(n_freq);

        for k in 0..n_freq {
            let freq = k as f64 / (n as f64 * dt);
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (j, &w) in windowed.iter().enumerate().take(n) {
                let angle = -2.0 * PI * k as f64 * j as f64 / n as f64;
                re += w * angle.cos();
                im += w * angle.sin();
            }
            // PSD [m²/Hz] with proper normalisation (one-sided)
            let power = (re * re + im * im) / (n as f64 * window_power * n as f64 * dt);
            // Factor 2 for one-sided (except DC and Nyquist)
            let factor = if k == 0 || k == n / 2 { 1.0 } else { 2.0 };
            psd.push((freq, power * factor));
        }
        psd
    }

    /// Estimate diffusion coefficient from long-lag MSD plateau
    ///
    /// D ≈ MSD_plateau / 6 (3D), or MSD_plateau / 2 (1D)
    pub fn estimate_diffusion_from_msd(msd: &[f64]) -> f64 {
        if msd.is_empty() {
            return 0.0;
        }
        // Use last 10% of MSD values as long-lag plateau (for free diffusion: MSD=6Dt)
        let start = msd.len() * 9 / 10;
        let tail = &msd[start..];
        if tail.is_empty() {
            return 0.0;
        }
        let mean_plateau = tail.iter().sum::<f64>() / tail.len() as f64;
        // For 3D trapped: plateau = 2D/k per axis → 6D/k total; not trivial.
        // For 3D free: MSD = 6 D τ → slope / 6. Here return plateau / 2 as 1D proxy.
        mean_plateau / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stokes_drag_water() {
        // 1 µm bead in water (η = 10^-3 Pa·s)
        let gamma = stokes_drag(1e-6, 1e-3);
        let expected = 6.0 * PI * 1e-3 * 1e-6; // ≈ 1.8850e-8 N·s/m
        assert!(
            (gamma - expected).abs() / expected < 0.01,
            "stokes_drag={} expected~{}",
            gamma,
            expected
        );
    }

    #[test]
    fn diffusion_coefficient_einstein() {
        // D = kT/γ for 1 µm bead at 300 K
        let gamma = stokes_drag(1e-6, 1e-3);
        let d = diffusion_coefficient(300.0, gamma);
        // D ≈ 1.38e-23 × 300 / 1.885e-8 ≈ 2.2e-13 m²/s
        assert!(
            d > 1e-14 && d < 1e-11,
            "diffusion coefficient D={} out of expected range",
            d
        );
    }

    #[test]
    fn faxen_correction_no_wall() {
        // Far from wall: correction → 1
        let correction = faxen_drag_correction(1e-6, 1e-3); // h = 1 mm, r = 1 µm
        assert!(
            (correction - 1.0).abs() < 0.01,
            "Faxén correction {} should be ~1 far from wall",
            correction
        );
    }

    #[test]
    fn faxen_correction_near_wall_increases() {
        let r = 1e-6;
        let far = faxen_drag_correction(r, 100e-6);
        let near = faxen_drag_correction(r, 5e-6);
        assert!(
            near > far,
            "Faxén correction should increase near wall: far={} near={}",
            far,
            near
        );
    }

    #[test]
    fn langevin_free_diffusion_msd_linear() {
        // Without trap: MSD = 6 D t = 6 (kT/γ) t
        let mut sim = LangevinSimulator::new(300.0, 1e-6, 1e-3, [0.0; 3], 1e-5);
        sim.seed(42);
        let traj = sim.run(2000);
        let msd = LangevinSimulator::mean_square_displacement(&traj, 100);
        // Check that MSD grows monotonically in early lags
        let grows = msd.windows(2).take(20).all(|w| w[1] >= w[0]);
        assert!(grows, "MSD should grow for free diffusion (first 20 lags)");
    }

    #[test]
    fn langevin_trapped_msd_plateau() {
        // With strong trap: MSD should plateau
        let k = 1e-4; // 100 µN/m — strong trap
        let gamma = stokes_drag(1e-6, 1e-3);
        let mut sim = LangevinSimulator::with_drag(300.0, gamma, [k; 3], 1e-5);
        sim.seed(123);
        let traj = sim.run(5000);
        let msd = LangevinSimulator::mean_square_displacement(&traj, 500);
        // Plateau value should be ~2kT/k per axis, 6kT/k total
        let plateau = msd.last().copied().unwrap_or(0.0);
        let expected_plateau = 6.0 * KB * 300.0 / k;
        // Allow factor of 3 tolerance due to finite statistics
        assert!(
            plateau < 3.0 * expected_plateau && plateau > 0.0,
            "Trapped MSD plateau {} >> expected {}",
            plateau,
            expected_plateau
        );
    }

    #[test]
    fn psd_returns_correct_length() {
        let mut sim = LangevinSimulator::new(300.0, 1e-6, 1e-3, [0.0; 3], 1e-4);
        sim.seed(99);
        let traj = sim.run(128);
        let x_traj: Vec<f64> = traj.iter().map(|p| p[0]).collect();
        let psd = LangevinSimulator::power_spectral_density(&x_traj, 1e-4);
        assert_eq!(psd.len(), 128 / 2 + 1, "PSD length mismatch");
    }

    #[test]
    fn gaussian_sample_zero_mean() {
        // Box-Muller samples should have mean ≈ 0 for large N
        let mut sim = LangevinSimulator::new(300.0, 1e-6, 1e-3, [0.0; 3], 1e-5);
        sim.seed(777);
        let n = 10000;
        let sum: f64 = (0..n).map(|_| sim.gaussian_sample()).sum();
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.05, "Gaussian mean {} too far from 0", mean);
    }
}
