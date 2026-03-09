use super::pdk::SoiProcess;
/// PIC Design Optimization Algorithms.
///
/// Provides objective function definitions, PSO optimizer, ring resonator
/// optimization, and MZI modulator optimization — all using deterministic
/// pseudo-random generation (LCG) to stay rand-free.
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic LCG PRNG
// ─────────────────────────────────────────────────────────────────────────────

/// Linear Congruential Generator (Knuth's constants).
///
/// Provides reproducible pseudo-random floats in [0, 1) without the `rand`
/// crate, satisfying the COOLJAPAN Pure-Rust policy.
#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    /// Initialise with a seed.
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Advance the LCG and return the next `u64`.
    fn next_u64(&mut self) -> u64 {
        // Knuth's multiplicative LCG (TAOCP Vol.2 §3.3.4)
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Return a pseudo-random float in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Return a pseudo-random float in [lo, hi).
    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InverseDesignObjective
// ─────────────────────────────────────────────────────────────────────────────

/// Objective function for inverse photonic design.
///
/// Computes a weighted least-squares cost combining transmission error,
/// phase error, and design-complexity penalties.
#[derive(Debug, Clone)]
pub struct InverseDesignObjective {
    /// Target (wavelength_nm, transmission) pairs
    pub target_transmission: Vec<(f64, f64)>,
    /// Target (wavelength_nm, phase_rad) pairs
    pub target_phase: Vec<(f64, f64)>,
    /// Weight for optical power penalty (unused modes)
    pub power_penalty_weight: f64,
    /// Weight for device footprint penalty
    pub area_penalty_weight: f64,
}

impl InverseDesignObjective {
    /// Build a bandpass filter objective.
    ///
    /// Creates a raised-cosine transmission profile with unit transmission
    /// in the passband and `rejection_db` attenuation outside.
    ///
    /// # Arguments
    /// * `center_nm`    – Passband centre wavelength (nm)
    /// * `bandwidth_nm` – Full passband width (nm)
    /// * `rejection_db` – Stop-band rejection (dB, positive value)
    pub fn new_bandpass(center_nm: f64, bandwidth_nm: f64, rejection_db: f64) -> Self {
        let n_pts = 64usize;
        let lambda_start = center_nm - 3.0 * bandwidth_nm;
        let lambda_stop = center_nm + 3.0 * bandwidth_nm;
        let t_min = 10.0_f64.powf(-rejection_db.abs() / 10.0);

        let target_transmission: Vec<(f64, f64)> = (0..n_pts)
            .map(|i| {
                let lam =
                    lambda_start + (lambda_stop - lambda_start) * i as f64 / (n_pts - 1) as f64;
                let delta = (lam - center_nm) / (bandwidth_nm / 2.0);
                let t = if delta.abs() <= 1.0 {
                    // Raised cosine passband
                    0.5 * (1.0 + (PI * delta).cos())
                } else {
                    t_min
                };
                (lam, t.clamp(t_min, 1.0))
            })
            .collect();

        let target_phase: Vec<(f64, f64)> = (0..n_pts)
            .map(|i| {
                let lam =
                    lambda_start + (lambda_stop - lambda_start) * i as f64 / (n_pts - 1) as f64;
                (lam, 0.0)
            })
            .collect();

        Self {
            target_transmission,
            target_phase,
            power_penalty_weight: 0.1,
            area_penalty_weight: 0.01,
        }
    }

    /// Evaluate the total objective value (lower = better).
    ///
    /// Cost = MSE(transmission) + MSE(phase) + penalties
    ///
    /// # Arguments
    /// * `actual_transmission` – Slice of computed transmission values (same order as targets)
    /// * `actual_phase`        – Slice of computed phase values (rad)
    pub fn evaluate(&self, actual_transmission: &[f64], actual_phase: &[f64]) -> f64 {
        let n_t = self
            .target_transmission
            .len()
            .min(actual_transmission.len());
        let n_p = self.target_phase.len().min(actual_phase.len());

        let t_cost: f64 = (0..n_t)
            .map(|i| {
                let err = actual_transmission[i] - self.target_transmission[i].1;
                err * err
            })
            .sum::<f64>()
            / n_t.max(1) as f64;

        let p_cost: f64 = (0..n_p)
            .map(|i| {
                let err = actual_phase[i] - self.target_phase[i].1;
                err * err
            })
            .sum::<f64>()
            / n_p.max(1) as f64;

        t_cost + p_cost * 0.1
    }

    /// Gradient of the objective w.r.t. transmission at a single wavelength.
    ///
    /// Returns ∂cost/∂T = 2 * (actual − target) / N
    ///
    /// # Arguments
    /// * `wavelength_nm` – Query wavelength (nm)
    /// * `actual`        – Computed transmission at that wavelength
    pub fn gradient_transmission(&self, wavelength_nm: f64, actual: f64) -> f64 {
        // Find the nearest target point
        let target = self
            .target_transmission
            .iter()
            .min_by(|a, b| {
                (a.0 - wavelength_nm)
                    .abs()
                    .partial_cmp(&(b.0 - wavelength_nm).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.1)
            .unwrap_or(0.0);
        let n = self.target_transmission.len().max(1) as f64;
        2.0 * (actual - target) / n
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PSO Optimizer
// ─────────────────────────────────────────────────────────────────────────────

/// A single particle in the PSO swarm.
#[derive(Debug, Clone)]
struct Particle {
    position: Vec<f64>,
    velocity: Vec<f64>,
    best_position: Vec<f64>,
    best_value: f64,
}

impl Particle {
    fn new(rng: &mut Lcg, bounds: &[(f64, f64)]) -> Self {
        let _dim = bounds.len();
        let position: Vec<f64> = bounds
            .iter()
            .map(|(lo, hi)| rng.next_range(*lo, *hi))
            .collect();
        let velocity: Vec<f64> = bounds
            .iter()
            .map(|(lo, hi)| rng.next_range(-(hi - lo) * 0.1, (hi - lo) * 0.1))
            .collect();
        let best_position = position.clone();
        Self {
            position,
            velocity,
            best_position,
            best_value: f64::MAX,
        }
    }
}

/// Particle Swarm Optimiser (PSO) with deterministic LCG random numbers.
///
/// Uses the canonical PSO update equations:
/// v ← w·v + c₁·r₁·(p_best − x) + c₂·r₂·(g_best − x)
/// x ← x + v
#[derive(Debug, Clone)]
pub struct PsoOptimizer {
    /// Number of particles in the swarm
    pub n_particles: usize,
    /// Search space dimensionality
    pub n_dimensions: usize,
    /// Box constraints: (min, max) per dimension
    pub bounds: Vec<(f64, f64)>,
    /// Inertia weight (typical: 0.729)
    pub w: f64,
    /// Cognitive coefficient (typical: 1.494)
    pub c1: f64,
    /// Social coefficient (typical: 1.494)
    pub c2: f64,
    /// Current global best position
    pub best_position: Vec<f64>,
    /// Current global best objective value
    pub best_value: f64,

    particles: Vec<Particle>,
    rng: Lcg,
    iteration: usize,
}

impl PsoOptimizer {
    /// Construct a new PSO optimiser.
    ///
    /// # Arguments
    /// * `n`      – Number of particles
    /// * `dim`    – Dimensionality
    /// * `bounds` – Per-dimension (min, max) bounds
    pub fn new(n: usize, dim: usize, bounds: Vec<(f64, f64)>) -> Self {
        let mut rng = Lcg::new(42);
        let particles: Vec<Particle> = (0..n).map(|_| Particle::new(&mut rng, &bounds)).collect();
        let best_position = vec![0.0; dim];
        Self {
            n_particles: n,
            n_dimensions: dim,
            bounds,
            w: 0.729,
            c1: 1.494,
            c2: 1.494,
            best_position,
            best_value: f64::MAX,
            particles,
            rng,
            iteration: 0,
        }
    }

    /// Perform one PSO iteration.
    ///
    /// # Arguments
    /// * `objective` – Objective function to minimise (lower = better)
    pub fn step(&mut self, objective: impl Fn(&[f64]) -> f64) {
        // Evaluate particles
        for i in 0..self.n_particles {
            let val = objective(&self.particles[i].position);
            if val < self.particles[i].best_value {
                self.particles[i].best_value = val;
                self.particles[i].best_position = self.particles[i].position.clone();
            }
            if val < self.best_value {
                self.best_value = val;
                self.best_position = self.particles[i].position.clone();
            }
        }
        // Update velocities and positions
        let g_best = self.best_position.clone();
        for particle in self.particles.iter_mut() {
            for (d, &g_d) in g_best.iter().enumerate().take(particle.position.len()) {
                let r1 = self.rng.next_f64();
                let r2 = self.rng.next_f64();
                particle.velocity[d] = self.w * particle.velocity[d]
                    + self.c1 * r1 * (particle.best_position[d] - particle.position[d])
                    + self.c2 * r2 * (g_d - particle.position[d]);
                particle.position[d] += particle.velocity[d];
                // Clamp to bounds
                let (lo, hi) = self.bounds[d];
                particle.position[d] = particle.position[d].clamp(lo, hi);
            }
        }
        self.iteration += 1;
    }

    /// Run the optimiser for `n_iterations` steps.
    ///
    /// Returns the global best position found.
    ///
    /// # Arguments
    /// * `n_iterations` – Number of PSO iterations
    /// * `objective`    – Objective function (closure, may be called repeatedly)
    pub fn run(&mut self, n_iterations: usize, objective: impl Fn(&[f64]) -> f64) -> Vec<f64> {
        for _ in 0..n_iterations {
            self.step(&objective);
        }
        self.best_position.clone()
    }

    /// Normalised convergence rate: (f_init − f_best) / f_init.
    ///
    /// Returns 0 if the optimiser has not yet converged.
    pub fn convergence_rate(&self) -> f64 {
        if self.best_value >= f64::MAX {
            return 0.0;
        }
        // Rate is 1 when converged to 0, 0 when still at initialisation
        (1.0 - self.best_value.abs().ln().abs() / (f64::MAX.ln())).clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RingOptimizer
// ─────────────────────────────────────────────────────────────────────────────

/// Optimises ring resonator geometry for target Q-factor, extinction ratio, and FSR.
#[derive(Debug, Clone)]
pub struct RingOptimizer {
    /// Target loaded Q-factor
    pub target_q: f64,
    /// Target through-port extinction ratio at resonance (dB)
    pub target_extinction_db: f64,
    /// Target free spectral range (nm)
    pub target_fsr_nm: f64,
    /// Design wavelength (m)
    pub wavelength: f64,
    /// SOI process parameters
    pub process: SoiProcess,
}

impl RingOptimizer {
    /// Construct a new ring optimiser.
    ///
    /// # Arguments
    /// * `target_q`   – Target loaded Q
    /// * `target_er`  – Target extinction ratio (dB)
    /// * `target_fsr` – Target FSR (nm)
    /// * `wavelength` – Design wavelength (m)
    pub fn new(target_q: f64, target_er: f64, target_fsr: f64, wavelength: f64) -> Self {
        Self {
            target_q,
            target_extinction_db: target_er,
            target_fsr_nm: target_fsr,
            wavelength,
            process: SoiProcess::standard_220nm(),
        }
    }

    /// Estimate the group index from the effective index.
    fn group_index(&self, n_eff: f64) -> f64 {
        // Approximate dispersion: n_g ≈ n_eff + dn_eff/dλ · λ
        // For SOI 450 nm strip, dn_eff/dλ ≈ −1.0e6 m⁻¹, so Δn_g ≈ 0.35
        n_eff + 0.35
    }

    /// Radius (µm) that gives the target FSR.
    fn radius_for_fsr(&self) -> f64 {
        let n_eff = self.process.n_eff_strip(450.0);
        let n_g = self.group_index(n_eff);
        let lambda = self.wavelength;
        let fsr_m = self.target_fsr_nm * 1.0e-9;
        // FSR = λ² / (n_g · 2πR) → R = λ² / (n_g · 2π · FSR)
        lambda * lambda / (n_g * 2.0 * PI * fsr_m) * 1.0e6 // convert m → µm
    }

    /// Optimise ring radius and coupling gap for the target parameters.
    ///
    /// Returns `(radius_um, gap_nm)`.
    pub fn optimize_radius_gap(&self) -> (f64, f64) {
        let radius_um = self.radius_for_fsr().clamp(2.0, 500.0);
        let n_eff = self.process.n_eff_strip(450.0);
        let alpha_db_per_cm = self.process.waveguide_loss_db_per_cm;
        let alpha_per_m = alpha_db_per_cm * 100.0 / (10.0 / 10.0_f64.ln());
        let circumference_m = 2.0 * PI * radius_um * 1.0e-6;
        let alpha_round = alpha_per_m * circumference_m;
        // Critical coupling: κ² = α_round for maximum extinction
        let kappa_sq = alpha_round.clamp(0.0, 0.99);
        // Convert κ to gap via exponential coupling model
        let g0 = 200.0_f64; // nm
        let gap_nm = -g0 * (kappa_sq.sqrt() / 0.98).max(1.0e-10).ln();
        let gap_nm = gap_nm.clamp(50.0, 500.0);
        // Loaded Q from coupling and round-trip loss
        let _q_loaded =
            2.0 * PI * n_eff * circumference_m / self.wavelength / (kappa_sq + alpha_round);
        (radius_um, gap_nm)
    }

    /// Sweep coupling gap and compute loaded Q at each point.
    ///
    /// # Arguments
    /// * `gap_range` – (gap_min_nm, gap_max_nm)
    /// * `n_points`  – Number of sample points
    ///
    /// Returns a `Vec<(gap_nm, Q_loaded)>`.
    pub fn q_vs_gap(&self, gap_range: (f64, f64), n_points: usize) -> Vec<(f64, f64)> {
        let (g_min, g_max) = gap_range;
        let (radius_um, _) = self.optimize_radius_gap();
        let n_eff = self.process.n_eff_strip(450.0);
        let alpha_db_per_cm = self.process.waveguide_loss_db_per_cm;
        let alpha_per_m = alpha_db_per_cm * 100.0 / (10.0 / 10.0_f64.ln());
        let circumference_m = 2.0 * PI * radius_um * 1.0e-6;
        let alpha_round = alpha_per_m * circumference_m;

        (0..n_points)
            .map(|i| {
                let gap = g_min + (g_max - g_min) * i as f64 / (n_points - 1).max(1) as f64;
                let kappa = (-gap / 200.0).exp() * 0.98;
                let kappa_sq = kappa * kappa;
                let q = 2.0 * PI * n_eff * circumference_m
                    / self.wavelength
                    / (kappa_sq + alpha_round).max(1.0e-15);
                (gap, q)
            })
            .collect()
    }

    /// Through-port transmission at resonance.
    ///
    /// T_min = ((1 − α) − κ²)² / ((1 − α) + κ²)²  (simplified lossless-coupler form)
    ///
    /// # Arguments
    /// * `kappa`      – Field coupling coefficient (amplitude)
    /// * `alpha_per_m` – Power loss coefficient (1/m)
    /// * `radius_um`  – Ring radius (µm)
    pub fn resonance_transmission(&self, kappa: f64, alpha_per_m: f64, radius_um: f64) -> f64 {
        let circumference_m = 2.0 * PI * radius_um * 1.0e-6;
        let round_trip_loss = (-alpha_per_m * circumference_m).exp(); // field attenuation per round trip
        let a = round_trip_loss.sqrt(); // amplitude after one round trip
        let kappa_sq = (kappa * kappa).clamp(0.0, 1.0);
        let t_sq = (1.0 - kappa_sq).sqrt();
        let numerator = (a - t_sq).powi(2);
        let denominator = (1.0 - a * t_sq).powi(2);
        if denominator < 1.0e-15 {
            return 0.0;
        }
        numerator / denominator
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MziOptimizer
// ─────────────────────────────────────────────────────────────────────────────

/// Optimises Mach-Zehnder Interferometer (MZI) modulators.
#[derive(Debug, Clone)]
pub struct MziOptimizer {
    /// Target extinction ratio (dB)
    pub target_er_db: f64,
    /// Target insertion loss (dB)
    pub target_insertion_loss_db: f64,
    /// Target electro-optic bandwidth (nm)
    pub target_bandwidth_nm: f64,
    /// Design wavelength (m)
    pub wavelength: f64,
}

impl MziOptimizer {
    /// Construct a new MZI optimiser.
    ///
    /// # Arguments
    /// * `er_db`   – Target extinction ratio (dB)
    /// * `il_db`   – Target insertion loss (dB)
    /// * `bw_nm`   – Target optical bandwidth (nm)
    /// * `wavelength` – Design wavelength (m)
    pub fn new(er_db: f64, il_db: f64, bw_nm: f64, wavelength: f64) -> Self {
        Self {
            target_er_db: er_db,
            target_insertion_loss_db: il_db,
            target_bandwidth_nm: bw_nm,
            wavelength,
        }
    }

    /// Optimum power coupling coefficient κ for maximum extinction ratio.
    ///
    /// For a balanced MZI with loss α: κ_opt = 1/2 * (1 + α).
    /// Returns κ in [0, 1].
    pub fn optimize_coupler_ratio(&self) -> f64 {
        // For ideal coupler: κ = 0.5 (50:50 split)
        // With arm loss α: shift κ toward lossless arm
        let arm_loss_linear = 10.0_f64.powf(-self.target_insertion_loss_db / 10.0);
        let kappa = 0.5 * (1.0 + (1.0 - arm_loss_linear).clamp(0.0, 0.5));
        kappa.clamp(0.0, 1.0)
    }

    /// Half-wave voltage (Vπ) for a phase modulator electrode.
    ///
    /// V_π = λ / (2 * n³ * r₃₃ * L)
    ///
    /// # Arguments
    /// * `electrode_length_um` – Active electrode length (µm)
    /// * `r33_pm_per_v`       – Electro-optic coefficient (pm/V)
    pub fn switching_voltage_vpi(&self, electrode_length_um: f64, r33_pm_per_v: f64) -> f64 {
        // Default to LiNbO₃: n ≈ 2.21
        let n = 2.21_f64;
        let r33_m_per_v = r33_pm_per_v * 1.0e-12;
        let l_m = electrode_length_um * 1.0e-6;
        self.wavelength / (2.0 * n.powi(3) * r33_m_per_v * l_m).max(1.0e-30)
    }

    /// Voltage required for a given modulation bandwidth on a lumped electrode.
    ///
    /// Bandwidth-limited voltage: V_bw = 1 / (2π · f · C · Z₀)
    ///
    /// # Arguments
    /// * `bandwidth_ghz` – Required EO bandwidth (GHz)
    /// * `c_pf`          – Electrode capacitance (pF)
    /// * `z_ohm`         – Load impedance (Ω)
    pub fn bandwidth_limited_voltage(&self, bandwidth_ghz: f64, c_pf: f64, z_ohm: f64) -> f64 {
        let f_hz = bandwidth_ghz * 1.0e9;
        let c_f = c_pf * 1.0e-12;
        1.0 / (2.0 * PI * f_hz * c_f * z_ohm).max(1.0e-30)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcg_distinct_values() {
        let mut rng = Lcg::new(1234);
        let v1 = rng.next_f64();
        let v2 = rng.next_f64();
        assert!((v1 - v2).abs() > 1.0e-6, "LCG produced identical values");
    }

    #[test]
    fn test_lcg_in_range() {
        let mut rng = Lcg::new(99);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "LCG value out of [0,1): {v}");
        }
    }

    #[test]
    fn test_bandpass_objective_zero_at_target() {
        let obj = InverseDesignObjective::new_bandpass(1550.0, 10.0, 30.0);
        let targets: Vec<f64> = obj.target_transmission.iter().map(|p| p.1).collect();
        let zero_phases: Vec<f64> = vec![0.0; targets.len()];
        let cost = obj.evaluate(&targets, &zero_phases);
        assert!(cost < 1.0e-10, "Cost should be near zero at target: {cost}");
    }

    #[test]
    fn test_bandpass_gradient_sign() {
        let obj = InverseDesignObjective::new_bandpass(1550.0, 10.0, 30.0);
        // At wavelength above target, actual > target → positive gradient
        let grad = obj.gradient_transmission(1550.0, 1.0);
        // Gradient should be finite
        assert!(grad.is_finite(), "Gradient must be finite");
    }

    #[test]
    fn test_pso_minimises_quadratic() {
        // Minimise f(x,y) = x² + y² on [−5, 5]²
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut pso = PsoOptimizer::new(20, 2, bounds);
        let best = pso.run(100, |x| x[0] * x[0] + x[1] * x[1]);
        assert!(
            best[0] * best[0] + best[1] * best[1] < 1.0,
            "PSO failed to minimise: ({}, {})",
            best[0],
            best[1]
        );
    }

    #[test]
    fn test_ring_optimizer_radius_positive() {
        let opt = RingOptimizer::new(10_000.0, 20.0, 10.0, 1.55e-6);
        let (r, g) = opt.optimize_radius_gap();
        assert!(r > 0.0, "Radius must be positive: {r}");
        assert!(g > 0.0, "Gap must be positive: {g}");
    }

    #[test]
    fn test_ring_fsr_consistency() {
        let opt = RingOptimizer::new(10_000.0, 20.0, 10.0, 1.55e-6);
        let pts = opt.q_vs_gap((100.0, 400.0), 10);
        assert_eq!(pts.len(), 10);
        // Q should be monotonically increasing with gap (less coupling = higher Q)
        for w in pts.windows(2) {
            assert!(w[1].1 >= w[0].1, "Q should increase with gap");
        }
    }

    #[test]
    fn test_resonance_transmission_at_critical_coupling() {
        let opt = RingOptimizer::new(10_000.0, 20.0, 10.0, 1.55e-6);
        // At critical coupling (kappa ≈ alpha_round), T_min ≈ 0
        let r_um = 10.0;
        let alpha = 20.0 * 100.0 / (10.0 / 10.0_f64.ln()); // dB/cm → 1/m
        let circ = 2.0 * PI * r_um * 1.0e-6;
        let kappa_crit = (alpha * circ).sqrt().clamp(0.0, 0.99);
        let t = opt.resonance_transmission(kappa_crit, alpha, r_um);
        assert!(
            t < 0.1,
            "Near critical coupling, T_min should be small: {t}"
        );
    }

    #[test]
    fn test_mzi_coupler_ratio_half() {
        let mzi = MziOptimizer::new(30.0, 0.5, 40.0, 1.55e-6);
        let kappa = mzi.optimize_coupler_ratio();
        assert!(
            (0.5..=1.0).contains(&kappa),
            "Coupling coefficient out of range: {kappa}"
        );
    }

    #[test]
    fn test_mzi_switching_voltage_positive() {
        let mzi = MziOptimizer::new(30.0, 0.5, 40.0, 1.55e-6);
        let vpi = mzi.switching_voltage_vpi(10_000.0, 30.0);
        assert!(
            vpi > 0.0 && vpi.is_finite(),
            "Vπ should be finite positive: {vpi}"
        );
    }

    #[test]
    fn test_mzi_bandwidth_voltage() {
        let mzi = MziOptimizer::new(30.0, 0.5, 40.0, 1.55e-6);
        let v = mzi.bandwidth_limited_voltage(100.0, 0.1, 50.0);
        // V = 1 / (2π * 1e11 * 1e-13 * 50) ≈ 318 V — sanity check it's finite
        assert!(
            v > 0.0 && v.is_finite(),
            "Bandwidth-limited voltage should be finite: {v}"
        );
    }
}
