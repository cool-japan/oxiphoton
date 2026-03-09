/// Transfer matrix method (TMM) for multilayer thin film optics.
///
/// Implements the characteristic matrix formalism for computing reflectance,
/// transmittance, absorptance, reflection phase, group delay, group delay
/// dispersion, and electric field distributions in layered media.
///
/// # Sign and admittance conventions
///
/// Following Heavens, "Optical Properties of Thin Solid Films" (1955) and
/// Macleod, "Thin-Film Optical Filters" (4th ed):
///
/// - Phase convention: exp(+iωt − ikz) for forward-propagating wave
/// - TE (s) admittance: η_s = n cos θ
/// - TM (p) admittance: η_p = n / cos θ
/// - Power transmittance: T = 4 Re(η_inc) Re(η_sub) / |η_inc B + C|²
/// - Power reflectance:   R = |(η_inc B − C) / (η_inc B + C)|²
///
/// These formulae satisfy R + T = 1 for lossless (real index) stacks at any
/// angle of incidence and for both polarisations.
use std::f64::consts::PI;

use num_complex::Complex64;

// Suppress unused import warning: OxiPhotonError is used only in Result type alias
#[allow(unused_imports)]
use crate::error::{OxiPhotonError, Result};

// ─── Polarization ────────────────────────────────────────────────────────────

/// Polarization state for transfer matrix calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarization {
    /// Transverse-electric (s-polarisation): E field perpendicular to plane of incidence.
    TE,
    /// Transverse-magnetic (p-polarisation): E field in plane of incidence.
    TM,
    /// Incoherent (power) average of TE and TM.
    Average,
}

// ─── Layer ───────────────────────────────────────────────────────────────────

/// A single homogeneous layer in a multilayer optical stack.
#[derive(Debug, Clone)]
pub struct Layer {
    /// Complex refractive index  ñ = n + ik.  Imaginary part k > 0 → absorption.
    pub n: Complex64,
    /// Physical thickness in nanometres.
    pub thickness_nm: f64,
    /// Human-readable label (e.g. "SiO2", "TiO2").
    pub name: String,
}

impl Layer {
    /// Create a layer with an arbitrary complex index.
    pub fn new(n: impl Into<Complex64>, thickness_nm: f64, name: impl Into<String>) -> Self {
        Self {
            n: n.into(),
            thickness_nm,
            name: name.into(),
        }
    }

    /// Create a lossless (real-index) layer.
    pub fn lossless(n_real: f64, thickness_nm: f64, name: impl Into<String>) -> Self {
        Self::new(Complex64::new(n_real, 0.0), thickness_nm, name)
    }

    /// Create a quarter-wave optical-thickness layer at `lambda_nm`.
    ///
    /// Physical thickness = λ / (4n).
    pub fn quarter_wave(n_real: f64, lambda_nm: f64, name: impl Into<String>) -> Self {
        let d = lambda_nm / (4.0 * n_real);
        Self::lossless(n_real, d, name)
    }

    /// Create a half-wave optical-thickness layer at `lambda_nm`.
    ///
    /// Physical thickness = λ / (2n).
    pub fn half_wave(n_real: f64, lambda_nm: f64, name: impl Into<String>) -> Self {
        let d = lambda_nm / (2.0 * n_real);
        Self::lossless(n_real, d, name)
    }

    /// Optical thickness n·d (nm).
    pub fn optical_thickness_nm(&self) -> f64 {
        self.n.re * self.thickness_nm
    }
}

// ─── Snell's law & Fresnel coefficients ──────────────────────────────────────

/// Snell's law: compute cos θ₂ for refraction from medium n₁ at angle θ₁ into n₂.
///
/// Returns the complex cosine of θ₂ with non-negative real part (branch cut
/// chosen for physical decay of evanescent waves).
pub fn snell_cos(n1: Complex64, theta1_rad: f64, n2: Complex64) -> Complex64 {
    let sin_t1 = Complex64::new(theta1_rad.sin(), 0.0);
    let sin_t2 = n1 / n2 * sin_t1;
    let cos_t2 = (Complex64::new(1.0, 0.0) - sin_t2 * sin_t2).sqrt();
    if cos_t2.re < 0.0 {
        -cos_t2
    } else {
        cos_t2
    }
}

/// Snell's law refraction angle in medium 2 (complex cosine).
///
/// Provided for API completeness; identical to `snell_cos`.
pub fn snell_law(n1: Complex64, theta1_rad: f64, n2: Complex64) -> Complex64 {
    snell_cos(n1, theta1_rad, n2)
}

/// Layer admittance η (dimensionless, Z₀ = 1 units).
///
/// - TE: η = n cos θ
/// - TM: η = n / cos θ
fn admittance(n: Complex64, cos_theta: Complex64, pol: Polarization) -> Complex64 {
    match pol {
        Polarization::TE | Polarization::Average => n * cos_theta,
        Polarization::TM => n / cos_theta,
    }
}

/// Fresnel reflection amplitude r_s (TE / s) at a single interface.
///
/// Convention (TE admittance η = n cos θ):
/// r_s = (η₁ − η₂) / (η₁ + η₂) = (n₁ cos θ₁ − n₂ cos θ₂) / (n₁ cos θ₁ + n₂ cos θ₂)
pub fn fresnel_r_s(n1: Complex64, n2: Complex64, theta1: f64) -> Complex64 {
    let cos_t1 = Complex64::new(theta1.cos(), 0.0);
    let cos_t2 = snell_cos(n1, theta1, n2);
    let eta1 = n1 * cos_t1;
    let eta2 = n2 * cos_t2;
    (eta1 - eta2) / (eta1 + eta2)
}

/// Fresnel reflection amplitude r_p (TM / p) at a single interface.
///
/// Convention (TM admittance η = n / cos θ):
/// r_p = (η₁ − η₂) / (η₁ + η₂) = (n₁/cos θ₁ − n₂/cos θ₂) / (n₁/cos θ₁ + n₂/cos θ₂)
///
/// Note: this equals −r_p(Born&Wolf) due to the admittance convention chosen
/// to ensure self-consistency with the characteristic matrix method.
pub fn fresnel_r_p(n1: Complex64, n2: Complex64, theta1: f64) -> Complex64 {
    let cos_t1 = Complex64::new(theta1.cos(), 0.0);
    let cos_t2 = snell_cos(n1, theta1, n2);
    let eta1 = n1 / cos_t1;
    let eta2 = n2 / cos_t2;
    (eta1 - eta2) / (eta1 + eta2)
}

/// Fresnel transmission amplitude t_s (TE / s) at a single interface.
///
/// t_s = 2 η₁ / (η₁ + η₂) = 2 n₁ cos θ₁ / (n₁ cos θ₁ + n₂ cos θ₂)
pub fn fresnel_t_s(n1: Complex64, n2: Complex64, theta1: f64) -> Complex64 {
    let cos_t1 = Complex64::new(theta1.cos(), 0.0);
    let cos_t2 = snell_cos(n1, theta1, n2);
    let eta1 = n1 * cos_t1;
    let eta2 = n2 * cos_t2;
    (Complex64::new(2.0, 0.0) * eta1) / (eta1 + eta2)
}

// ─── MultilayerStack ─────────────────────────────────────────────────────────

/// Multilayer optical stack computed using the characteristic matrix method.
///
/// The stack is ordered from the first deposited layer (closest to the incident
/// medium) to the last layer (closest to the substrate).  The incident medium
/// and substrate are semi-infinite.
pub struct MultilayerStack {
    /// Layers from incident-side to substrate-side.
    pub layers: Vec<Layer>,
    /// Refractive index of the incident (ambient) medium.
    pub n_incident: Complex64,
    /// Refractive index of the exit substrate medium.
    pub n_substrate: Complex64,
}

impl MultilayerStack {
    /// Create an empty stack with the given incident and substrate indices.
    pub fn new(n_incident: impl Into<Complex64>, n_substrate: impl Into<Complex64>) -> Self {
        Self {
            layers: Vec::new(),
            n_incident: n_incident.into(),
            n_substrate: n_substrate.into(),
        }
    }

    /// Append a single layer to the stack (nearest to substrate side).
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    /// Append multiple layers in order.
    pub fn add_layers(&mut self, layers: Vec<Layer>) {
        self.layers.extend(layers);
    }

    /// Compute the 2×2 characteristic matrix for a single layer at (λ, θ, pol).
    ///
    /// ```text
    /// M = [ cos δ          −i sin δ / η ]
    ///     [ −i η sin δ      cos δ       ]
    /// ```
    /// where δ = 2π n d cos θ_layer / λ  (phase thickness in the layer)
    /// and η is the layer admittance (TE: n cos θ; TM: n / cos θ).
    ///
    /// The refraction angle in the layer is obtained via Snell's law from the
    /// incident medium angle `theta_rad`.
    pub fn layer_matrix(
        &self,
        layer: &Layer,
        lambda_nm: f64,
        theta_rad: f64,
        polarization: Polarization,
    ) -> [[Complex64; 2]; 2] {
        // For Average, caller handles TE/TM separately; use TE here as fallback.
        let pol = match polarization {
            Polarization::Average => Polarization::TE,
            p => p,
        };

        // Refraction angle in layer from incident medium
        let cos_t = snell_cos(self.n_incident, theta_rad, layer.n);

        // Phase thickness δ = 2π n d cosθ / λ
        let delta =
            Complex64::new(2.0 * PI / lambda_nm, 0.0) * layer.n * cos_t * layer.thickness_nm;

        let cos_delta = delta.cos();
        let sin_delta = delta.sin();
        let eta = admittance(layer.n, cos_t, pol);

        let i = Complex64::new(0.0, 1.0);

        [
            [cos_delta, -i * sin_delta / eta],
            [-i * eta * sin_delta, cos_delta],
        ]
    }

    /// Multiply two 2×2 complex matrices A · B.
    fn mat_mul(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
        let zero = Complex64::new(0.0, 0.0);
        let mut c = [[zero; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        c
    }

    /// 2×2 identity matrix.
    fn mat_identity() -> [[Complex64; 2]; 2] {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        [[one, zero], [zero, one]]
    }

    /// Compute the product characteristic matrix M = M₁ · M₂ · … · Mₙ for a
    /// single polarisation (must be TE or TM, not Average).
    fn total_matrix_single_pol(
        &self,
        lambda_nm: f64,
        theta_rad: f64,
        pol: Polarization,
    ) -> [[Complex64; 2]; 2] {
        let mut m = Self::mat_identity();
        for layer in &self.layers {
            let mi = self.layer_matrix(layer, lambda_nm, theta_rad, pol);
            m = Self::mat_mul(&m, &mi);
        }
        m
    }

    /// Total characteristic matrix (product of all layer matrices).
    ///
    /// For `Polarization::Average` this returns the TE matrix; use the public
    /// API methods (`reflectance`, `transmittance`, …) which handle averaging.
    pub fn total_matrix(
        &self,
        lambda_nm: f64,
        theta_rad: f64,
        polarization: Polarization,
    ) -> [[Complex64; 2]; 2] {
        let pol = match polarization {
            Polarization::Average => Polarization::TE,
            p => p,
        };
        self.total_matrix_single_pol(lambda_nm, theta_rad, pol)
    }

    /// Admittances of the incident and substrate media for a given polarisation.
    fn medium_admittances(&self, theta_rad: f64, pol: Polarization) -> (Complex64, Complex64) {
        let cos_inc = Complex64::new(theta_rad.cos(), 0.0);
        let eta_inc = admittance(self.n_incident, cos_inc, pol);

        let cos_sub = snell_cos(self.n_incident, theta_rad, self.n_substrate);
        let eta_sub = admittance(self.n_substrate, cos_sub, pol);

        (eta_inc, eta_sub)
    }

    /// Compute the B and C scalars from the characteristic matrix.
    ///
    /// Given the total matrix M and substrate admittance η_sub:
    /// B = M[0][0] + M[0][1] · η_sub
    /// C = M[1][0] + M[1][1] · η_sub
    fn b_c(m: &[[Complex64; 2]; 2], eta_sub: Complex64) -> (Complex64, Complex64) {
        let b = m[0][0] + m[0][1] * eta_sub;
        let c = m[1][0] + m[1][1] * eta_sub;
        (b, c)
    }

    /// Amplitude reflection coefficient r for a single polarisation.
    ///
    /// r = (η_inc B − C) / (η_inc B + C)
    fn reflection_amplitude_single(
        &self,
        lambda_nm: f64,
        theta_rad: f64,
        pol: Polarization,
    ) -> Complex64 {
        let m = self.total_matrix_single_pol(lambda_nm, theta_rad, pol);
        let (eta_inc, eta_sub) = self.medium_admittances(theta_rad, pol);
        let (b, c) = Self::b_c(&m, eta_sub);
        (eta_inc * b - c) / (eta_inc * b + c)
    }

    /// Power reflectance R = |r|² for a single polarisation.
    fn reflectance_single(&self, lambda_nm: f64, theta_rad: f64, pol: Polarization) -> f64 {
        let r = self.reflection_amplitude_single(lambda_nm, theta_rad, pol);
        r.norm_sqr()
    }

    /// Power transmittance T for a single polarisation (Heavens formula).
    ///
    /// T = 4 Re(η_inc) Re(η_sub) / |η_inc B + C|²
    ///
    /// This satisfies R + T = 1 for any lossless stack at any angle.
    fn transmittance_single(&self, lambda_nm: f64, theta_rad: f64, pol: Polarization) -> f64 {
        let m = self.total_matrix_single_pol(lambda_nm, theta_rad, pol);
        let (eta_inc, eta_sub) = self.medium_admittances(theta_rad, pol);
        let (b, c) = Self::b_c(&m, eta_sub);
        let denom = (eta_inc * b + c).norm_sqr();
        if denom < f64::EPSILON {
            return 0.0;
        }
        4.0 * eta_inc.re * eta_sub.re / denom
    }

    /// Amplitude reflection coefficient r at (λ, θ, pol).
    ///
    /// For `Average`, returns the power-averaged mean of TE and TM amplitudes
    /// (approximate — prefer `reflectance` for energy-conserving average).
    pub fn reflection_amplitude(
        &self,
        lambda_nm: f64,
        theta_rad: f64,
        polarization: Polarization,
    ) -> Complex64 {
        match polarization {
            Polarization::Average => {
                let r_te = self.reflection_amplitude_single(lambda_nm, theta_rad, Polarization::TE);
                let r_tm = self.reflection_amplitude_single(lambda_nm, theta_rad, Polarization::TM);
                (r_te + r_tm) * Complex64::new(0.5, 0.0)
            }
            p => self.reflection_amplitude_single(lambda_nm, theta_rad, p),
        }
    }

    /// Power reflectance R = |r|².
    pub fn reflectance(&self, lambda_nm: f64, theta_rad: f64, polarization: Polarization) -> f64 {
        match polarization {
            Polarization::Average => {
                0.5 * (self.reflectance_single(lambda_nm, theta_rad, Polarization::TE)
                    + self.reflectance_single(lambda_nm, theta_rad, Polarization::TM))
            }
            p => self.reflectance_single(lambda_nm, theta_rad, p),
        }
    }

    /// Power transmittance T (accounts for beam cross-section and index change).
    ///
    /// Uses the Heavens formula `T = 4 Re(η_inc) Re(η_sub) / |η_inc B + C|²`
    /// which is energy-conserving for lossless stacks.
    pub fn transmittance(&self, lambda_nm: f64, theta_rad: f64, polarization: Polarization) -> f64 {
        match polarization {
            Polarization::Average => {
                0.5 * (self.transmittance_single(lambda_nm, theta_rad, Polarization::TE)
                    + self.transmittance_single(lambda_nm, theta_rad, Polarization::TM))
            }
            p => self.transmittance_single(lambda_nm, theta_rad, p),
        }
    }

    /// Absorptance A = 1 − R − T  (clamped to [0, 1]).
    pub fn absorptance(&self, lambda_nm: f64, theta_rad: f64, polarization: Polarization) -> f64 {
        let r = self.reflectance(lambda_nm, theta_rad, polarization);
        let t = self.transmittance(lambda_nm, theta_rad, polarization);
        (1.0 - r - t).clamp(0.0, 1.0)
    }

    /// Phase of reflected wave φ = arg(r) in radians ∈ (−π, π].
    pub fn reflection_phase_rad(
        &self,
        lambda_nm: f64,
        theta_rad: f64,
        polarization: Polarization,
    ) -> f64 {
        self.reflection_amplitude(lambda_nm, theta_rad, polarization)
            .arg()
    }

    /// Group delay GD = −dφ/dω computed by central finite difference in ω.
    ///
    /// Returns GD in femtoseconds.  Uses Δλ = 0.1 nm step.
    pub fn group_delay_fs(
        &self,
        lambda_nm: f64,
        theta_rad: f64,
        polarization: Polarization,
    ) -> f64 {
        let dl = 0.1_f64; // nm
        let l1 = (lambda_nm - dl).max(1.0);
        let l2 = lambda_nm + dl;

        // c in nm/fs
        let c_nm_per_fs = 299.792_458_f64;
        // ω₁ corresponds to longer λ (lower freq), ω₂ to shorter λ (higher freq)
        let omega1 = 2.0 * PI * c_nm_per_fs / l2;
        let omega2 = 2.0 * PI * c_nm_per_fs / l1;
        let d_omega = omega2 - omega1;

        let phi1 = self.reflection_phase_rad(l2, theta_rad, polarization);
        let phi2 = self.reflection_phase_rad(l1, theta_rad, polarization);

        // Phase difference with 2π unwrapping
        let mut dphi = phi2 - phi1;
        while dphi > PI {
            dphi -= 2.0 * PI;
        }
        while dphi < -PI {
            dphi += 2.0 * PI;
        }

        -dphi / d_omega // fs
    }

    /// Group delay dispersion GDD = d²φ/dω² in fs² (central finite difference).
    pub fn group_delay_dispersion_fs2(
        &self,
        lambda_nm: f64,
        theta_rad: f64,
        polarization: Polarization,
    ) -> f64 {
        let dl = 0.5_f64; // nm
        let l1 = (lambda_nm - dl).max(1.0);
        let l2 = lambda_nm + dl;

        let c_nm_per_fs = 299.792_458_f64;
        let omega1 = 2.0 * PI * c_nm_per_fs / l2;
        let omega2 = 2.0 * PI * c_nm_per_fs / l1;
        let d_omega = (omega2 - omega1) / 2.0;

        let gd_lo = self.group_delay_fs(l2, theta_rad, polarization);
        let gd_hi = self.group_delay_fs(l1, theta_rad, polarization);

        (gd_hi - gd_lo) / (2.0 * d_omega) // fs²
    }

    /// Compute reflectance, transmittance, and absorptance spectrum.
    ///
    /// Returns `Vec<(lambda_nm, R, T, A)>` with `n_pts` uniformly spaced
    /// wavelengths from `lambda_min_nm` to `lambda_max_nm` (inclusive).
    pub fn spectrum(
        &self,
        lambda_min_nm: f64,
        lambda_max_nm: f64,
        n_pts: usize,
        theta_rad: f64,
        polarization: Polarization,
    ) -> Vec<(f64, f64, f64, f64)> {
        if n_pts == 0 {
            return Vec::new();
        }
        let denom = (n_pts.saturating_sub(1)).max(1) as f64;
        (0..n_pts)
            .map(|i| {
                let t = i as f64 / denom;
                let lambda = lambda_min_nm + t * (lambda_max_nm - lambda_min_nm);
                let r = self.reflectance(lambda, theta_rad, polarization);
                let tr = self.transmittance(lambda, theta_rad, polarization);
                let a = (1.0 - r - tr).clamp(0.0, 1.0);
                (lambda, r, tr, a)
            })
            .collect()
    }

    /// Compute the normalised electric field intensity |E(z)|² through the stack.
    ///
    /// Uses forward-propagation of the state vector `[B, C]` from the top of
    /// the first layer (z = 0) to the bottom of the last layer.
    ///
    /// Returns `Vec<(z_nm, |E|²)>` where z = 0 is the first interface.
    /// Total number of points = `layers.len() × n_pts_per_layer`.
    pub fn field_distribution(
        &self,
        lambda_nm: f64,
        theta_rad: f64,
        polarization: Polarization,
        n_pts_per_layer: usize,
    ) -> Vec<(f64, f64)> {
        let pol = match polarization {
            Polarization::Average => Polarization::TE,
            p => p,
        };

        let n_pts = n_pts_per_layer.max(2);
        let mut result = Vec::with_capacity(self.layers.len() * n_pts);

        // Compute r so we can initialise the field state at the first interface.
        let r = self.reflection_amplitude_single(lambda_nm, theta_rad, pol);
        let (eta_inc, _) = self.medium_admittances(theta_rad, pol);

        // At z = 0⁺ (top of first layer):
        //   B(0) = 1 + r   (total E field)
        //   C(0) = η_inc (1 − r)  (total H-like field × η)
        let b0 = Complex64::new(1.0, 0.0) + r;
        let c0 = eta_inc * (Complex64::new(1.0, 0.0) - r);

        let mut z_nm = 0.0_f64;
        let mut b_cur = b0;
        let mut c_cur = c0;

        for layer in &self.layers {
            let cos_t = snell_cos(self.n_incident, theta_rad, layer.n);
            let eta_layer = admittance(layer.n, cos_t, pol);
            let k_z = Complex64::new(2.0 * PI / lambda_nm, 0.0) * layer.n * cos_t;

            for p in 0..n_pts {
                let frac = p as f64 / (n_pts - 1) as f64;
                let dz = frac * layer.thickness_nm;
                let z_total = z_nm + dz;

                // Propagate [B, C] from layer start by dz using partial matrix:
                let delta = k_z * dz;
                let cos_d = delta.cos();
                let sin_d = delta.sin();
                let i = Complex64::new(0.0, 1.0);

                let b_z = cos_d * b_cur - i * sin_d / eta_layer * c_cur;
                result.push((z_total, b_z.norm_sqr()));
            }

            // Advance full state to end of layer
            let delta_full = k_z * layer.thickness_nm;
            let cos_d = delta_full.cos();
            let sin_d = delta_full.sin();
            let i = Complex64::new(0.0, 1.0);
            let b_next = cos_d * b_cur - i * sin_d / eta_layer * c_cur;
            let c_next = -i * eta_layer * sin_d * b_cur + cos_d * c_cur;
            b_cur = b_next;
            c_cur = c_next;
            z_nm += layer.thickness_nm;
        }

        result
    }

    /// Total optical thickness Σ nᵢ · dᵢ (nm).
    pub fn total_optical_thickness_nm(&self) -> f64 {
        self.layers.iter().map(|l| l.optical_thickness_nm()).sum()
    }

    /// Number of layers in the stack.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Analytic bare-interface reflectance at normal incidence (air/glass).
    fn bare_glass_r(n_sub: f64) -> f64 {
        let r = (1.0 - n_sub) / (1.0 + n_sub);
        r * r
    }

    // ── test 1: empty stack equals bare-interface Fresnel ─────────────────────
    #[test]
    fn test_empty_stack_r_equals_fresnel() {
        let n_glass = 1.52_f64;
        let stack = MultilayerStack::new(1.0_f64, n_glass);

        let r_stack = stack.reflectance(550.0, 0.0, Polarization::TE);
        let r_expected = bare_glass_r(n_glass);

        assert_abs_diff_eq!(r_stack, r_expected, epsilon = 1e-10);
    }

    // ── test 2: QW MgF2 on glass reduces reflection ───────────────────────────
    #[test]
    fn test_qw_single_layer_r_at_design() {
        let n_mgf2 = 1.38_f64;
        let n_glass = 1.52_f64;
        let lambda_nm = 550.0_f64;

        let mut stack = MultilayerStack::new(1.0_f64, n_glass);
        stack.add_layer(Layer::quarter_wave(n_mgf2, lambda_nm, "MgF2"));

        let r_coated = stack.reflectance(lambda_nm, 0.0, Polarization::TE);
        let r_bare = bare_glass_r(n_glass);

        assert!(
            r_coated < r_bare,
            "QW AR coating should reduce R: coated={r_coated:.6} bare={r_bare:.6}"
        );
    }

    // ── test 3: energy conservation R + T = 1 ────────────────────────────────
    #[test]
    fn test_energy_conservation() {
        let n_tio2 = 2.35_f64;
        let n_sio2 = 1.46_f64;
        let n_glass = 1.52_f64;
        let lambda_nm = 600.0_f64;

        let mut stack = MultilayerStack::new(1.0_f64, n_glass);
        for _ in 0..4 {
            stack.add_layer(Layer::quarter_wave(n_tio2, lambda_nm, "TiO2"));
            stack.add_layer(Layer::quarter_wave(n_sio2, lambda_nm, "SiO2"));
        }

        for lambda in [400.0_f64, 550.0, 600.0, 700.0, 900.0] {
            for theta in [0.0_f64, 0.1, 0.3, 0.5] {
                for &pol in &[Polarization::TE, Polarization::TM] {
                    let r = stack.reflectance(lambda, theta, pol);
                    let t = stack.transmittance(lambda, theta, pol);
                    let total = r + t;
                    assert!(
                        (total - 1.0).abs() < 1e-8,
                        "R+T={total:.10} ≠ 1 at λ={lambda} θ={theta} {pol:?}"
                    );
                }
            }
        }
    }

    // ── test 4: TE = TM at normal incidence ──────────────────────────────────
    #[test]
    fn test_normal_incidence_te_tm_equal() {
        let n_glass = 1.52_f64;
        let mut stack = MultilayerStack::new(1.0_f64, n_glass);
        stack.add_layer(Layer::lossless(1.38, 100.0, "AR"));

        let r_te = stack.reflectance(550.0, 0.0, Polarization::TE);
        let r_tm = stack.reflectance(550.0, 0.0, Polarization::TM);

        assert_abs_diff_eq!(r_te, r_tm, epsilon = 1e-10);
    }

    // ── test 5: reflection phase ≈ π for denser medium ───────────────────────
    #[test]
    fn test_reflection_phase_180_for_metal() {
        // Bare air → dense medium interface: r = (n_inc − n_sub)/(n_inc + n_sub) < 0
        // → phase = π
        let n_dense = 3.5_f64;
        let stack = MultilayerStack::new(1.0_f64, n_dense);
        let phase = stack.reflection_phase_rad(800.0, 0.0, Polarization::TE);

        assert!(
            (phase.abs() - PI).abs() < 1e-10,
            "Expected |φ| ≈ π, got φ={phase:.6}"
        );
    }

    // ── test 6: field distribution has correct number of points ──────────────
    #[test]
    fn test_field_distribution_length() {
        let n_glass = 1.52_f64;
        let mut stack = MultilayerStack::new(1.0_f64, n_glass);
        let n_layers = 3_usize;
        let pts = 10_usize;
        for i in 0..n_layers {
            stack.add_layer(Layer::lossless(
                1.5 + 0.1 * i as f64,
                100.0,
                format!("L{i}"),
            ));
        }

        let fd = stack.field_distribution(550.0, 0.0, Polarization::TE, pts);
        assert_eq!(fd.len(), n_layers * pts);
    }

    // ── test 7: optical thickness sum ─────────────────────────────────────────
    #[test]
    fn test_optical_thickness() {
        let mut stack = MultilayerStack::new(1.0_f64, 1.5_f64);
        stack.add_layer(Layer::lossless(1.46, 100.0, "SiO2"));
        stack.add_layer(Layer::lossless(2.35, 50.0, "TiO2"));

        let expected = 1.46 * 100.0 + 2.35 * 50.0;
        assert_abs_diff_eq!(
            stack.total_optical_thickness_nm(),
            expected,
            epsilon = 1e-10
        );
    }

    // ── test 8: empty stack = Fresnel (both polarisations) ───────────────────
    #[test]
    fn test_fresnel_single_interface() {
        let n1 = Complex64::new(1.0, 0.0);
        let n2 = Complex64::new(1.52, 0.0);
        let theta = 0.4_f64;

        // fresnel_r_s / fresnel_r_p use the SAME admittance convention as TMM
        let r_s = fresnel_r_s(n1, n2, theta);
        let r_p = fresnel_r_p(n1, n2, theta);

        let stack = MultilayerStack::new(n1, n2);
        let r_te = stack.reflection_amplitude_single(550.0, theta, Polarization::TE);
        let r_tm = stack.reflection_amplitude_single(550.0, theta, Polarization::TM);

        assert_abs_diff_eq!(r_s.re, r_te.re, epsilon = 1e-10);
        assert_abs_diff_eq!(r_s.im, r_te.im, epsilon = 1e-10);
        assert_abs_diff_eq!(r_p.re, r_tm.re, epsilon = 1e-10);
        assert_abs_diff_eq!(r_p.im, r_tm.im, epsilon = 1e-10);
    }
}
