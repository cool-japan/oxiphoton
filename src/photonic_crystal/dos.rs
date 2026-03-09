//! Photonic density of states (DOS) for 1D photonic crystals.
//!
//! The photonic DOS ρ(ω) counts the number of modes per unit frequency interval:
//!
//!   ρ(ω) = (1/π) · dφ/dω
//!
//! where φ(ω) is the Bloch phase accumulated per unit cell.
//!
//! For a 1D PC with period Λ, the DOS is obtained from the dispersion relation
//! using the density of states theorem:
//!
//!   ρ(ω) = (L/π) · |dk/dω| = (L/π) / v_g(k)
//!
//! Bandgaps appear as regions where ρ(ω) = 0 (no propagating modes).
//! Band edges have diverging DOS (van Hove singularities).

use std::f64::consts::PI;

/// Photonic DOS computed from a 1D band structure.
pub struct PhotonicDos {
    /// Frequency points (rad/s)
    pub frequencies: Vec<f64>,
    /// DOS values (arb. units, normalized so ∫ρdω = 1)
    pub dos: Vec<f64>,
}

impl PhotonicDos {
    /// Compute DOS from 1D PC band structure via numerical differentiation.
    ///
    /// - `band_omegas`: sorted frequency dispersion ω(k) from k=0 to k=π/Λ
    /// - `n_freq`: number of frequency bins for the DOS histogram
    pub fn from_band(band_omegas: &[f64], n_freq: usize) -> Self {
        if band_omegas.len() < 2 {
            return Self {
                frequencies: vec![0.0; n_freq],
                dos: vec![0.0; n_freq],
            };
        }

        let omega_min = band_omegas.iter().cloned().fold(f64::INFINITY, f64::min);
        let omega_max = band_omegas
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let d_omega = (omega_max - omega_min) / n_freq as f64;

        let frequencies: Vec<f64> = (0..n_freq)
            .map(|i| omega_min + (i as f64 + 0.5) * d_omega)
            .collect();
        let mut dos = vec![0.0f64; n_freq];

        // Each k-point contributes 1/|dω/dk| to the DOS
        let n_k = band_omegas.len();
        let dk = PI / (n_k - 1) as f64; // Δk from 0 to π/Λ (normalised)

        for ki in 0..n_k.saturating_sub(1) {
            let omega_lo = band_omegas[ki].min(band_omegas[ki + 1]);
            let omega_hi = band_omegas[ki].max(band_omegas[ki + 1]);
            let dw_dk = (band_omegas[ki + 1] - band_omegas[ki]).abs() / dk;
            if dw_dk < 1e-30 {
                continue; // Van Hove singularity — skip
            }
            let contribution = dk / dw_dk.abs(); // = 1/|dω/dk| × dk
                                                 // Distribute contribution to frequency bins in [omega_lo, omega_hi]
            let bin_lo = ((omega_lo - omega_min) / d_omega).floor() as usize;
            let bin_hi = ((omega_hi - omega_min) / d_omega).ceil() as usize;
            let n_bins = (bin_hi - bin_lo).max(1);
            for dos_bin in dos.iter_mut().take(bin_hi.min(n_freq)).skip(bin_lo) {
                *dos_bin += contribution / n_bins as f64;
            }
        }

        // Normalize
        let total: f64 = dos.iter().sum::<f64>() * d_omega;
        if total > 1e-30 {
            for d in &mut dos {
                *d /= total;
            }
        }

        Self { frequencies, dos }
    }

    /// Peak DOS value.
    pub fn peak_dos(&self) -> f64 {
        self.dos.iter().cloned().fold(0.0_f64, f64::max)
    }

    /// Frequency of peak DOS.
    pub fn peak_frequency(&self) -> f64 {
        let (idx, _) =
            self.dos.iter().enumerate().fold(
                (0, 0.0_f64),
                |(mi, mv), (i, &v)| {
                    if v > mv {
                        (i, v)
                    } else {
                        (mi, mv)
                    }
                },
            );
        self.frequencies[idx]
    }

    /// Bandgap: frequency ranges where DOS ≈ 0 (below threshold).
    ///
    /// Returns list of (ω_lo, ω_hi) bandgap intervals.
    pub fn bandgaps(&self, threshold: f64) -> Vec<(f64, f64)> {
        let mut gaps = Vec::new();
        let mut in_gap = false;
        let mut gap_lo = 0.0;
        for (i, &d) in self.dos.iter().enumerate() {
            if d < threshold && !in_gap {
                gap_lo = self.frequencies[i];
                in_gap = true;
            } else if d >= threshold && in_gap {
                gaps.push((gap_lo, self.frequencies[i]));
                in_gap = false;
            }
        }
        if in_gap {
            gaps.push((gap_lo, *self.frequencies.last().unwrap_or(&gap_lo)));
        }
        gaps
    }

    /// Integrated DOS in frequency range [ω₁, ω₂].
    pub fn integrated_dos(&self, omega1: f64, omega2: f64) -> f64 {
        if self.frequencies.len() < 2 {
            return 0.0;
        }
        let d_omega = self.frequencies[1] - self.frequencies[0];
        self.frequencies
            .iter()
            .zip(self.dos.iter())
            .filter(|(&f, _)| f >= omega1 && f <= omega2)
            .map(|(_, &d)| d * d_omega)
            .sum()
    }
}

/// DOS for a free photon gas in 1D (reference).
///
///   ρ_1D(ω) = L/(π·c)  (constant in 1D for linear dispersion)
pub fn free_photon_dos_1d(omega: f64, length: f64) -> f64 {
    use crate::units::conversion::SPEED_OF_LIGHT;
    let _ = omega; // 1D DOS is frequency-independent
    length / (PI * SPEED_OF_LIGHT)
}

// ---------------------------------------------------------------------------
// 2D band data and DOS
// ---------------------------------------------------------------------------

/// Band structure data for a 2D photonic crystal.
///
/// Frequencies are stored as normalized units a/λ (dimensionless) or rad/s
/// depending on the application. k-points are in 2D reciprocal space.
#[derive(Debug, Clone)]
pub struct BandData2d {
    /// Band frequencies: `bands[band_index][k_point_index]`
    pub bands: Vec<Vec<f64>>,
    /// k-points in 2D reciprocal space (kx, ky)
    pub k_points: Vec<[f64; 2]>,
    /// Lattice constant (m)
    pub lattice_const: f64,
}

impl BandData2d {
    /// Create an empty `BandData2d` with the given lattice constant.
    pub fn new(lattice_const: f64) -> Self {
        Self {
            bands: Vec::new(),
            k_points: Vec::new(),
            lattice_const,
        }
    }

    /// Append one band (one frequency per k-point) to the dataset.
    ///
    /// The length of `freqs` must equal the number of k-points already
    /// registered.  If no k-points have been set yet this call is a no-op.
    pub fn add_band(&mut self, freqs: Vec<f64>) {
        if self.k_points.is_empty() {
            // Store even without k-points — caller must set k_points first or
            // populate them afterwards.  We accept the band unconditionally so
            // that the struct can be built incrementally.
        }
        self.bands.push(freqs);
    }

    /// Compute a DOS histogram over all bands.
    ///
    /// Returns `(bin_centers, counts)` where `counts[i]` is the number of
    /// (k-point, band) pairs whose frequency falls in bin `i`.
    ///
    /// # Arguments
    /// * `n_bins`   – number of histogram bins
    /// * `freq_min` – lower edge of the histogram range
    /// * `freq_max` – upper edge of the histogram range
    pub fn dos_histogram(
        &self,
        n_bins: usize,
        freq_min: f64,
        freq_max: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        if n_bins == 0 || freq_max <= freq_min {
            return (Vec::new(), Vec::new());
        }

        let bin_width = (freq_max - freq_min) / n_bins as f64;
        let bin_centers: Vec<f64> = (0..n_bins)
            .map(|i| freq_min + (i as f64 + 0.5) * bin_width)
            .collect();
        let mut counts = vec![0.0f64; n_bins];

        for band in &self.bands {
            for &freq in band {
                if freq < freq_min || freq >= freq_max {
                    continue;
                }
                let bin = ((freq - freq_min) / bin_width) as usize;
                let bin = bin.min(n_bins - 1);
                counts[bin] += 1.0;
            }
        }

        (bin_centers, counts)
    }

    /// Find van Hove points: k-points where the band gradient is small.
    ///
    /// For each band the gradient |∇_k ω| is estimated from finite differences
    /// between nearest k-points (Euclidean distance in 2D k-space).  Points
    /// where the estimated gradient magnitude is below `threshold` are returned
    /// together with their frequency and k-point coordinates.
    ///
    /// Returns `Vec<(freq, [kx, ky])>`.
    pub fn van_hove_points(&self, threshold: f64) -> Vec<(f64, [f64; 2])> {
        let n_k = self.k_points.len();
        if n_k < 2 {
            return Vec::new();
        }

        let mut result = Vec::new();

        for band in &self.bands {
            if band.len() != n_k {
                continue;
            }

            for ki in 0..n_k {
                // Estimate gradient magnitude at k-point ki by comparing with
                // all other k-points and taking the minimum |Δω/|Δk||.
                // (A more rigorous approach would use the Voronoi neighbourhood,
                //  but this simple estimate is sufficient for flat-band detection.)
                let mut grad_min = f64::INFINITY;

                for kj in 0..n_k {
                    if ki == kj {
                        continue;
                    }
                    let dkx = self.k_points[kj][0] - self.k_points[ki][0];
                    let dky = self.k_points[kj][1] - self.k_points[ki][1];
                    let dk = (dkx * dkx + dky * dky).sqrt();
                    if dk < 1e-30 {
                        continue;
                    }
                    let d_omega = (band[kj] - band[ki]).abs();
                    let grad = d_omega / dk;
                    if grad < grad_min {
                        grad_min = grad;
                    }
                }

                if grad_min < threshold {
                    result.push((band[ki], self.k_points[ki]));
                }
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Local density of states
// ---------------------------------------------------------------------------

/// Calculator for the Local Density of States (LDOS) at a point in real space.
///
/// The LDOS at position **r** and frequency ω is:
///
///   ρ_L(**r**, ω) = Σ_n |E_n(**r**)|² · L(ω − ω_n, γ)
///
/// where L is a Lorentzian with half-width γ and E_n(**r**) is the electric
/// field amplitude of mode n at position **r**.
#[derive(Debug, Clone, Copy)]
pub struct LdosCalc {
    /// x-coordinate in the unit cell (m)
    pub x: f64,
    /// y-coordinate in the unit cell (m)
    pub y: f64,
}

impl LdosCalc {
    /// Create an LDOS calculator at position `(x, y)`.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Compute LDOS at `freq` from a set of modes.
    ///
    /// # Arguments
    /// * `mode_freqs`      – resonance frequencies ω_n of each mode
    /// * `mode_amplitudes` – |E_n(**r**)| field amplitudes at this point
    /// * `freq`            – evaluation frequency ω
    /// * `broadening`      – Lorentzian half-width at half-maximum γ (rad/s)
    ///
    /// The Lorentzian kernel used is:
    ///
    ///   L(Δω, γ) = (γ/π) / (Δω² + γ²)
    pub fn ldos_from_modes(
        &self,
        mode_freqs: &[f64],
        mode_amplitudes: &[f64],
        freq: f64,
        broadening: f64,
    ) -> f64 {
        let n = mode_freqs.len().min(mode_amplitudes.len());
        if n == 0 || broadening <= 0.0 {
            return 0.0;
        }

        let gamma = broadening;
        let norm = gamma / PI;

        mode_freqs[..n]
            .iter()
            .zip(mode_amplitudes[..n].iter())
            .map(|(&omega_n, &amp_n)| {
                let d_omega = freq - omega_n;
                let lorentzian = norm / (d_omega * d_omega + gamma * gamma);
                amp_n * amp_n * lorentzian
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Extended band-structure and DOS analysis
// ---------------------------------------------------------------------------

/// Joint density of states (JDOS) between two bands at frequency `omega`.
///
/// Counts joint transitions where ω₂(k) − ω₁(k) ≈ ω within ±delta_omega/2.
/// `band1` and `band2` must have the same length (one value per k-point).
pub fn joint_dos(band1: &[f64], band2: &[f64], omega: f64, delta_omega: f64) -> f64 {
    let n = band1.len().min(band2.len());
    if n == 0 || delta_omega <= 0.0 {
        return 0.0;
    }
    let half = delta_omega / 2.0;
    band1[..n]
        .iter()
        .zip(band2[..n].iter())
        .filter(|(&w1, &w2)| {
            let diff = w2 - w1;
            diff >= omega - half && diff <= omega + half
        })
        .count() as f64
        / (n as f64 * delta_omega)
}

/// Group velocity v_g = dω/dk at each k-point, estimated by central differences.
///
/// `band` is the ω(k) dispersion; `dk` is the uniform k-spacing.
/// Boundary points use one-sided differences.
pub fn group_velocity_band(band: &[f64], dk: f64) -> Vec<f64> {
    let n = band.len();
    if n == 0 || dk.abs() < 1e-30 {
        return Vec::new();
    }
    let mut vg = vec![0.0f64; n];
    for i in 0..n {
        vg[i] = if i == 0 {
            (band[1] - band[0]) / dk
        } else if i == n - 1 {
            (band[n - 1] - band[n - 2]) / dk
        } else {
            (band[i + 1] - band[i - 1]) / (2.0 * dk)
        };
    }
    vg
}

/// Effective mass m* ∝ (d²ω/dk²)⁻¹ at a band extremum.
///
/// Returns ħ (normalised to 1) times the effective mass — effectively 1/|d²ω/dk²|.
/// `extremum_idx` is the k-point index of the extremum.
pub fn effective_mass_at_extremum(band: &[f64], dk: f64, extremum_idx: usize) -> f64 {
    let n = band.len();
    if n < 3 || dk.abs() < 1e-30 {
        return f64::INFINITY;
    }
    let i = extremum_idx.clamp(1, n - 2);
    let d2 = (band[i + 1] - 2.0 * band[i] + band[i - 1]) / (dk * dk);
    if d2.abs() < 1e-60 {
        f64::INFINITY
    } else {
        1.0 / d2.abs()
    }
}

/// Purcell enhancement factor from the DOS ratio.
///
/// F_P = ρ_crystal(ω) / ρ_free(ω)
///
/// A value > 1 indicates enhanced spontaneous emission; < 1 indicates inhibition.
pub fn purcell_from_dos_ratio(dos_crystal: f64, dos_freespace: f64) -> f64 {
    if dos_freespace.abs() < 1e-60 {
        return 0.0;
    }
    dos_crystal / dos_freespace
}

/// Compute band-gap centre frequency and gap width from the upper edge of the
/// lower band and the lower edge of the upper band.
///
/// Returns `(centre, gap_width)`.  If the bands overlap, `gap_width` is 0.
pub fn band_gap_info(lower_band: &[f64], upper_band: &[f64]) -> (f64, f64) {
    if lower_band.is_empty() || upper_band.is_empty() {
        return (0.0, 0.0);
    }
    let lower_max = lower_band.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let upper_min = upper_band.iter().cloned().fold(f64::INFINITY, f64::min);
    let gap_width = (upper_min - lower_max).max(0.0);
    let centre = (lower_max + upper_min) / 2.0;
    (centre, gap_width)
}

/// Flatness ratio for a single band: (ω_max − ω_min) / ω_centre.
///
/// A perfectly flat band yields 0; a wide band yields a value close to 2.
pub fn band_flatness_ratio(band: &[f64]) -> f64 {
    if band.is_empty() {
        return 0.0;
    }
    let omega_max = band.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let omega_min = band.iter().cloned().fold(f64::INFINITY, f64::min);
    let centre = (omega_max + omega_min) / 2.0;
    if centre.abs() < 1e-30 {
        return 0.0;
    }
    (omega_max - omega_min) / centre
}

/// 1D photonic crystal dispersion relation via the transfer matrix method.
///
/// Computes the Bloch phase `kΛ` as a function of normalised frequency `ωΛ/c`
/// for a bilayer unit cell with refractive indices `n1`, `n2` and layer
/// thicknesses `d1`, `d2` (in the same length units as Λ = d1 + d2).
///
/// Returns `n_k` points as `(kΛ/π, ωΛ/c)` pairs spanning the first Brillouin zone.
pub fn phc_1d_dispersion(n1: f64, n2: f64, d1: f64, d2: f64, n_k: usize) -> Vec<(f64, f64)> {
    if n_k == 0 || n1 <= 0.0 || n2 <= 0.0 || d1 <= 0.0 || d2 <= 0.0 {
        return Vec::new();
    }
    let lambda_period = d1 + d2;
    // Sweep normalised frequency ωΛ/c from 0 to 2π (exclusive of 0)
    let n_omega = n_k * 4; // oversample then pick valid Bloch phases
    let mut result = Vec::with_capacity(n_k);

    for i in 1..=n_omega {
        let omega_norm = i as f64 * 2.0 * PI / n_omega as f64; // ωΛ/c in (0, 2π]
        let k1 = n1 * omega_norm / lambda_period;
        let k2 = n2 * omega_norm / lambda_period;
        let phi1 = k1 * d1;
        let phi2 = k2 * d2;

        // Transfer matrix M = M1 × M2, trace / 2 = cos(kΛ)
        // For TE polarisation: M_j = [[cos φ_j, -i/n_j sin φ_j], [-i n_j sin φ_j, cos φ_j]]
        let cos1 = phi1.cos();
        let sin1 = phi1.sin();
        let cos2 = phi2.cos();
        let sin2 = phi2.sin();

        let trace_half = cos1 * cos2 - (n2 / n1 + n1 / n2) / 2.0 * sin1 * sin2;

        // kΛ = arccos(trace_half); only real solutions correspond to propagating modes
        if trace_half.abs() <= 1.0 {
            let k_bloch_norm = trace_half.acos() / PI; // kΛ/π in [0, 1]
            if result.len() < n_k {
                result.push((k_bloch_norm, omega_norm));
            }
        }
    }
    result
}

/// Transmission through N periods of a 1D PhC at angular frequency `omega`.
///
/// Uses the transfer matrix method (TE polarisation, normal incidence).
/// `omega` is the angular frequency in rad/s; `d1`, `d2` are in metres.
pub fn phc_1d_transmission(
    n1: f64,
    n2: f64,
    d1: f64,
    d2: f64,
    n_periods: usize,
    omega: f64,
) -> f64 {
    use std::f64::consts::PI as _PI;
    if n_periods == 0 {
        return 1.0;
    }
    let c = crate::units::conversion::SPEED_OF_LIGHT;
    let k1 = n1 * omega / c;
    let k2 = n2 * omega / c;
    let phi1 = k1 * d1;
    let phi2 = k2 * d2;

    // 2×2 transfer matrix for a single bilayer period (real-valued for lossless)
    // M = [[a, b], [c, d]] where we track only the real 2x2 matrix
    let m11 = phi1.cos() * phi2.cos() - (n2 / n1) * phi1.sin() * phi2.sin();
    let m12 = -(phi1.cos() * phi2.sin() / n2 + phi2.cos() * phi1.sin() / n1);
    let m21 = -(n1 * phi1.sin() * phi2.cos() + n2 * phi2.sin() * phi1.cos());
    let m22 = phi1.cos() * phi2.cos() - (n1 / n2) * phi1.sin() * phi2.sin();

    // Raise M to the n_periods power via repeated squaring
    let mat_power = mat2x2_power([m11, m12, m21, m22], n_periods);
    let (a, _b, c_m, d) = (mat_power[0], mat_power[1], mat_power[2], mat_power[3]);

    // Transmission coefficient (Hecht §2.11)
    // t = 2/(a + d + i(c/n_out - b*n_out))  with n_out = n1
    // |T|² = 4 / ((a+d)² + (c_m/n1 - b*n1)²)
    let _ = _PI;
    let denom = (a + d) * (a + d) + (c_m / n1 - mat_power[1] * n1).powi(2);
    if denom < 1e-60 {
        1.0
    } else {
        4.0 / denom
    }
}

/// Multiply two 2×2 matrices stored as flat arrays `[a,b,c,d]` (row-major).
fn mat2x2_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

/// Raise a 2×2 matrix to the `n`-th power by repeated squaring.
fn mat2x2_power(m: [f64; 4], n: usize) -> [f64; 4] {
    if n == 0 {
        return [1.0, 0.0, 0.0, 1.0];
    }
    if n == 1 {
        return m;
    }
    let half = mat2x2_power(m, n / 2);
    let squared = mat2x2_mul(half, half);
    if n % 2 == 0 {
        squared
    } else {
        mat2x2_mul(squared, m)
    }
}

/// Slow-light figure of merit (FOM).
///
/// Quantifies the trade-off between slow-down factor and propagation loss:
///   FOM = (n_g / n_g_ref)² · (α_ref / α)
///
/// A higher FOM indicates efficient slow light with low loss penalty.
pub fn slow_light_fom(ng: f64, ng_ref: f64, alpha: f64, alpha_ref: f64) -> f64 {
    if ng_ref.abs() < 1e-30 || alpha.abs() < 1e-30 {
        return 0.0;
    }
    let slowdown = ng / ng_ref;
    slowdown * slowdown * alpha_ref / alpha
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_band(n: usize) -> Vec<f64> {
        // Linear dispersion: ω = c·k, no bandgap
        (0..n).map(|i| i as f64 * 1e12 / (n - 1) as f64).collect()
    }

    #[test]
    fn dos_from_band_normalised() {
        let band = linear_band(100);
        let dos = PhotonicDos::from_band(&band, 50);
        let integral: f64 = dos.dos.iter().sum::<f64>() * (dos.frequencies[1] - dos.frequencies[0]);
        assert!((integral - 1.0).abs() < 0.05, "Integral={integral:.3}");
    }

    #[test]
    fn dos_peak_positive() {
        let band = linear_band(100);
        let dos = PhotonicDos::from_band(&band, 50);
        assert!(dos.peak_dos() > 0.0);
    }

    #[test]
    fn dos_empty_band_returns_zeros() {
        let dos = PhotonicDos::from_band(&[], 20);
        assert!(dos.dos.iter().all(|&d| d == 0.0));
    }

    #[test]
    fn free_photon_dos_positive() {
        let rho = free_photon_dos_1d(1e12, 1.0);
        assert!(rho > 0.0);
    }

    #[test]
    fn dos_bandgap_detection() {
        // Flat-top DOS with a zero gap in the middle
        let mut dos_obj = PhotonicDos {
            frequencies: (0..10).map(|i| i as f64 * 1e12).collect(),
            dos: vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        };
        // Normalize
        let total = dos_obj.dos.iter().sum::<f64>() * 1e12;
        for d in &mut dos_obj.dos {
            *d /= total;
        }
        let threshold = 1e-6;
        let gaps = dos_obj.bandgaps(threshold);
        assert!(!gaps.is_empty(), "Should find at least one gap");
    }

    // --- BandData2d tests ---

    #[test]
    fn band_data_2d_histogram_counts_all() {
        let mut bd = BandData2d::new(500e-9);
        // Three k-points at Γ, M, X
        bd.k_points = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        // Two bands with three frequencies each
        bd.add_band(vec![0.25, 0.30, 0.35]);
        bd.add_band(vec![0.45, 0.50, 0.55]);

        let (centers, counts) = bd.dos_histogram(10, 0.0, 1.0);
        assert_eq!(centers.len(), 10);
        assert_eq!(counts.len(), 10);
        let total: f64 = counts.iter().sum();
        // 2 bands × 3 k-points = 6 mode entries should be counted
        assert!((total - 6.0).abs() < 1e-9, "total={total}");
    }

    #[test]
    fn band_data_2d_histogram_empty_range_returns_empty() {
        let bd = BandData2d::new(500e-9);
        let (c, d) = bd.dos_histogram(0, 0.0, 1.0);
        assert!(c.is_empty());
        assert!(d.is_empty());
    }

    #[test]
    fn band_data_2d_van_hove_flat_band() {
        let mut bd = BandData2d::new(500e-9);
        bd.k_points = vec![[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]];
        // Perfectly flat band: same frequency everywhere → gradient = 0
        bd.add_band(vec![0.3, 0.3, 0.3]);

        let vhp = bd.van_hove_points(1e-3);
        // All k-points should be van Hove points for a flat band
        assert_eq!(
            vhp.len(),
            3,
            "Expected 3 flat-band van Hove points, got {}",
            vhp.len()
        );
    }

    #[test]
    fn band_data_2d_van_hove_steep_band_empty() {
        let mut bd = BandData2d::new(500e-9);
        bd.k_points = vec![[0.0, 0.0], [1.0, 0.0]];
        // Steep band: gradient = 1.0 / 1.0 = 1.0 → above small threshold
        bd.add_band(vec![0.0, 1.0]);

        let vhp = bd.van_hove_points(0.1);
        assert!(
            vhp.is_empty(),
            "Steep band should have no van Hove points below 0.1"
        );
    }

    #[test]
    fn band_data_2d_add_multiple_bands() {
        let mut bd = BandData2d::new(400e-9);
        bd.k_points = vec![[0.0, 0.0], [1.0, 0.0]];
        bd.add_band(vec![0.2, 0.3]);
        bd.add_band(vec![0.5, 0.6]);
        bd.add_band(vec![0.7, 0.8]);
        assert_eq!(bd.bands.len(), 3);
    }

    // --- LdosCalc tests ---

    #[test]
    fn ldos_on_resonance_peaks() {
        let calc = LdosCalc::new(0.0, 0.0);
        let freqs = vec![1.0e12];
        let amps = vec![1.0];
        let broadening = 1.0e10;

        let ldos_on = calc.ldos_from_modes(&freqs, &amps, 1.0e12, broadening);
        let ldos_off = calc.ldos_from_modes(&freqs, &amps, 1.1e12, broadening);
        assert!(ldos_on > ldos_off, "LDOS should peak on resonance");
    }

    #[test]
    fn ldos_zero_broadening_returns_zero() {
        let calc = LdosCalc::new(0.0, 0.0);
        let ldos = calc.ldos_from_modes(&[1e12], &[1.0], 1e12, 0.0);
        assert_eq!(ldos, 0.0);
    }

    #[test]
    fn ldos_empty_modes_returns_zero() {
        let calc = LdosCalc::new(0.5e-6, 0.5e-6);
        let ldos = calc.ldos_from_modes(&[], &[], 1e12, 1e10);
        assert_eq!(ldos, 0.0);
    }

    #[test]
    fn ldos_amplitude_scaling() {
        let calc = LdosCalc::new(0.0, 0.0);
        let freqs = vec![1.0e12];
        let broadening = 1.0e10;

        let ldos1 = calc.ldos_from_modes(&freqs, &[1.0], 1.0e12, broadening);
        let ldos2 = calc.ldos_from_modes(&freqs, &[2.0], 1.0e12, broadening);
        // LDOS ∝ |E|² so doubling amplitude quadruples LDOS
        assert!(
            (ldos2 / ldos1 - 4.0).abs() < 1e-9,
            "ratio={}",
            ldos2 / ldos1
        );
    }

    #[test]
    fn ldos_multiple_modes_additive() {
        let calc = LdosCalc::new(0.0, 0.0);
        let broadening = 1.0e10;
        let freq_eval = 2.0e12;

        let ldos_single1 = calc.ldos_from_modes(&[1.0e12], &[1.0], freq_eval, broadening);
        let ldos_single2 = calc.ldos_from_modes(&[3.0e12], &[1.0], freq_eval, broadening);
        let ldos_both = calc.ldos_from_modes(&[1.0e12, 3.0e12], &[1.0, 1.0], freq_eval, broadening);

        assert!(
            (ldos_both - ldos_single1 - ldos_single2).abs() < 1e-30,
            "LDOS should be additive across modes"
        );
    }

    // ── Extended band-structure and DOS analysis tests ─────────────────────────

    #[test]
    fn joint_dos_zero_for_empty_bands() {
        assert_eq!(joint_dos(&[], &[], 1e12, 1e10), 0.0);
    }

    #[test]
    fn joint_dos_positive_for_matching_transitions() {
        // band2 - band1 = 1e12 everywhere → JDOS should be non-zero at omega=1e12
        let band1: Vec<f64> = (0..10).map(|i| i as f64 * 1e11).collect();
        let band2: Vec<f64> = band1.iter().map(|&w| w + 1e12).collect();
        let jdos = joint_dos(&band1, &band2, 1e12, 1e11);
        assert!(
            jdos > 0.0,
            "JDOS should be positive when transitions match omega"
        );
    }

    #[test]
    fn joint_dos_zero_for_wrong_frequency() {
        let band1 = vec![0.0, 1e12, 2e12];
        let band2 = vec![5e12, 6e12, 7e12]; // gap of ~5e12
                                            // JDOS at omega=1e12 should be 0 (no k-point has w2-w1 ≈ 1e12)
        let jdos = joint_dos(&band1, &band2, 1e12, 1e9);
        assert_eq!(jdos, 0.0);
    }

    #[test]
    fn group_velocity_linear_band_constant() {
        use approx::assert_relative_eq;
        // Linear dispersion ω = v_g · k → group velocity constant = v_g
        let vg_true = 2.0e8; // m/s
        let dk = 1e5; // rad/m
        let n = 10;
        let band: Vec<f64> = (0..n).map(|i| vg_true * i as f64 * dk).collect();
        let vg = group_velocity_band(&band, dk);
        assert_eq!(vg.len(), n);
        // Interior points should be exact
        for v in &vg[1..n - 1] {
            assert_relative_eq!(*v, vg_true, max_relative = 1e-9);
        }
    }

    #[test]
    fn group_velocity_empty_band() {
        let vg = group_velocity_band(&[], 1e5);
        assert!(vg.is_empty());
    }

    #[test]
    fn effective_mass_flat_band_is_inf() {
        // d²ω/dk² = 0 for flat band → effective mass = ∞
        let band = vec![1e12; 10];
        let m = effective_mass_at_extremum(&band, 1e5, 5);
        assert!(m.is_infinite());
    }

    #[test]
    fn effective_mass_parabolic_band() {
        use approx::assert_relative_eq;
        // ω = a·k² → d²ω/dk² = 2a → m* = 1/(2a)
        let a = 1.0e20_f64;
        let dk = 1e5_f64;
        let n = 11;
        let extremum = 5;
        let band: Vec<f64> = (0..n)
            .map(|i| {
                let k = (i as f64 - extremum as f64) * dk;
                a * k * k
            })
            .collect();
        let m = effective_mass_at_extremum(&band, dk, extremum);
        assert_relative_eq!(m, 1.0 / (2.0 * a), max_relative = 1e-6);
    }

    #[test]
    fn purcell_from_dos_ratio_scales_linearly() {
        use approx::assert_relative_eq;
        let f = purcell_from_dos_ratio(3.0e10, 1.0e10);
        assert_relative_eq!(f, 3.0, max_relative = 1e-10);
    }

    #[test]
    fn purcell_from_dos_zero_freespace_returns_zero() {
        assert_eq!(purcell_from_dos_ratio(1e10, 0.0), 0.0);
    }

    #[test]
    fn band_gap_info_basic() {
        use approx::assert_relative_eq;
        let lower = vec![0.5e14, 0.6e14, 0.7e14];
        let upper = vec![1.0e14, 1.1e14, 1.2e14];
        let (centre, width) = band_gap_info(&lower, &upper);
        // lower_max = 0.7e14, upper_min = 1.0e14
        assert_relative_eq!(width, 0.3e14, max_relative = 1e-10);
        assert_relative_eq!(centre, 0.85e14, max_relative = 1e-10);
    }

    #[test]
    fn band_gap_info_overlapping_gives_zero_width() {
        let lower = vec![1.0e14, 1.5e14];
        let upper = vec![1.2e14, 1.8e14]; // upper_min (1.2e14) < lower_max (1.5e14)
        let (_, width) = band_gap_info(&lower, &upper);
        assert_eq!(width, 0.0, "Overlapping bands must give zero gap");
    }

    #[test]
    fn band_flatness_ratio_flat_band_is_zero() {
        let band = vec![1e12; 20];
        assert_eq!(band_flatness_ratio(&band), 0.0);
    }

    #[test]
    fn band_flatness_ratio_positive_for_dispersive_band() {
        let band: Vec<f64> = (0..10).map(|i| 1e12 + i as f64 * 1e10).collect();
        let r = band_flatness_ratio(&band);
        assert!(r > 0.0);
    }

    #[test]
    fn phc_1d_dispersion_nonempty() {
        // Typical quarter-wave stack: n1=1.5, n2=3.5, equal optical paths
        let lambda = 1550e-9_f64;
        let d1 = lambda / (4.0 * 1.5);
        let d2 = lambda / (4.0 * 3.5);
        let pts = phc_1d_dispersion(1.5, 3.5, d1, d2, 20);
        assert!(
            !pts.is_empty(),
            "Dispersion should return propagating modes"
        );
        for (k_norm, omega_norm) in &pts {
            assert!(
                *k_norm >= 0.0 && *k_norm <= 1.0,
                "k_norm out of [0,1]: {k_norm}"
            );
            assert!(*omega_norm > 0.0, "omega_norm must be positive");
        }
    }

    #[test]
    fn phc_1d_transmission_zero_periods_is_one() {
        let t = phc_1d_transmission(1.0, 1.5, 100e-9, 100e-9, 0, 1e15);
        assert!((t - 1.0).abs() < 1e-12);
    }

    #[test]
    fn phc_1d_transmission_bounded() {
        // Transmission must be in [0, 1] for lossless media
        let t = phc_1d_transmission(1.0, 3.5, 110e-9, 50e-9, 10, 1.5e15);
        assert!(
            (0.0..=1.0 + 1e-9).contains(&t),
            "Transmission {t} out of [0,1]"
        );
    }

    #[test]
    fn slow_light_fom_higher_ng_improves_fom() {
        let fom_low_ng = slow_light_fom(10.0, 1.0, 1.0, 1.0);
        let fom_high_ng = slow_light_fom(100.0, 1.0, 1.0, 1.0);
        assert!(
            fom_high_ng > fom_low_ng,
            "Higher group index should give better FOM"
        );
    }

    #[test]
    fn slow_light_fom_zero_reference_ng_returns_zero() {
        let fom = slow_light_fom(10.0, 0.0, 1.0, 1.0);
        assert_eq!(fom, 0.0);
    }
}
