/// Transfer-matrix method for cascaded PIC elements.
///
/// Provides waveguide sections, grating couplers, Y-junctions,
/// and a cascade container for full circuit analysis.
use super::circuit_elements::{Complex, TransferMatrix2x2};

// ---------------------------------------------------------------------------
// Waveguide Section
// ---------------------------------------------------------------------------

/// Single-mode waveguide propagation section.
///
/// Models phase accumulation and propagation loss for a straight waveguide
/// segment used as an arm in MZIs, ring resonators, or delay lines.
#[derive(Clone, Debug)]
pub struct WaveguideSection {
    /// Physical length of the waveguide segment (m).
    pub length_m: f64,
    /// Effective refractive index n_eff.
    pub n_eff: f64,
    /// Propagation loss in dB/m.
    pub loss_db_per_m: f64,
    /// Operating wavelength (m).
    pub wavelength_m: f64,
}

impl WaveguideSection {
    /// Construct a waveguide section with explicit parameters.
    pub fn new(length_m: f64, n_eff: f64, loss_db_per_m: f64, wavelength_m: f64) -> Self {
        Self {
            length_m,
            n_eff,
            loss_db_per_m,
            wavelength_m,
        }
    }

    /// Accumulated phase φ = 2π n_eff L / λ (rad).
    pub fn phase_rad(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.n_eff * self.length_m / self.wavelength_m
    }

    /// Amplitude transmission a = 10^(−loss_dB / 20).
    pub fn amplitude_transmission(&self) -> f64 {
        let loss_db = self.loss_db_per_m * self.length_m;
        10.0_f64.powf(-loss_db / 20.0)
    }

    /// Complex field transmission factor a·e^{iφ} (single-port amplitude).
    pub fn transfer_matrix_single(&self) -> Complex {
        Complex::from_polar(self.amplitude_transmission(), self.phase_rad())
    }

    /// 2×2 diagonal transfer matrix for a two-arm context.
    ///
    /// The waveguide occupies one arm; the other is unaffected (identity).
    /// Upper arm: a·e^{iφ}, lower arm: 1.
    pub fn transfer_matrix_upper_arm(&self) -> TransferMatrix2x2 {
        let field = self.transfer_matrix_single();
        TransferMatrix2x2 {
            m: [
                [field, Complex::new(0.0, 0.0)],
                [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
            ],
        }
    }

    /// 2×2 diagonal transfer matrix with this section in both arms (balanced).
    pub fn transfer_matrix_balanced(&self) -> TransferMatrix2x2 {
        let field = self.transfer_matrix_single();
        TransferMatrix2x2 {
            m: [
                [field, Complex::new(0.0, 0.0)],
                [Complex::new(0.0, 0.0), field],
            ],
        }
    }

    /// Power transmission over this section (linear, not dB).
    pub fn power_transmission(&self) -> f64 {
        let a = self.amplitude_transmission();
        a * a
    }

    /// Insertion loss for this section in dB.
    pub fn insertion_loss_db(&self) -> f64 {
        self.loss_db_per_m * self.length_m
    }

    /// Group delay τ = n_g L / c (s) given the group index.
    pub fn group_delay_s(&self, n_group: f64) -> f64 {
        n_group * self.length_m / crate::pic_simulation::noise_model::C_LIGHT
    }
}

// ---------------------------------------------------------------------------
// Grating Coupler
// ---------------------------------------------------------------------------

/// Surface grating coupler — couples between on-chip waveguide and free space.
///
/// The spectral response is modelled as a Gaussian centred at the design
/// wavelength with a 1/e² half-width equal to `bandwidth_nm` (which is taken
/// as the 3 dB bandwidth here for practical convenience).
#[derive(Clone, Debug)]
pub struct GratingCoupler {
    /// Peak coupling efficiency η ∈ (0, 1).
    pub coupling_efficiency: f64,
    /// Centre wavelength (m).
    pub center_wavelength_m: f64,
    /// 3 dB bandwidth (nm).
    pub bandwidth_nm: f64,
    /// Additional insertion loss on top of coupling efficiency (dB).
    pub insertion_loss_db: f64,
}

impl GratingCoupler {
    /// Create a standard silicon grating coupler.
    pub fn standard_si() -> Self {
        Self {
            coupling_efficiency: 0.5,
            center_wavelength_m: 1550e-9,
            bandwidth_nm: 40.0,
            insertion_loss_db: 3.0,
        }
    }

    /// Transmission as a function of wavelength (Gaussian spectral shape).
    ///
    /// T(λ) = η × 10^(-IL/10) × exp(−(λ − λ0)² / (2 σ²))
    /// where σ = BW / (2√(2 ln 2)) for a FWHM = BW Gaussian.
    pub fn transmission(&self, wavelength_m: f64) -> f64 {
        let lambda_nm = wavelength_m * 1e9;
        let center_nm = self.center_wavelength_m * 1e9;
        let sigma = self.bandwidth_nm / (2.0 * (2.0 * 2.0_f64.ln()).sqrt());
        let gauss = (-((lambda_nm - center_nm).powi(2)) / (2.0 * sigma * sigma)).exp();
        let il_linear = 10.0_f64.powf(-self.insertion_loss_db / 10.0);
        self.coupling_efficiency * il_linear * gauss
    }

    /// 1 dB bandwidth (nm): BW_1dB ≈ BW_3dB / 1.23.
    pub fn bandwidth_1db_nm(&self) -> f64 {
        self.bandwidth_nm / 1.23
    }

    /// Back-reflection into the waveguide (approximately −20 dB for standard designs).
    pub fn back_reflection_db(&self) -> f64 {
        -20.0
    }

    /// Peak insertion loss including coupling efficiency.
    pub fn peak_insertion_loss_db(&self) -> f64 {
        -10.0 * self.coupling_efficiency.log10() + self.insertion_loss_db
    }
}

// ---------------------------------------------------------------------------
// Y-Junction
// ---------------------------------------------------------------------------

/// Y-junction (1×2 power splitter / combiner).
///
/// Adiabatic taper-based splitter; modelled as a lossless or low-loss
/// power divider with controllable split ratio and imbalance.
#[derive(Clone, Debug)]
pub struct YJunction {
    /// Fraction of input power directed to port 1 (0.5 for balanced).
    pub split_ratio: f64,
    /// Total excess loss (dB).
    pub excess_loss_db: f64,
    /// Power imbalance between outputs |P1 − P2| (dB).
    pub imbalance_db: f64,
}

impl YJunction {
    /// Balanced 50/50 Y-junction with typical silicon-photonics performance.
    pub fn balanced() -> Self {
        Self {
            split_ratio: 0.5,
            excess_loss_db: 0.2,
            imbalance_db: 0.1,
        }
    }

    /// Ideal (lossless, perfectly balanced) Y-junction.
    pub fn ideal() -> Self {
        Self {
            split_ratio: 0.5,
            excess_loss_db: 0.0,
            imbalance_db: 0.0,
        }
    }

    /// Power delivered to output port 1.
    pub fn power_to_port1(&self, input_power: f64) -> f64 {
        let il = 10.0_f64.powf(-self.excess_loss_db / 10.0);
        input_power * il * self.split_ratio
    }

    /// Power delivered to output port 2.
    pub fn power_to_port2(&self, input_power: f64) -> f64 {
        let il = 10.0_f64.powf(-self.excess_loss_db / 10.0);
        input_power * il * (1.0 - self.split_ratio)
    }

    /// Power imbalance (linear ratio T_max / T_min).
    pub fn imbalance_linear(&self) -> f64 {
        10.0_f64.powf(self.imbalance_db / 10.0)
    }
}

// ---------------------------------------------------------------------------
// PIC Cascade
// ---------------------------------------------------------------------------

/// Ordered cascade of 2×2 photonic elements for circuit-level simulation.
///
/// Elements are applied in insertion order (left-to-right / input-to-output).
/// The total transfer matrix is M_N × … × M_2 × M_1.
#[derive(Clone, Debug)]
pub struct PicCascade {
    /// Human-readable labels for each element.
    pub elements: Vec<String>,
    /// Transfer matrices in application order.
    pub matrices: Vec<TransferMatrix2x2>,
}

impl Default for PicCascade {
    fn default() -> Self {
        Self::new()
    }
}

impl PicCascade {
    /// Create an empty cascade.
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            matrices: Vec::new(),
        }
    }

    /// Append an element to the cascade.
    pub fn add_element(&mut self, label: &str, m: TransferMatrix2x2) {
        self.elements.push(label.to_string());
        self.matrices.push(m);
    }

    /// Compute the total transfer matrix of the cascade.
    ///
    /// Returns the identity matrix for an empty cascade.
    pub fn total_matrix(&self) -> TransferMatrix2x2 {
        self.matrices
            .iter()
            .fold(TransferMatrix2x2::identity(), |acc, m| m.cascade(&acc))
    }

    /// Through port power transmission (input port 1, output port 1).
    ///
    /// The `wavelength_m` parameter is accepted for API consistency with
    /// wavelength-sweep use cases; the current implementation uses the
    /// pre-computed matrices and is wavelength-independent at this layer.
    pub fn through_transmission(&self, _wavelength_m: f64) -> f64 {
        let m = self.total_matrix();
        let (b1, _) = m.apply(Complex::new(1.0, 0.0), Complex::new(0.0, 0.0));
        b1.abs2()
    }

    /// Cross port power transmission (input port 1, output port 2).
    pub fn cross_transmission(&self, _wavelength_m: f64) -> f64 {
        let m = self.total_matrix();
        let (_, b2) = m.apply(Complex::new(1.0, 0.0), Complex::new(0.0, 0.0));
        b2.abs2()
    }

    /// Insertion loss in dB for the through port.
    pub fn insertion_loss_db(&self, wavelength_m: f64) -> f64 {
        let t = self.through_transmission(wavelength_m);
        if t <= 0.0 {
            return f64::INFINITY;
        }
        -10.0 * t.log10()
    }

    /// Number of elements in the cascade.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// True if the cascade contains no elements.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Print a simple summary of the cascade to a String.
    pub fn summary(&self) -> String {
        let mut s = format!("PicCascade ({} elements):\n", self.elements.len());
        for (i, label) in self.elements.iter().enumerate() {
            s.push_str(&format!("  [{:02}] {}\n", i, label));
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::circuit_elements::DirectionalCoupler;
    use super::*;

    #[test]
    fn waveguide_phase_positive() {
        let wg = WaveguideSection::new(100e-6, 2.4, 3000.0, 1550e-9);
        assert!(wg.phase_rad() > 0.0);
    }

    #[test]
    fn waveguide_lossless_amplitude_is_one() {
        let wg = WaveguideSection::new(100e-6, 2.4, 0.0, 1550e-9);
        assert!((wg.amplitude_transmission() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn waveguide_loss_monotone_with_length() {
        let wg1 = WaveguideSection::new(100e-6, 2.4, 3000.0, 1550e-9);
        let wg2 = WaveguideSection::new(200e-6, 2.4, 3000.0, 1550e-9);
        assert!(wg1.amplitude_transmission() > wg2.amplitude_transmission());
    }

    #[test]
    fn grating_coupler_peak_transmission() {
        let gc = GratingCoupler::standard_si();
        let t_peak = gc.transmission(gc.center_wavelength_m);
        // Peak should be η × 10^(-IL/10)
        let expected = gc.coupling_efficiency * 10.0_f64.powf(-gc.insertion_loss_db / 10.0);
        assert!((t_peak - expected).abs() < 1e-10, "t_peak={}", t_peak);
    }

    #[test]
    fn grating_coupler_off_wavelength_lower() {
        let gc = GratingCoupler::standard_si();
        let t_peak = gc.transmission(gc.center_wavelength_m);
        let t_off = gc.transmission(gc.center_wavelength_m + 30e-9);
        assert!(t_off < t_peak, "off-centre should be lower");
    }

    #[test]
    fn y_junction_power_conservation() {
        let yj = YJunction::ideal();
        let p1 = yj.power_to_port1(1.0);
        let p2 = yj.power_to_port2(1.0);
        assert!((p1 + p2 - 1.0).abs() < 1e-10, "P1+P2={}", p1 + p2);
    }

    #[test]
    fn pic_cascade_empty_identity() {
        let cascade = PicCascade::new();
        let t = cascade.through_transmission(1550e-9);
        assert!((t - 1.0).abs() < 1e-12, "empty cascade T={}", t);
    }

    #[test]
    fn pic_cascade_two_couplers_power_conserved() {
        let dc1 = DirectionalCoupler::new(0.5).transfer_matrix();
        let dc2 = DirectionalCoupler::new(0.5).transfer_matrix();
        let mut cascade = PicCascade::new();
        cascade.add_element("dc1", dc1);
        cascade.add_element("dc2", dc2);
        let m = cascade.total_matrix();
        let (b1, b2) = m.apply(Complex::new(1.0, 0.0), Complex::new(0.0, 0.0));
        let total = b1.abs2() + b2.abs2();
        assert!((total - 1.0).abs() < 1e-10, "Power={}", total);
    }
}
