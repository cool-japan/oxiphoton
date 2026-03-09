use std::f64::consts::PI;

/// Mach-Zehnder Interferometer (MZI) optical modulator.
///
/// Transfer function for a balanced MZI with phase shift Δφ in one arm:
///   T(Δφ) = cos²(Δφ/2)
///
/// With half-wave voltage V_π, the applied voltage V gives:
///   Δφ = π·V/V_π
///
/// Electro-optic MZI modulator (Pockels effect):
///   Δn = -n³·r·E/2  where r is the Pockels coefficient, E = V/d
#[derive(Debug, Clone, Copy)]
pub struct MziModulator {
    /// Half-wave voltage V_π (V) — voltage for π phase shift
    pub v_pi: f64,
    /// Insertion loss (linear: 0 = no loss, 1 = complete loss)
    pub insertion_loss: f64,
    /// Extinction ratio (dB): 10·log10(T_max/T_min)
    pub extinction_ratio_db: f64,
    /// DC bias phase (rad)
    pub bias_phase: f64,
}

impl MziModulator {
    /// Create an ideal MZI with given V_π.
    pub fn new(v_pi: f64) -> Self {
        Self {
            v_pi,
            insertion_loss: 0.0,
            bias_phase: 0.0,
            extinction_ratio_db: f64::INFINITY,
        }
    }

    /// Create a realistic modulator with finite ER and insertion loss.
    pub fn with_params(v_pi: f64, insertion_loss_db: f64, extinction_ratio_db: f64) -> Self {
        let insertion_loss = 1.0 - 10.0_f64.powf(-insertion_loss_db / 10.0);
        Self {
            v_pi,
            insertion_loss,
            bias_phase: 0.0,
            extinction_ratio_db,
        }
    }

    /// Phase shift for applied voltage V (V): Δφ = π·V/V_π
    pub fn phase_shift(&self, voltage: f64) -> f64 {
        PI * voltage / self.v_pi
    }

    /// Optical transmission T(V) ∈ [0, 1].
    ///
    /// For ideal MZI: T = cos²((Δφ + φ_bias)/2)
    /// With finite ER: T = T_max·cos²(...) where T_max = 1/(1+ε) for extinction ε
    pub fn transmission(&self, voltage: f64) -> f64 {
        let dphi = self.phase_shift(voltage) + self.bias_phase;
        let ideal_t = (dphi / 2.0).cos().powi(2);
        (1.0 - self.insertion_loss) * ideal_t
    }

    /// Optical power in dB relative to input: 10·log10(T)
    pub fn transmission_db(&self, voltage: f64) -> f64 {
        let t = self.transmission(voltage);
        if t < 1e-30 {
            -300.0
        } else {
            10.0 * t.log10()
        }
    }

    /// 3dB bandwidth voltage: V for which T = 0.5·T_max
    pub fn v_3db(&self) -> f64 {
        self.v_pi / 2.0
    }

    /// Quadrature point: V_Q = V_π/2 (bias for linear operation)
    pub fn v_quadrature(&self) -> f64 {
        self.v_pi / 2.0
    }

    /// Chirp-free condition: MZI at quadrature has no frequency chirp.
    /// Returns the RF small-signal modulation depth.
    pub fn small_signal_response(&self, v_rf: f64) -> f64 {
        // At quadrature: dT/dV|_VQ = -π/(2·V_π)·sin(π/2) = -π/(2·V_π)
        let slope = PI / (2.0 * self.v_pi);
        slope * v_rf
    }

    /// Compute the voltage sweep spectrum [V_min, V_max] with n_pts points.
    pub fn transfer_curve(&self, v_min: f64, v_max: f64, n_pts: usize) -> Vec<(f64, f64)> {
        (0..n_pts)
            .map(|i| {
                let v = v_min + i as f64 / (n_pts - 1) as f64 * (v_max - v_min);
                (v, self.transmission(v))
            })
            .collect()
    }
}

/// Electro-optic (Pockels) phase modulator.
///
/// Phase shift: Δφ = π·n³·r₃₃·V·L / (λ·d)
/// where r₃₃ is the Pockels coefficient (m/V), L is electrode length, d is electrode gap.
#[derive(Debug, Clone, Copy)]
pub struct PockelsModulator {
    /// Linear refractive index
    pub n: f64,
    /// Pockels coefficient r₃₃ (m/V)
    pub r33: f64,
    /// Electrode length (m)
    pub length: f64,
    /// Electrode gap (m)
    pub gap: f64,
    /// Operating wavelength (m)
    pub wavelength: f64,
}

impl PockelsModulator {
    /// LiNbO₃ modulator (standard telecom)
    pub fn linbo3(length: f64, gap: f64, wavelength: f64) -> Self {
        Self {
            n: 2.14,
            r33: 30.8e-12, // pm/V = 30.8×10⁻¹² m/V
            length,
            gap,
            wavelength,
        }
    }

    /// Half-wave voltage: V_π = λ·d / (n³·r₃₃·L)
    pub fn v_pi(&self) -> f64 {
        self.wavelength * self.gap / (self.n.powi(3) * self.r33 * self.length)
    }

    /// Phase shift for applied voltage V (V)
    pub fn phase_shift(&self, voltage: f64) -> f64 {
        PI * voltage / self.v_pi()
    }

    /// Equivalent MZI modulator built from this Pockels material.
    pub fn as_mzi(&self) -> MziModulator {
        MziModulator::new(self.v_pi())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mzi_zero_voltage_max_transmission() {
        let m = MziModulator::new(5.0);
        assert!((m.transmission(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mzi_v_pi_gives_zero_transmission() {
        let m = MziModulator::new(5.0);
        let t = m.transmission(5.0);
        assert!(t < 1e-10, "T at V_π should be 0, got {t:.2e}");
    }

    #[test]
    fn mzi_half_v_pi_is_half_transmission() {
        let m = MziModulator::new(5.0);
        let t = m.transmission(2.5); // V_π/2
        assert!(
            (t - 0.5).abs() < 1e-6,
            "T at V_π/2 should be 0.5, got {t:.4}"
        );
    }

    #[test]
    fn mzi_transfer_curve_length() {
        let m = MziModulator::new(5.0);
        let curve = m.transfer_curve(0.0, 10.0, 100);
        assert_eq!(curve.len(), 100);
    }

    #[test]
    fn mzi_transmission_in_unit_range() {
        let m = MziModulator::new(5.0);
        for v in [0.0, 1.0, 2.5, 5.0, 7.5, 10.0] {
            let t = m.transmission(v);
            assert!((0.0..=1.0).contains(&t), "T={t:.4} at V={v:.1}");
        }
    }

    #[test]
    fn mzi_v3db_is_half_v_pi() {
        let m = MziModulator::new(10.0);
        assert!((m.v_3db() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn pockels_linbo3_v_pi_physical() {
        let mod_ = PockelsModulator::linbo3(1e-2, 15e-6, 1550e-9);
        let vpi = mod_.v_pi();
        // LiNbO₃ V_π·L ≈ 3-5 V·cm → for 1cm electrode: V_π ≈ 3-8V
        assert!(
            vpi > 1.0 && vpi < 100.0,
            "V_π={vpi:.2}V out of expected range"
        );
    }

    #[test]
    fn pockels_phase_shift_proportional_to_voltage() {
        let m = PockelsModulator::linbo3(1e-2, 15e-6, 1550e-9);
        let phi1 = m.phase_shift(1.0);
        let phi2 = m.phase_shift(2.0);
        assert!((phi2 / phi1 - 2.0).abs() < 1e-10);
    }
}
