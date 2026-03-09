use std::f64::consts::PI;

/// Multi-mode interference (MMI) coupler.
///
/// Based on the self-imaging principle in multimode waveguides.
/// The MMI length for N-fold imaging: L_π = n_r W_eff² / λ
///
/// For a 1×2 splitter, the imaging length is 3L_π/4.
/// For a 2×2 splitter, the imaging length is L_π/2.
///
/// Reference: Soldano & Pennings, J. Lightwave Technol. 13(4), 1995.
#[derive(Debug, Clone)]
pub struct MmiCoupler {
    /// MMI waveguide width W (m).
    pub width: f64,
    /// Effective refractive index of the MMI section.
    pub n_eff: f64,
    /// Free-space wavelength (m).
    pub wavelength: f64,
    /// Effective width (accounts for lateral penetration into cladding).
    /// W_eff ≈ W + λ/(2π) * (n_clad/n_core) for TE modes.
    pub width_eff: f64,
}

impl MmiCoupler {
    /// Create an MMI coupler.
    pub fn new(width: f64, n_eff: f64, wavelength: f64, n_clad: f64) -> Self {
        // Effective width correction (penetration depth into cladding for TE)
        // Using paraxial approximation: W_eff ≈ W + 2 * λ / (π * sqrt(n_eff² - n_clad²))
        let width_eff = if n_eff > n_clad {
            let delta = 2.0 * wavelength / (PI * (n_eff * n_eff - n_clad * n_clad).sqrt());
            width + delta
        } else {
            width
        };
        Self {
            width,
            n_eff,
            wavelength,
            width_eff,
        }
    }

    /// Fundamental beat length L_π = π n_r W_eff² / λ.
    ///
    /// This is the length at which the fundamental and first-order modes
    /// accumulate a π phase difference.
    pub fn beat_length(&self) -> f64 {
        PI * self.n_eff * self.width_eff * self.width_eff / self.wavelength
    }

    /// Length for 1×2 splitting (3L_π/4).
    pub fn length_1x2(&self) -> f64 {
        3.0 * self.beat_length() / 4.0
    }

    /// Length for 2×2 (3dB) coupling (L_π/2).
    pub fn length_2x2(&self) -> f64 {
        self.beat_length() / 2.0
    }

    /// Length for N×N splitting using general self-imaging.
    pub fn length_nxn(&self, n: usize) -> f64 {
        self.beat_length() / (n as f64)
    }

    /// Power in each output port of a 1×N splitter (ideal: 1/N each).
    pub fn output_power_1xn(&self, n: usize) -> f64 {
        1.0 / n as f64
    }

    /// Crossing length for 2×2 coupler where all power crosses.
    pub fn crossing_length(&self) -> f64 {
        self.beat_length()
    }
}

/// MMI-based 1×2 power splitter descriptor.
#[derive(Debug, Clone)]
pub struct Mmi1x2 {
    pub coupler: MmiCoupler,
    /// Gap between the two output waveguides (m).
    pub output_gap: f64,
}

impl Mmi1x2 {
    pub fn new(width: f64, n_eff: f64, wavelength: f64, n_clad: f64, output_gap: f64) -> Self {
        Self {
            coupler: MmiCoupler::new(width, n_eff, wavelength, n_clad),
            output_gap,
        }
    }

    pub fn length(&self) -> f64 {
        self.coupler.length_1x2()
    }

    /// Power in each output port (nominally 0.5 each).
    pub fn output_power(&self) -> f64 {
        0.5
    }

    /// Position of output waveguide centers relative to MMI center.
    pub fn output_positions(&self) -> (f64, f64) {
        let half_gap = self.output_gap / 2.0;
        (-half_gap, half_gap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmi_beat_length() {
        // Si MMI at 1550nm, W = 6μm
        let mmi = MmiCoupler::new(6e-6, 2.8, 1550e-9, 1.444);
        let lpi = mmi.beat_length();
        // L_π = π * n_r * W_eff² / λ ≈ π * 2.8 * (6e-6)² / 1550e-9
        // ≈ 205 μm (rough estimate before W_eff correction)
        assert!(
            lpi > 10e-6 && lpi < 500e-6,
            "Beat length={:.2}μm",
            lpi * 1e6
        );
    }

    #[test]
    fn mmi_1x2_length_positive() {
        let mmi = MmiCoupler::new(4e-6, 2.8, 1550e-9, 1.444);
        let l = mmi.length_1x2();
        assert!(l > 0.0, "1x2 MMI length must be positive");
    }

    #[test]
    fn mmi_power_split() {
        let splitter = Mmi1x2::new(4e-6, 2.8, 1550e-9, 1.444, 2e-6);
        // Ideal 50/50 split
        assert!((splitter.output_power() - 0.5).abs() < 1e-10);
    }
}
