use crate::material::DispersiveMaterial;
use crate::units::{RefractiveIndex, Wavelength};

/// Cauchy dispersion model: n(lambda) = A + B/lambda^2 + C/lambda^4
///
/// Lambda in micrometers for coefficient convention.
#[derive(Debug, Clone)]
pub struct Cauchy {
    pub name: String,
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Cauchy {
    pub fn new(name: impl Into<String>, a: f64, b: f64, c: f64) -> Self {
        Self {
            name: name.into(),
            a,
            b,
            c,
        }
    }

    /// BK7 glass (Schott) — approximate Cauchy coefficients
    pub fn bk7() -> Self {
        Self::new("BK7", 1.5046, 0.004_20, 0.0)
    }
}

impl DispersiveMaterial for Cauchy {
    fn refractive_index(&self, wavelength: Wavelength) -> RefractiveIndex {
        let lambda_um = wavelength.as_um();
        let l2 = lambda_um * lambda_um;
        let l4 = l2 * l2;
        let n = self.a + self.b / l2 + self.c / l4;
        RefractiveIndex { n, k: 0.0 }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bk7_at_550nm() {
        let bk7 = Cauchy::bk7();
        let ri = bk7.refractive_index(Wavelength::from_nm(550.0));
        // BK7 n ~ 1.518 at 550nm
        assert_relative_eq!(ri.n, 1.518, epsilon = 0.01);
    }

    #[test]
    fn cauchy_dispersion_increases_with_shorter_wavelength() {
        let mat = Cauchy::new("test", 1.5, 0.005, 0.0);
        let n_blue = mat.refractive_index(Wavelength::from_nm(400.0)).n;
        let n_red = mat.refractive_index(Wavelength::from_nm(700.0)).n;
        assert!(n_blue > n_red, "Normal dispersion: n(blue) > n(red)");
    }
}
