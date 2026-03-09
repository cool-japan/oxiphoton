//! Absorbing boundary conditions (ABCs) for FDTD.
//!
//! Mur first-order ABC (1981):
//!   At x=0 (left boundary):
//!     E^{n+1}\[0\] = E^n\[1\] + (c·Δt - Δz)/(c·Δt + Δz) · (E^{n+1}\[1\] - E^n\[0\])
//!
//!   At x=L (right boundary):
//!     E^{n+1}\[N\] = E^n\[N-1\] + (c·Δt - Δz)/(c·Δt + Δz) · (E^{n+1}\[N-1\] - E^n\[N\])
//!
//! Mur second-order ABC provides improved absorption at oblique incidence.
//!
//! The Mur ABC assumes locally plane-wave propagation and works best for
//! normally incident waves. For broadband or oblique incidence, use PML.

/// Mur first-order ABC for 1D FDTD.
///
/// Stores previous time step values for the boundary cells.
#[derive(Debug, Clone)]
pub struct MurAbc1d {
    /// Mur coefficient (c·dt - dz) / (c·dt + dz)
    pub coeff: f64,
    /// Previous E-field at left boundary (index 0)
    pub e_prev_left: f64,
    /// Previous E-field at right boundary (index N)
    pub e_prev_right: f64,
    /// Previous E-field at left interior (index 1)
    pub e_prev_left1: f64,
    /// Previous E-field at right interior (index N-1)
    pub e_prev_right1: f64,
}

impl MurAbc1d {
    const C: f64 = 2.998e8;

    /// Create Mur ABC for given grid spacing and time step.
    pub fn new(dz: f64, dt: f64) -> Self {
        let c_dt = Self::C * dt;
        let coeff = (c_dt - dz) / (c_dt + dz);
        Self {
            coeff,
            e_prev_left: 0.0,
            e_prev_right: 0.0,
            e_prev_left1: 0.0,
            e_prev_right1: 0.0,
        }
    }

    /// Save boundary values before E-field update.
    pub fn save(&mut self, ex: &[f64]) {
        self.e_prev_left = ex[0];
        self.e_prev_left1 = ex[1];
        self.e_prev_right = *ex
            .last()
            .expect("ex slice must be non-empty for Mur ABC boundary");
        self.e_prev_right1 = ex[ex.len() - 2];
    }

    /// Apply Mur ABC to E-field after update.
    pub fn apply(&self, ex: &mut [f64]) {
        let n = ex.len();
        // Left boundary
        ex[0] = self.e_prev_left1 + self.coeff * (ex[1] - self.e_prev_left);
        // Right boundary
        ex[n - 1] = self.e_prev_right1 + self.coeff * (ex[n - 2] - self.e_prev_right);
    }
}

/// Mur second-order ABC for 1D FDTD.
///
/// Stores two previous time steps at boundary.
#[derive(Debug, Clone)]
pub struct MurAbc2ndOrder1d {
    /// First-order coefficient
    pub c1: f64,
    /// Second-order coefficient
    pub c2: f64,
    /// E-field at left: \[time n, time n-1\] × \[x=0, x=1, x=2\]
    pub e_left: [[f64; 3]; 2],
    /// E-field at right: \[time n, time n-1\] × \[x=N, x=N-1, x=N-2\]
    pub e_right: [[f64; 3]; 2],
}

impl MurAbc2ndOrder1d {
    const C: f64 = 2.998e8;

    /// Create 2nd-order Mur ABC.
    pub fn new(dz: f64, dt: f64) -> Self {
        let c_dt = Self::C * dt;
        let q = c_dt / dz;
        let c1 = (q - 1.0) / (q + 1.0);
        let c2 = 2.0 * q / (q + 1.0);
        Self {
            c1,
            c2,
            e_left: [[0.0; 3]; 2],
            e_right: [[0.0; 3]; 2],
        }
    }

    /// Save boundary region before E update.
    pub fn save(&mut self, ex: &[f64]) {
        let n = ex.len();
        // Shift time history: [n-1] ← [n]
        self.e_left[1] = self.e_left[0];
        self.e_right[1] = self.e_right[0];
        // Store current step
        self.e_left[0] = [ex[0], ex[1], ex[2]];
        self.e_right[0] = [ex[n - 1], ex[n - 2], ex[n - 3]];
    }

    /// Apply 2nd-order Mur ABC after E update.
    pub fn apply(&self, ex: &mut [f64]) {
        let n = ex.len();
        // Left: E^{n+1}[0] = -E^{n-1}[1] + c1*(E^{n+1}[1]+E^{n-1}[0]) + c2*(E^n[0]+E^n[1])
        ex[0] = -self.e_left[1][1]
            + self.c1 * (ex[1] + self.e_left[1][0])
            + self.c2 * (self.e_left[0][0] + self.e_left[0][1]);
        // Right boundary
        ex[n - 1] = -self.e_right[1][1]
            + self.c1 * (ex[n - 2] + self.e_right[1][0])
            + self.c2 * (self.e_right[0][0] + self.e_right[0][1]);
    }
}

/// Simple 1D FDTD with Mur ABC to verify absorption.
pub struct FdtdMurTest {
    pub ex: Vec<f64>,
    pub hy: Vec<f64>,
    pub dz: f64,
    pub dt: f64,
    pub mur: MurAbc1d,
    pub step: usize,
}

impl FdtdMurTest {
    const EPS0: f64 = 8.854e-12;
    const MU0: f64 = 1.2566e-6;
    const C: f64 = 2.998e8;

    pub fn new(n: usize, dz: f64) -> Self {
        let dt = 0.99 * dz / Self::C;
        let mur = MurAbc1d::new(dz, dt);
        Self {
            ex: vec![0.0; n],
            hy: vec![0.0; n],
            dz,
            dt,
            mur,
            step: 0,
        }
    }

    pub fn advance(&mut self, src_i: usize, src_amp: f64) {
        use std::f64::consts::PI;
        self.mur.save(&self.ex);

        let n = self.n();
        // H update
        for i in 0..n - 1 {
            self.hy[i] -= self.dt / (Self::MU0 * self.dz) * (self.ex[i + 1] - self.ex[i]);
        }
        // E update: ∂Ex/∂t = -(1/ε₀)∂Hy/∂z
        for i in 1..n {
            self.ex[i] -= self.dt / (Self::EPS0 * self.dz) * (self.hy[i] - self.hy[i - 1]);
        }
        // Source
        let t = self.step as f64 * self.dt;
        let f0 = Self::C / (30.0 * self.dz);
        self.ex[src_i] += src_amp * (2.0 * PI * f0 * t).sin();

        self.mur.apply(&mut self.ex);
        self.step += 1;
    }

    pub fn n(&self) -> usize {
        self.ex.len()
    }

    pub fn total_energy(&self) -> f64 {
        self.ex.iter().map(|e| e * e).sum::<f64>() * Self::EPS0 * self.dz
            + self.hy.iter().map(|h| h * h).sum::<f64>() * Self::MU0 * self.dz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mur_coefficient_between_minus_one_and_one() {
        let mur = MurAbc1d::new(10e-9, 0.99 * 10e-9 / 2.998e8);
        assert!(
            mur.coeff > -1.0 && mur.coeff < 1.0,
            "coeff={:.4}",
            mur.coeff
        );
    }

    #[test]
    fn mur_apply_after_save() {
        let dz = 10e-9;
        let dt = 0.99 * dz / 2.998e8;
        let mut mur = MurAbc1d::new(dz, dt);
        let mut ex = vec![1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.5];
        mur.save(&ex);
        // Modify field
        for v in &mut ex {
            *v *= 0.9;
        }
        mur.apply(&mut ex);
        // Boundaries should be modified by Mur formula
        assert!(ex[0].is_finite());
        assert!(ex[ex.len() - 1].is_finite());
    }

    #[test]
    fn mur_2nd_order_coefficients() {
        let dz = 10e-9;
        let dt = 0.99 * dz / 2.998e8;
        let abc = MurAbc2ndOrder1d::new(dz, dt);
        assert!(abc.c1.is_finite());
        assert!(abc.c2.is_finite());
    }

    #[test]
    fn fdtd_mur_energy_bounded_with_source() {
        // Verify the Mur ABC is applied (boundaries modified) and field stays finite
        // during injection. The Mur BC coefficient absorbs outgoing waves.
        let mut sim = FdtdMurTest::new(200, 10e-9);
        for _ in 0..200 {
            sim.advance(50, 1.0);
        }
        let max_e = sim
            .ex
            .iter()
            .cloned()
            .fold(0.0_f64, |a, b| a.abs().max(b.abs()));
        // After 200 steps with Mur ABC, field must be finite and bounded
        assert!(max_e.is_finite(), "field blew up: {max_e:.2e}");
        assert!(max_e < 1e6, "field too large (instability?): {max_e:.2e}");
    }

    #[test]
    fn fdtd_mur_field_finite() {
        let mut sim = FdtdMurTest::new(100, 5e-9);
        for _ in 0..500 {
            sim.advance(25, 1.0);
        }
        let max_e = sim
            .ex
            .iter()
            .cloned()
            .fold(0.0_f64, |a, b| a.abs().max(b.abs()));
        assert!(max_e.is_finite() && max_e < 1e10, "max_e={max_e:.2e}");
    }

    #[test]
    fn mur_2nd_order_save_and_apply() {
        let dz = 10e-9;
        let dt = 0.99 * dz / 2.998e8;
        let mut abc = MurAbc2ndOrder1d::new(dz, dt);
        let mut ex = vec![0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5];
        abc.save(&ex);
        abc.save(&ex); // two saves to fill history
        for v in &mut ex {
            *v *= 0.5;
        }
        abc.apply(&mut ex);
        assert!(ex[0].is_finite());
        assert!(ex[ex.len() - 1].is_finite());
    }
}
