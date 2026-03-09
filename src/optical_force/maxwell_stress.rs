//! Maxwell Stress Tensor and Radiation Pressure
//!
//! Implements the Maxwell stress tensor for computing electromagnetic forces
//! on objects via integration over a closed surface.
//!
//! The Maxwell stress tensor is:
//!   T_ij = ε0*(Ei*Ej - δij*|E|²/2) + (1/μ0)*(Bi*Bj - δij*|B|²/2)
//!
//! For fields in a medium with B = μ0*H:
//!   T_ij = ε0*(Ei*Ej - δij*|E|²/2) + μ0*(Hi*Hj - δij*|H|²/2)

/// Permittivity of free space (F/m)
const EPS0: f64 = 8.854_187_817e-12;
/// Permeability of free space (H/m)
const MU0: f64 = 1.256_637_061_4e-6;
/// Speed of light in vacuum (m/s)
const C0: f64 = 2.997_924_58e8;
/// Planck constant (J·s)
const H_PLANCK: f64 = 6.626_070_15e-34;

/// Maxwell stress tensor T_ij at a single point in space.
///
/// The tensor relates the electromagnetic stress (force per unit area) across
/// a surface element. The force on a volume V is:
///   F_i = ∮_S T_ij n_j dA
///
/// where the integral is over the closed surface S enclosing V.
#[derive(Debug, Clone)]
pub struct MaxwellStressTensor {
    /// Stress tensor components T\[i\]\[j\] at a point (N/m²)
    pub t: [[f64; 3]; 3],
}

impl MaxwellStressTensor {
    /// Compute the Maxwell stress tensor from E and H field vectors at a point.
    ///
    /// Uses the vacuum formulation:
    ///   T_ij = ε0*(Ei*Ej - δij*|E|²/2) + μ0*(Hi*Hj - δij*|H|²/2)
    ///
    /// # Arguments
    /// * `e` - Electric field vector \[Ex, Ey, Ez\] (V/m)
    /// * `h` - Magnetic field vector \[Hx, Hy, Hz\] (A/m)
    ///
    /// # Returns
    /// Maxwell stress tensor at the field point
    pub fn from_fields(e: [f64; 3], h: [f64; 3]) -> Self {
        let e_sq = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        let h_sq = h[0] * h[0] + h[1] * h[1] + h[2] * h[2];

        let mut t = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let delta = if i == j { 1.0 } else { 0.0 };
                let e_term = EPS0 * (e[i] * e[j] - delta * e_sq / 2.0);
                let h_term = MU0 * (h[i] * h[j] - delta * h_sq / 2.0);
                t[i][j] = e_term + h_term;
            }
        }

        Self { t }
    }

    /// Compute the electromagnetic force on a surface element dA.
    ///
    /// The force is: dF_i = T_ij * n_j * dA
    ///
    /// # Arguments
    /// * `normal` - Unit outward normal vector \[nx, ny, nz\]
    /// * `area` - Surface element area (m²)
    ///
    /// # Returns
    /// Force vector \[Fx, Fy, Fz\] (N)
    pub fn force_on_surface(&self, normal: [f64; 3], area: f64) -> [f64; 3] {
        let mut force = [0.0f64; 3];
        for (i, f_i) in force.iter_mut().enumerate() {
            for (j, &n_j) in normal.iter().enumerate() {
                *f_i += self.t[i][j] * n_j;
            }
            *f_i *= area;
        }
        force
    }

    /// Compute the electromagnetic momentum density g = S/c² = (E×H)/c² (kg/m²/s).
    ///
    /// # Arguments
    /// * `e` - Electric field vector (V/m)
    /// * `h` - Magnetic field vector (A/m)
    ///
    /// # Returns
    /// Momentum density vector (kg/m²/s)
    pub fn momentum_density(e: [f64; 3], h: [f64; 3]) -> [f64; 3] {
        let s = Self::poynting_vector(e, h);
        let c2 = C0 * C0;
        [s[0] / c2, s[1] / c2, s[2] / c2]
    }

    /// Compute the Poynting vector S = E×H (W/m²).
    ///
    /// # Arguments
    /// * `e` - Electric field vector (V/m)
    /// * `h` - Magnetic field vector (A/m)
    ///
    /// # Returns
    /// Poynting vector (W/m²)
    pub fn poynting_vector(e: [f64; 3], h: [f64; 3]) -> [f64; 3] {
        [
            e[1] * h[2] - e[2] * h[1],
            e[2] * h[0] - e[0] * h[2],
            e[0] * h[1] - e[1] * h[0],
        ]
    }

    /// Radiation pressure on a perfect mirror: P = 2*I/c (Pa).
    ///
    /// A perfect mirror reflects all incident light, giving twice the momentum transfer
    /// compared to a perfect absorber.
    ///
    /// # Arguments
    /// * `intensity_w_per_m2` - Incident intensity (W/m²)
    ///
    /// # Returns
    /// Radiation pressure (Pa)
    pub fn radiation_pressure_mirror(intensity_w_per_m2: f64) -> f64 {
        2.0 * intensity_w_per_m2 / C0
    }

    /// Radiation pressure on a perfect absorber: P = I/c (Pa).
    ///
    /// A perfect absorber absorbs all incident photon momentum.
    ///
    /// # Arguments
    /// * `intensity_w_per_m2` - Incident intensity (W/m²)
    ///
    /// # Returns
    /// Radiation pressure (Pa)
    pub fn radiation_pressure_absorber(intensity_w_per_m2: f64) -> f64 {
        intensity_w_per_m2 / C0
    }

    /// Compute the total electromagnetic force on a closed surface by integrating
    /// the Maxwell stress tensor over all surface elements.
    ///
    /// The total force is: F_i = ∮ T_ij n_j dA
    ///
    /// # Arguments
    /// * `e_field` - Electric field at each surface point, shape \[N\]\[3\] (V/m)
    /// * `h_field` - Magnetic field at each surface point, shape \[N\]\[3\] (A/m)
    /// * `normals` - Outward unit normals at each surface point, shape \[N\]\[3\]
    /// * `areas` - Area element at each surface point (m²), length N
    ///
    /// # Returns
    /// Total force vector \[Fx, Fy, Fz\] (N)
    pub fn total_force(
        e_field: &[[f64; 3]],
        h_field: &[[f64; 3]],
        normals: &[[f64; 3]],
        areas: &[f64],
    ) -> [f64; 3] {
        let n = e_field.len();
        assert_eq!(h_field.len(), n, "h_field length must match e_field");
        assert_eq!(normals.len(), n, "normals length must match e_field");
        assert_eq!(areas.len(), n, "areas length must match e_field");

        let mut total = [0.0f64; 3];
        for idx in 0..n {
            let mst = Self::from_fields(e_field[idx], h_field[idx]);
            let df = mst.force_on_surface(normals[idx], areas[idx]);
            total[0] += df[0];
            total[1] += df[1];
            total[2] += df[2];
        }
        total
    }
}

/// Radiation pressure force analysis tools for various optical configurations.
///
/// Provides utilities for computing radiation pressure effects in practical
/// scenarios such as solar sails, optical manipulation, and photon counting.
pub struct RadiationPressure;

impl RadiationPressure {
    /// Solar radiation pressure at 1 AU from the Sun.
    ///
    /// Based on solar irradiance ≈ 1361 W/m² at 1 AU:
    ///   P_solar = I_solar / c ≈ 9.08×10⁻⁶ Pa
    ///
    /// # Returns
    /// Solar radiation pressure (Pa)
    pub fn solar_pressure_pa() -> f64 {
        // Solar irradiance at 1 AU: 1361 W/m²
        let solar_irradiance = 1361.0_f64;
        solar_irradiance / C0
    }

    /// Radiation force on a flat perfect mirror of given area.
    ///
    /// F = 2 * I * A / c
    ///
    /// # Arguments
    /// * `intensity` - Incident intensity (W/m²)
    /// * `area_m2` - Mirror area (m²)
    ///
    /// # Returns
    /// Force (N)
    pub fn mirror_force_n(intensity: f64, area_m2: f64) -> f64 {
        2.0 * intensity * area_m2 / C0
    }

    /// Radiation force on a solar sail at angle θ to the beam.
    ///
    /// For a perfectly reflecting sail tilted at angle θ:
    ///   F = 2 * I * A * cos²(θ) / c
    ///
    /// At normal incidence (θ=0), this equals the mirror force.
    ///
    /// # Arguments
    /// * `intensity` - Incident intensity (W/m²)
    /// * `area_m2` - Sail area (m²)
    /// * `angle_rad` - Angle between beam and sail normal (rad)
    ///
    /// # Returns
    /// Force magnitude along beam direction (N)
    pub fn sail_force_n(intensity: f64, area_m2: f64, angle_rad: f64) -> f64 {
        let cos_theta = angle_rad.cos();
        2.0 * intensity * area_m2 * cos_theta * cos_theta / C0
    }

    /// Single-photon momentum: p = ħk = h/λ (kg·m/s).
    ///
    /// # Arguments
    /// * `lambda_nm` - Photon wavelength (nm)
    ///
    /// # Returns
    /// Photon momentum (kg·m/s)
    pub fn photon_momentum(lambda_nm: f64) -> f64 {
        let lambda_m = lambda_nm * 1.0e-9;
        H_PLANCK / lambda_m
    }

    /// Force due to photon absorption for given optical power and wavelength.
    ///
    /// F = (P/E_photon) * p_photon = P * h/λ / (hc/λ) = P/c
    ///
    /// This is the force on a perfect absorber.
    ///
    /// # Arguments
    /// * `power_w` - Optical power (W)
    /// * `lambda_nm` - Wavelength (nm), used to verify the formula is wavelength-independent
    ///
    /// # Returns
    /// Force (N)
    pub fn photon_force_n(power_w: f64, lambda_nm: f64) -> f64 {
        // Force = (photons/s) * p_photon = (P / E_photon) * (h/λ)
        // E_photon = hc/λ, so F = P/(hc/λ) * (h/λ) = P*λ/(hc) * h/λ = P/c
        // Wavelength cancels, confirming F = P/c regardless of λ
        let _ = lambda_nm; // wavelength cancels analytically
        power_w / C0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_radiation_pressure_mirror() {
        // Perfect mirror: P = 2*I/c
        let intensity = 1000.0_f64; // 1000 W/m²
        let pressure = MaxwellStressTensor::radiation_pressure_mirror(intensity);
        let expected = 2.0 * intensity / C0;
        assert_abs_diff_eq!(pressure, expected, epsilon = 1.0e-20);
        // Numerical check: ~6.67e-6 Pa for 1000 W/m²
        assert!((pressure - 6.671e-6).abs() < 1.0e-9);
    }

    #[test]
    fn test_radiation_pressure_absorber() {
        // Perfect absorber: P = I/c
        let intensity = 1000.0_f64;
        let pressure = MaxwellStressTensor::radiation_pressure_absorber(intensity);
        let expected = intensity / C0;
        assert_abs_diff_eq!(pressure, expected, epsilon = 1.0e-20);
        // Mirror should be exactly twice absorber
        let mirror_p = MaxwellStressTensor::radiation_pressure_mirror(intensity);
        assert_abs_diff_eq!(mirror_p, 2.0 * pressure, epsilon = 1.0e-30);
    }

    #[test]
    fn test_photon_momentum() {
        // p = h/λ for λ = 500 nm (green light)
        let lambda_nm = 500.0_f64;
        let p = RadiationPressure::photon_momentum(lambda_nm);
        let lambda_m = lambda_nm * 1.0e-9;
        let expected = H_PLANCK / lambda_m;
        assert_abs_diff_eq!(p, expected, epsilon = 1.0e-40);
        // ~1.33e-27 kg·m/s for 500 nm
        assert!((p - 1.326e-27).abs() < 1.0e-30);
    }

    #[test]
    fn test_photon_force() {
        // For 100% absorption: F = P/c
        let power = 1.0_f64; // 1 W
        let lambda_nm = 532.0_f64; // 532 nm Nd:YAG doubled
        let force = RadiationPressure::photon_force_n(power, lambda_nm);
        let expected = power / C0;
        assert_abs_diff_eq!(force, expected, epsilon = 1.0e-20);
        // ~3.34e-9 N per watt
        assert!((force - 3.336e-9).abs() < 1.0e-12);
    }

    #[test]
    fn test_poynting_vector() {
        // For plane wave in +z direction: E along x, H along y
        // S = E×H should point in +z
        let e = [1.0, 0.0, 0.0_f64];
        let h = [0.0, 1.0, 0.0_f64];
        let s = MaxwellStressTensor::poynting_vector(e, h);
        // E×H = (Ey*Hz - Ez*Hy, Ez*Hx - Ex*Hz, Ex*Hy - Ey*Hx)
        //     = (0*0 - 0*1, 0*0 - 1*0, 1*1 - 0*0) = (0, 0, 1)
        assert_abs_diff_eq!(s[0], 0.0, epsilon = 1.0e-15);
        assert_abs_diff_eq!(s[1], 0.0, epsilon = 1.0e-15);
        assert_abs_diff_eq!(s[2], 1.0, epsilon = 1.0e-15);
    }

    #[test]
    fn test_mst_symmetric() {
        // The Maxwell stress tensor must be symmetric: T_ij = T_ji
        let e = [1.5e5, 2.3e5, 0.8e5_f64]; // V/m
        let h = [400.0, 150.0, 600.0_f64]; // A/m
        let mst = MaxwellStressTensor::from_fields(e, h);
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(mst.t[i][j], mst.t[j][i], epsilon = 1.0e-10);
            }
        }
    }

    #[test]
    fn test_sail_force_normal_incidence() {
        // At normal incidence (θ=0), sail force = mirror force
        let intensity = 1361.0_f64; // Solar irradiance at 1 AU
        let area = 100.0_f64; // 100 m² sail
        let sail_force = RadiationPressure::sail_force_n(intensity, area, 0.0);
        let mirror_force = RadiationPressure::mirror_force_n(intensity, area);
        assert_abs_diff_eq!(sail_force, mirror_force, epsilon = 1.0e-20);
    }

    #[test]
    fn test_total_force_single_point() {
        // Force on a single surface element
        // Plane wave propagates in +z: E along x, H along y.
        // For radiation pressure to act on an object, we integrate MST over a
        // closed surface enclosing the object. The upstream face (facing the beam,
        // at z = -z_max) has outward normal pointing in -z direction.
        // F_z = T_zz * n_z = T_zz * (-1) > 0 (pushes in +z).
        let e_amp = 1.0e4_f64; // V/m
        let h_amp = e_amp / (MU0 * C0); // A/m, from impedance Z0 = μ0*c
        let e_field = vec![[e_amp, 0.0, 0.0]];
        let h_field = vec![[0.0, h_amp, 0.0]];
        // Outward normal of the UPSTREAM face points toward the source: -z
        let normals = vec![[0.0, 0.0, -1.0]];
        let area = 1.0e-6_f64; // 1 mm² surface element

        let force = MaxwellStressTensor::total_force(&e_field, &h_field, &normals, &[area]);
        // Force should be in +z direction (radiation pressure)
        assert!(
            force[2] > 0.0,
            "Radiation pressure force must be positive in z, got {}",
            force[2]
        );
        assert!(
            force[0].abs() < 1.0e-20,
            "No lateral force for normal incidence"
        );
        assert!(
            force[1].abs() < 1.0e-20,
            "No lateral force for normal incidence"
        );

        // Verify magnitude: F = |T_zz| * area = (I/c) * area
        let intensity = e_amp * h_amp; // E×H for plane wave
        let expected_force = intensity / C0 * area;
        assert_abs_diff_eq!(force[2], expected_force, epsilon = 1.0e-15);
    }

    #[test]
    fn test_momentum_density_direction() {
        // Momentum density should be parallel to Poynting vector
        let e = [1.0e4, 0.0, 0.0_f64];
        let h = [0.0, 1.0e3 / 377.0, 0.0_f64]; // roughly impedance matched
        let s = MaxwellStressTensor::poynting_vector(e, h);
        let g = MaxwellStressTensor::momentum_density(e, h);
        // g = S/c², so direction must match
        let c2 = C0 * C0;
        assert_abs_diff_eq!(g[0], s[0] / c2, epsilon = 1.0e-20);
        assert_abs_diff_eq!(g[1], s[1] / c2, epsilon = 1.0e-20);
        assert_abs_diff_eq!(g[2], s[2] / c2, epsilon = 1.0e-20);
    }
}
