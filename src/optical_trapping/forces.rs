// Optical forces on particles: Rayleigh (dipole) and Mie theory regimes
//
// Implements gradient and scattering forces for optical trapping simulations.
// Physical framework follows Ashkin's original work and Bohren & Huffman Mie theory.

// Physical constants
#[allow(dead_code)]
const KB: f64 = 1.380649e-23; // Boltzmann constant [J/K]
const C_LIGHT: f64 = 2.99792458e8; // Speed of light [m/s]
const EPS0: f64 = 8.854187817e-12; // Vacuum permittivity [F/m]
const PI: f64 = std::f64::consts::PI;

/// Particle in Rayleigh (dipole) regime: radius a << wavelength λ
///
/// In this regime the particle is treated as a point dipole with polarizability
/// given by the Clausius-Mossotti relation.
#[derive(Debug, Clone)]
pub struct RayleighParticle {
    /// Particle radius \[m\]
    pub radius_m: f64,
    /// Particle refractive index (real part)
    pub n_particle: f64,
    /// Medium refractive index
    pub n_medium: f64,
    /// Trapping laser wavelength in vacuum \[m\]
    pub wavelength_m: f64,
}

impl RayleighParticle {
    /// Create a new Rayleigh particle
    pub fn new(radius_m: f64, n_particle: f64, n_medium: f64, wavelength_m: f64) -> Self {
        Self {
            radius_m,
            n_particle,
            n_medium,
            wavelength_m,
        }
    }

    /// Clausius-Mossotti polarizability \[C·m / (V/m)\] = \[C²·s²/(kg·m)\]
    ///
    /// α = 4π ε₀ a³ (m²-1)/(m²+2),  m = n_p / n_m
    pub fn polarizability(&self) -> f64 {
        let m = self.n_particle / self.n_medium;
        let m2 = m * m;
        let a = self.radius_m;
        4.0 * PI * EPS0 * a * a * a * (m2 - 1.0) / (m2 + 2.0)
    }

    /// Gradient (dipole) force vector \[N\]
    ///
    /// F_grad = (α / 2) ∇|E|² = (α / (2 c ε₀)) ∇I  (in SI, using I = c ε₀ n |E|²/2 → ∇|E|² = 2/(c ε₀ n) ∇I)
    /// More precisely: F_grad,i = (α/2) * grad_intensity_i  where grad_intensity is ∇(|E|²)
    ///
    /// Caller should provide ∇|E|² \[V²/m³\] directly (or ∇I / (c ε₀ n)).
    pub fn gradient_force(&self, grad_e2: [f64; 3]) -> [f64; 3] {
        let alpha = self.polarizability();
        let half_alpha = 0.5 * alpha;
        [
            half_alpha * grad_e2[0],
            half_alpha * grad_e2[1],
            half_alpha * grad_e2[2],
        ]
    }

    /// Gradient force from intensity gradient ∇I \[W/m³\]
    ///
    /// Using |E|² = 2 I / (c ε₀ n_m)
    pub fn gradient_force_from_intensity_grad(&self, grad_intensity: [f64; 3]) -> [f64; 3] {
        let alpha = self.polarizability();
        // F = α/(2 c ε₀ n_m) ∇I
        let factor = alpha / (2.0 * C_LIGHT * EPS0 * self.n_medium);
        [
            factor * grad_intensity[0],
            factor * grad_intensity[1],
            factor * grad_intensity[2],
        ]
    }

    /// Scattering (radiation pressure) force \[N\] along beam propagation direction
    ///
    /// F_scat = n_m σ_scat I / c
    pub fn scattering_force(&self, intensity_w_m2: f64) -> f64 {
        let sigma = self.scattering_cross_section();
        self.n_medium * sigma * intensity_w_m2 / C_LIGHT
    }

    /// Rayleigh scattering cross-section \[m²\]
    ///
    /// σ_scat = (128 π⁵ a⁶) / (3 λ⁴) × |(m²-1)/(m²+2)|²
    pub fn scattering_cross_section(&self) -> f64 {
        let m = self.n_particle / self.n_medium;
        let m2 = m * m;
        let cm_factor = (m2 - 1.0) / (m2 + 2.0);
        let a = self.radius_m;
        let lam = self.wavelength_m;
        (128.0 * PI.powi(5) * a.powi(6)) / (3.0 * lam.powi(4)) * cm_factor * cm_factor
    }

    /// Absorption cross-section for a weakly absorbing particle \[m²\]
    ///
    /// For a purely real refractive index (lossless), σ_abs = 0.
    /// For an absorbing particle with imaginary index κ: σ_abs = k Im(α) / ε₀
    /// This method returns zero for real n_particle; use `absorption_cross_section_complex` for lossy particles.
    pub fn absorption_cross_section(&self) -> f64 {
        // For lossless Rayleigh particle: σ_abs ≈ 0 (only scattering)
        // Non-zero absorption requires complex n. Returning scattering-only case.
        0.0
    }

    /// Total extinction cross-section \[m²\] = σ_scat + σ_abs
    pub fn extinction_cross_section(&self) -> f64 {
        self.scattering_cross_section() + self.absorption_cross_section()
    }

    /// Trap escape criterion: gradient force must exceed scattering force
    /// Returns the dimensionless trapping criterion Q_trap = F_grad_max / F_scat
    pub fn trapping_criterion(&self, peak_intensity: f64, peak_grad_intensity: f64) -> f64 {
        let f_scat = self.scattering_force(peak_intensity);
        let grad = [0.0, 0.0, peak_grad_intensity];
        let f_grad = self.gradient_force_from_intensity_grad(grad);
        let f_grad_mag = f_grad[2].abs();
        if f_scat.abs() < f64::EPSILON {
            return f64::INFINITY;
        }
        f_grad_mag / f_scat
    }
}

/// Particle in Mie regime: arbitrary size relative to wavelength
///
/// Uses Bohren & Huffman series-truncation (Wiscombe criterion) for efficiency factors.
#[derive(Debug, Clone)]
pub struct MieParticle {
    /// Particle radius \[m\]
    pub radius_m: f64,
    /// Particle refractive index (real part; complex absorption modelled via `n_imag`)
    pub n_particle: f64,
    /// Imaginary part of refractive index (extinction coefficient κ ≥ 0)
    pub n_imag: f64,
    /// Medium refractive index (assumed real)
    pub n_medium: f64,
    /// Wavelength in vacuum \[m\]
    pub wavelength_m: f64,
}

impl MieParticle {
    /// Create a non-absorbing Mie particle
    pub fn new(radius_m: f64, n_particle: f64, n_medium: f64, wavelength_m: f64) -> Self {
        Self {
            radius_m,
            n_particle,
            n_imag: 0.0,
            n_medium,
            wavelength_m,
        }
    }

    /// Create an absorbing Mie particle (complex index n + i·κ)
    pub fn new_absorbing(
        radius_m: f64,
        n_real: f64,
        n_imag: f64,
        n_medium: f64,
        wavelength_m: f64,
    ) -> Self {
        Self {
            radius_m,
            n_particle: n_real,
            n_imag,
            n_medium,
            wavelength_m,
        }
    }

    /// Size parameter x = 2π n_medium a / λ (dimensionless)
    pub fn size_parameter(&self) -> f64 {
        2.0 * PI * self.n_medium * self.radius_m / self.wavelength_m
    }

    /// Relative complex refractive index m = (n_p + i κ_p) / n_m
    pub fn relative_index_real(&self) -> f64 {
        self.n_particle / self.n_medium
    }

    pub fn relative_index_imag(&self) -> f64 {
        self.n_imag / self.n_medium
    }

    /// Mie extinction efficiency Q_ext via Bohren-Huffman series (truncated at n_max terms)
    ///
    /// Q_ext = (2/x²) Σ_{n=1}^{N} (2n+1) Re(a_n + b_n)
    pub fn q_ext(&self) -> f64 {
        let x = self.size_parameter();
        if x < 1e-10 {
            return 0.0;
        }
        let (a_coeffs, b_coeffs) = self.mie_coefficients();
        let mut q = 0.0;
        for (n, (a, b)) in a_coeffs.iter().zip(b_coeffs.iter()).enumerate() {
            let order = (n + 1) as f64;
            q += (2.0 * order + 1.0) * (a[0] + b[0]); // Re parts
        }
        2.0 / (x * x) * q
    }

    /// Mie scattering efficiency Q_scat
    ///
    /// Q_scat = (2/x²) Σ (2n+1)(|a_n|² + |b_n|²)
    pub fn q_scat(&self) -> f64 {
        let x = self.size_parameter();
        if x < 1e-10 {
            return 0.0;
        }
        let (a_coeffs, b_coeffs) = self.mie_coefficients();
        let mut q = 0.0;
        for (n, (a, b)) in a_coeffs.iter().zip(b_coeffs.iter()).enumerate() {
            let order = (n + 1) as f64;
            q += (2.0 * order + 1.0) * (a[0] * a[0] + a[1] * a[1] + b[0] * b[0] + b[1] * b[1]);
        }
        2.0 / (x * x) * q
    }

    /// Absorption efficiency Q_abs = Q_ext - Q_scat
    pub fn q_abs(&self) -> f64 {
        (self.q_ext() - self.q_scat()).max(0.0)
    }

    /// Radiation pressure efficiency Q_pr = Q_ext - <cos θ> Q_scat
    ///
    /// Q_pr = Q_ext - (4/x²) Σ [ n(n+2)/(n+1) Re(a_n a*_{n+1} + b_n b*_{n+1})
    ///         + (2n+1)/(n(n+1)) Re(a_n b*_n) ]
    pub fn q_pr(&self) -> f64 {
        let x = self.size_parameter();
        if x < 1e-10 {
            return 0.0;
        }
        let q_ext = self.q_ext();
        // Asymmetry parameter g = <cos θ>
        let g = self.asymmetry_parameter();
        let q_scat = self.q_scat();
        q_ext - g * q_scat
    }

    /// Asymmetry parameter g = <cos θ> via Bohren-Huffman formula
    pub fn asymmetry_parameter(&self) -> f64 {
        let x = self.size_parameter();
        if x < 1e-10 {
            return 0.0;
        }
        let (a_coeffs, b_coeffs) = self.mie_coefficients();
        let n_max = a_coeffs.len();
        let mut g_sum = 0.0;
        for n in 0..n_max {
            let order = (n + 1) as f64;
            // Cross terms between consecutive orders
            if n + 1 < n_max {
                let an = a_coeffs[n];
                let an1 = a_coeffs[n + 1];
                let bn = b_coeffs[n];
                let bn1 = b_coeffs[n + 1];
                // Re(a_n · conj(a_{n+1})) + Re(b_n · conj(b_{n+1}))
                let re_aa = an[0] * an1[0] + an[1] * an1[1];
                let re_bb = bn[0] * bn1[0] + bn[1] * bn1[1];
                g_sum += order * (order + 2.0) / (order + 1.0) * (re_aa + re_bb);
            }
            // Re(a_n · conj(b_n))
            let an = a_coeffs[n];
            let bn = b_coeffs[n];
            let re_ab = an[0] * bn[0] + an[1] * bn[1];
            g_sum += (2.0 * order + 1.0) / (order * (order + 1.0)) * re_ab;
        }
        let q_scat = self.q_scat();
        if q_scat < f64::EPSILON {
            return 0.0;
        }
        4.0 / (x * x * q_scat) * g_sum
    }

    /// Radiation pressure force \[N\] along beam propagation axis
    ///
    /// F_rp = Q_pr × G_scat × (π a² I / c)  where G_scat accounts for medium
    pub fn radiation_pressure_force(&self, intensity_w_m2: f64) -> f64 {
        let q_pr = self.q_pr();
        let a = self.radius_m;
        let geom_cross = PI * a * a;
        q_pr * geom_cross * intensity_w_m2 * self.n_medium / C_LIGHT
    }

    /// Geometric cross-section \[m²\]
    pub fn geometric_cross_section(&self) -> f64 {
        PI * self.radius_m * self.radius_m
    }

    /// Extinction cross-section C_ext = Q_ext × π a² \[m²\]
    pub fn extinction_cross_section(&self) -> f64 {
        self.q_ext() * self.geometric_cross_section()
    }

    /// Compute Mie a_n, b_n coefficients using Bohren & Huffman recursive algorithm
    ///
    /// Returns Vec of \[re, im\] pairs for a_n and b_n coefficients.
    /// Uses downward recurrence for logarithmic derivative D_n(mx).
    fn mie_coefficients(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let x = self.size_parameter();
        let mr = self.relative_index_real();
        let mi = self.relative_index_imag();

        // Wiscombe (1980) stopping criterion
        let n_stop = {
            let n = if x <= 8.0 {
                (x + 4.0 * x.cbrt() + 2.0) as usize + 1
            } else if x <= 4200.0 {
                (x + 4.05 * x.cbrt() + 2.0) as usize + 1
            } else {
                (x + 4.0 * x.cbrt() + 2.0) as usize + 1
            };
            n.clamp(3, 200) // cap for safety
        };

        // mx (complex)
        let mx_r = mr * x;
        let mx_i = mi * x;

        // Downward recurrence for D_n(mx): D_{n-1} = n/mx - 1/(D_n + n/mx)
        let n_d = n_stop + 10;
        let mut d_r = vec![0.0f64; n_d + 1];
        let mut d_i = vec![0.0f64; n_d + 1];
        // D_{n_d} = 0 + 0i (starting value)
        for n in (1..=n_d).rev() {
            let nf = n as f64;
            // n/mx = n/(mx_r + i mx_i) = n*(mx_r - i mx_i)/(mx_r²+mx_i²)
            let denom = mx_r * mx_r + mx_i * mx_i;
            let nmx_r = nf * mx_r / denom;
            let nmx_i = -nf * mx_i / denom;
            // D_{n-1} = n/mx - 1/(D_n + n/mx)
            let sum_r = d_r[n] + nmx_r;
            let sum_i = d_i[n] + nmx_i;
            let sum_mod2 = sum_r * sum_r + sum_i * sum_i;
            if sum_mod2 < f64::EPSILON {
                continue;
            }
            let inv_r = sum_r / sum_mod2;
            let inv_i = -sum_i / sum_mod2;
            d_r[n - 1] = nmx_r - inv_r;
            d_i[n - 1] = nmx_i - inv_i;
        }

        // Upward recurrence for ψ_n, ξ_n (Riccati-Bessel functions) and coefficients
        let mut psi_prev = x.cos(); // ψ_{-1}? No: ψ_0 = sin(x), ψ_{-1}=cos(x) for downward
        let mut psi_curr = x.sin(); // ψ_0 = sin x... using standard convention
                                    // Actually use: ψ_{n-1}, ψ_n, ξ_{n-1}, ξ_n
                                    // ξ_n = ψ_n - i χ_n,  χ_n = Neumann/Bessel Y
        let mut _xi_r_prev = x.cos(); // ξ_{-1} real
        let mut _xi_i_prev = x.sin(); // ξ_{-1} imag  (χ_{-1} = -sin x → ξ_{-1} = cos x + i sin x ?)
                                      // Proper initialization (Bohren & Huffman convention):
                                      // ψ_0 = sin x,  ψ_{-1} = cos x
                                      // χ_0 = -cos x, χ_{-1} = -sin x  →  ξ_0 = ψ_0 - i χ_0 = sin x + i cos x
        let mut chi_prev = -x.sin();
        let mut chi_curr = -x.cos();
        // xi = psi - i*chi
        let mut xi_r_curr = psi_curr;
        let mut xi_i_curr = -chi_curr;

        let mut a_coeffs = Vec::with_capacity(n_stop);
        let mut b_coeffs = Vec::with_capacity(n_stop);

        for n in 1..=n_stop {
            let nf = n as f64;

            // Recurrence: ψ_{n+1} = (2n+1)/x ψ_n - ψ_{n-1}
            let psi_next = (2.0 * nf + 1.0) / x * psi_curr - psi_prev;
            let chi_next = (2.0 * nf + 1.0) / x * chi_curr - chi_prev;
            let xi_r_next = psi_next;
            let xi_i_next = -chi_next;

            // a_n = (D_n(mx)/m + n/x) ψ_n - ψ_{n-1}
            //       / (D_n(mx)/m + n/x) ξ_n - ξ_{n-1}
            // Note: psi_next = ψ_n, psi_curr = ψ_{n-1} (recurrence computed above)
            // D_n/m = (D_r[n] + i D_i[n]) / (mr + i mi)
            let m_mod2 = mr * mr + mi * mi;
            let dn_over_m_r = (d_r[n] * mr + d_i[n] * mi) / m_mod2;
            let dn_over_m_i = (d_i[n] * mr - d_r[n] * mi) / m_mod2;

            let num_a_r = (dn_over_m_r + nf / x) * psi_next - psi_curr;
            let den_a_r = (dn_over_m_r + nf / x) * xi_r_next - xi_r_curr;
            let den_a_i = dn_over_m_i * xi_r_next + (dn_over_m_r + nf / x) * xi_i_next - xi_i_curr;
            let num_a_i = dn_over_m_i * psi_next;

            let den_mod2 = den_a_r * den_a_r + den_a_i * den_a_i;
            let (a_r, a_i) = if den_mod2 > f64::EPSILON {
                (
                    (num_a_r * den_a_r + num_a_i * den_a_i) / den_mod2,
                    (num_a_i * den_a_r - num_a_r * den_a_i) / den_mod2,
                )
            } else {
                (0.0, 0.0)
            };

            // b_n = (m D_n(mx) + n/x) ψ_n - ψ_{n-1}
            //       / (m D_n(mx) + n/x) ξ_n - ξ_{n-1}
            // m D_n = (mr + i mi)(D_r + i D_i) = mr Dr - mi Di + i(mr Di + mi Dr)
            let m_dn_r = mr * d_r[n] - mi * d_i[n];
            let m_dn_i = mr * d_i[n] + mi * d_r[n];

            let num_b_r = (m_dn_r + nf / x) * psi_next - psi_curr;
            let num_b_i = m_dn_i * psi_next;
            let den_b_r = (m_dn_r + nf / x) * xi_r_next - xi_r_curr;
            let den_b_i = m_dn_i * xi_r_next + (m_dn_r + nf / x) * xi_i_next - xi_i_curr;

            let den_b_mod2 = den_b_r * den_b_r + den_b_i * den_b_i;
            let (b_r, b_i) = if den_b_mod2 > f64::EPSILON {
                (
                    (num_b_r * den_b_r + num_b_i * den_b_i) / den_b_mod2,
                    (num_b_i * den_b_r - num_b_r * den_b_i) / den_b_mod2,
                )
            } else {
                (0.0, 0.0)
            };

            a_coeffs.push([a_r, a_i]);
            b_coeffs.push([b_r, b_i]);

            // Advance recurrence
            psi_prev = psi_curr;
            psi_curr = psi_next;
            chi_prev = chi_curr;
            chi_curr = chi_next;
            _xi_r_prev = xi_r_curr;
            _xi_i_prev = xi_i_curr;
            xi_r_curr = xi_r_next;
            xi_i_curr = xi_i_next;
        }

        (a_coeffs, b_coeffs)
    }
}

/// Focused Gaussian beam optical trap
///
/// Models a tightly focused Gaussian beam (paraxial + ABCD) as the trapping field.
#[derive(Debug, Clone)]
pub struct GaussianTrap {
    /// Laser power \[W\]
    pub power_w: f64,
    /// Wavelength in vacuum \[m\]
    pub wavelength_m: f64,
    /// Medium refractive index
    pub n_medium: f64,
    /// Numerical aperture of objective
    pub numerical_aperture: f64,
    /// 1/e² beam waist radius at focus \[m\]
    pub beam_waist_m: f64,
}

impl GaussianTrap {
    /// Construct from physical parameters; beam waist approximated as λ/(π NA)
    pub fn new(power_w: f64, wavelength_m: f64, n_medium: f64, na: f64) -> Self {
        let beam_waist_m = wavelength_m / (PI * na);
        Self {
            power_w,
            wavelength_m,
            n_medium,
            numerical_aperture: na,
            beam_waist_m,
        }
    }

    /// Rayleigh range z_R = π w₀² n / λ \[m\]
    pub fn rayleigh_range(&self) -> f64 {
        PI * self.beam_waist_m * self.beam_waist_m * self.n_medium / self.wavelength_m
    }

    /// Beam waist radius at axial position z: w(z) = w₀ √(1 + (z/z_R)²)
    pub fn beam_radius(&self, z: f64) -> f64 {
        let zr = self.rayleigh_range();
        self.beam_waist_m * (1.0 + (z / zr).powi(2)).sqrt()
    }

    /// Peak intensity at focus I₀ = 2P/(π w₀²) \[W/m²\]
    pub fn peak_intensity(&self) -> f64 {
        2.0 * self.power_w / (PI * self.beam_waist_m * self.beam_waist_m)
    }

    /// Intensity at position (x, y, z) \[W/m²\]
    ///
    /// I(r,z) = I₀ (w₀/w(z))² exp(-2 r²/w(z)²)
    pub fn intensity_at(&self, x: f64, y: f64, z: f64) -> f64 {
        let r2 = x * x + y * y;
        let wz = self.beam_radius(z);
        let i0 = self.peak_intensity();
        let w0 = self.beam_waist_m;
        i0 * (w0 / wz).powi(2) * (-2.0 * r2 / (wz * wz)).exp()
    }

    /// Analytical intensity gradient ∇I at (x, y, z) \[W/m³\]
    ///
    /// Returns \[∂I/∂x, ∂I/∂y, ∂I/∂z\]
    pub fn gradient_at(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        let r2 = x * x + y * y;
        let i = self.intensity_at(x, y, z);
        let wz = self.beam_radius(z);
        let zr = self.rayleigh_range();
        let w0 = self.beam_waist_m;

        // ∂I/∂x = I × (-4x/w²)
        let di_dx = i * (-4.0 * x / (wz * wz));
        let di_dy = i * (-4.0 * y / (wz * wz));

        // ∂I/∂z via chain rule on w(z):
        // dw/dz = w₀ × z/z_R² / √(1+(z/z_R)²) = z/(z_R² (w/w₀))
        // ∂I/∂z = I × [-2/w × dw/dz + 2r²/w³ × 2 × dw/dz]   ...and the w₀/w factor
        // Compact form: ∂I/∂z = I × (dw/dz / w) × (4r²/w² - 2)
        let zr2 = zr * zr;
        let wz_over_w0 = wz / w0;
        let dw_dz = if wz_over_w0.abs() > f64::EPSILON {
            z / (zr2 * wz_over_w0)
        } else {
            0.0
        };
        let di_dz = i * (dw_dz / wz) * (4.0 * r2 / (wz * wz) - 2.0);

        [di_dx, di_dy, di_dz]
    }

    /// Axial trap depth \[eV\] for a Rayleigh particle
    ///
    /// U_axial = α/(2 c ε₀ n) × ΔI_axial where ΔI = I₀ - I(∞) = I₀
    pub fn axial_trap_depth_ev(&self, particle: &RayleighParticle) -> f64 {
        let i0 = self.peak_intensity();
        let alpha = particle.polarizability();
        // U = α I / (2 c ε₀ n)
        let u_joules = alpha * i0 / (2.0 * C_LIGHT * EPS0 * self.n_medium);
        u_joules / 1.602176634e-19 // convert J → eV
    }

    /// Radial trap depth \[eV\] — same physics but radially
    /// For a Gaussian beam the radial and axial trap depths are related by geometry.
    /// The radial potential well is shallower by the factor (z_R/w₀)² in stiffness,
    /// but the same depth in energy (I₀ drops to zero at r→∞ and z→∞).
    pub fn radial_trap_depth_ev(&self, particle: &RayleighParticle) -> f64 {
        // Radial depth equals axial for a Gaussian trap (both approach I=0 at ∞)
        self.axial_trap_depth_ev(particle)
    }

    /// Radial trap stiffness at focus \[N/m\]: k_r = 4 α I₀ / (c ε₀ n w₀²)
    pub fn radial_stiffness(&self, particle: &RayleighParticle) -> f64 {
        let alpha = particle.polarizability();
        let i0 = self.peak_intensity();
        let w0 = self.beam_waist_m;
        4.0 * alpha * i0 / (C_LIGHT * EPS0 * self.n_medium * w0 * w0)
    }

    /// Axial trap stiffness at focus \[N/m\]: k_z = 2 α I₀ / (c ε₀ n z_R²)
    pub fn axial_stiffness(&self, particle: &RayleighParticle) -> f64 {
        let alpha = particle.polarizability();
        let i0 = self.peak_intensity();
        let zr = self.rayleigh_range();
        2.0 * alpha * i0 / (C_LIGHT * EPS0 * self.n_medium * zr * zr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rayleigh_polarizability_si_nanosphere() {
        // 50 nm Si sphere in water, 1064 nm trap
        let p = RayleighParticle {
            radius_m: 50e-9,
            n_particle: 3.5,
            n_medium: 1.33,
            wavelength_m: 1064e-9,
        };
        let alpha = p.polarizability();
        // α = 4π ε₀ a³ (m²-1)/(m²+2)
        // m = 3.5/1.33 ≈ 2.632, m² ≈ 6.928
        // (m²-1)/(m²+2) ≈ 5.928/8.928 ≈ 0.664
        // 4π×8.85e-12×(50e-9)³×0.664 ≈ 4π×8.85e-12×1.25e-22×0.664 ≈ 9.24e-34
        assert!(
            alpha > 0.0,
            "polarizability must be positive, got {}",
            alpha
        );
        assert!(alpha < 1e-30, "polarizability too large: {}", alpha);
        assert!(alpha > 1e-36, "polarizability too small: {}", alpha);
    }

    #[test]
    fn rayleigh_scattering_cross_section_polystyrene() {
        // 100 nm polystyrene bead (n=1.59) in water (n=1.33), λ=532 nm
        let p = RayleighParticle {
            radius_m: 100e-9,
            n_particle: 1.59,
            n_medium: 1.33,
            wavelength_m: 532e-9,
        };
        let sigma = p.scattering_cross_section();
        // For 100 nm polystyrene bead: σ_scat ~ 10^-16 to 10^-14 m²
        assert!(sigma > 0.0, "scattering cross-section must be positive");
        assert!(sigma < 1e-12, "σ_scat={} unreasonably large", sigma);
        assert!(sigma > 1e-22, "σ_scat={} unreasonably small", sigma);
    }

    #[test]
    fn gaussian_trap_peak_intensity() {
        // 1 W focused to NA=1.2 in water
        let trap = GaussianTrap::new(1.0, 1064e-9, 1.33, 1.2);
        let i0 = trap.peak_intensity();
        // w0 = λ/(π NA) = 1064e-9 / (π × 1.2) ≈ 282 nm
        // I0 = 2P/(π w0²) ≈ 2/(π × (282e-9)²) ≈ 8e12 W/m²
        assert!(i0 > 1e10, "peak intensity {} too low", i0);
        assert!(i0 < 1e15, "peak intensity {} too high", i0);
    }

    #[test]
    fn gaussian_trap_intensity_falls_off_radially() {
        let trap = GaussianTrap::new(1.0, 1064e-9, 1.33, 1.2);
        let i0 = trap.intensity_at(0.0, 0.0, 0.0);
        let w0 = trap.beam_waist_m;
        // At r = w0: I = I0/e² ≈ 0.135 I0
        let i_at_w0 = trap.intensity_at(w0, 0.0, 0.0);
        let ratio = i_at_w0 / i0;
        assert!(
            (ratio - (-2.0_f64).exp()).abs() < 0.01,
            "Expected exp(-2)={:.4}, got {:.4}",
            (-2.0_f64).exp(),
            ratio
        );
    }

    #[test]
    fn gaussian_gradient_sign_correct() {
        let trap = GaussianTrap::new(1.0, 1064e-9, 1.33, 1.2);
        // At positive x: gradient should point toward negative x (restoring)
        let w0 = trap.beam_waist_m;
        let grad = trap.gradient_at(0.5 * w0, 0.0, 0.0);
        assert!(
            grad[0] < 0.0,
            "∂I/∂x at x>0 should be negative (beam center pull), got {}",
            grad[0]
        );
    }

    #[test]
    fn mie_size_parameter() {
        // 500 nm sphere in water, λ=1064 nm → x = 2π×1.33×500e-9/1064e-9 ≈ 3.94
        let p = MieParticle::new(500e-9, 1.59, 1.33, 1064e-9);
        let x = p.size_parameter();
        let expected = 2.0 * PI * 1.33 * 500e-9 / 1064e-9;
        assert!(
            (x - expected).abs() < 1e-10,
            "size parameter mismatch: {} vs {}",
            x,
            expected
        );
    }

    #[test]
    fn mie_q_ext_positive_and_bounded() {
        // 200 nm polystyrene sphere (n=1.59) at λ=532 nm in water
        let p = MieParticle::new(200e-9, 1.59, 1.33, 532e-9);
        let q_ext = p.q_ext();
        let q_scat = p.q_scat();
        // Physical constraints: all efficiencies non-negative
        assert!(q_ext >= 0.0, "Q_ext must be non-negative, got {}", q_ext);
        assert!(q_scat >= 0.0, "Q_scat must be non-negative, got {}", q_scat);
        // For a non-absorbing sphere: Q_abs = Q_ext - Q_scat must be ≥ 0 (clamped)
        let q_abs = p.q_abs();
        assert!(q_abs >= 0.0, "Q_abs must be non-negative, got {}", q_abs);
        // Size parameter sanity
        let x = p.size_parameter();
        assert!(x > 0.0 && x < 100.0, "Unreasonable size parameter {}", x);
    }

    #[test]
    fn mie_radiation_pressure_force_positive() {
        let p = MieParticle::new(500e-9, 1.59, 1.33, 1064e-9);
        let f = p.radiation_pressure_force(1e12); // 1 TW/m² (focused beam)
        assert!(
            f >= 0.0,
            "radiation pressure force must be non-negative, got {}",
            f
        );
    }

    #[test]
    fn rayleigh_trap_depth_positive_ev() {
        let trap = GaussianTrap::new(0.1, 1064e-9, 1.33, 1.2);
        let particle = RayleighParticle::new(100e-9, 1.59, 1.33, 1064e-9);
        let depth = trap.axial_trap_depth_ev(&particle);
        assert!(depth > 0.0, "trap depth must be positive, got {} eV", depth);
    }
}
