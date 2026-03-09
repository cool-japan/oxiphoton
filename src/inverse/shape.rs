//! Shape optimization using level-set and boundary parametric methods.
//!
//! Level-set method represents the design boundary as the zero contour of
//! a scalar function φ(x):
//!   φ(x) > 0 → inside material
//!   φ(x) = 0 → boundary
//!   φ(x) < 0 → outside (void)
//!
//! Shape gradient (sensitivity to boundary motion):
//!   dJ/dΩ = -∫_∂Ω (dJ/dε) · δε · v_n dS
//!
//! where v_n is the normal boundary velocity (Hamilton-Jacobi evolution).
//!
//! The level-set function evolves via:
//!   ∂φ/∂t + v_n · |∇φ| = 0
//!
//! Regularisation: periodic reinitialization to keep |∇φ| ≈ 1 (signed distance).

/// Level-set field for shape optimization.
///
/// The level-set φ(x,y) is defined on a 2D grid. Positive values indicate
/// material, negative values indicate void.
#[derive(Debug, Clone)]
pub struct LevelSet {
    /// Grid size in x
    pub nx: usize,
    /// Grid size in y
    pub ny: usize,
    /// Grid spacing (m)
    pub dx: f64,
    /// Level-set values φ(i,j)
    pub phi: Vec<f64>,
}

impl LevelSet {
    /// Create a level-set initialized to a solid slab.
    pub fn solid(nx: usize, ny: usize, dx: f64) -> Self {
        Self {
            nx,
            ny,
            dx,
            phi: vec![1.0; nx * ny],
        }
    }

    /// Create a level-set initialized to void.
    pub fn void(nx: usize, ny: usize, dx: f64) -> Self {
        Self {
            nx,
            ny,
            dx,
            phi: vec![-1.0; nx * ny],
        }
    }

    /// Create a level-set from a circular inclusion.
    ///
    /// φ(x,y) = r - √((x-cx)² + (y-cy)²)  (positive inside circle)
    pub fn circle(nx: usize, ny: usize, dx: f64, cx_m: f64, cy_m: f64, radius_m: f64) -> Self {
        let mut phi = vec![0.0f64; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 * dx;
                let y = j as f64 * dx;
                let dist = ((x - cx_m).powi(2) + (y - cy_m).powi(2)).sqrt();
                phi[j * nx + i] = radius_m - dist;
            }
        }
        Self { nx, ny, dx, phi }
    }

    /// Get value at grid point (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.phi[j * self.nx + i]
    }

    /// Set value at grid point (i, j).
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.phi[j * self.nx + i] = val;
    }

    /// Material indicator: 1.0 if φ > 0 (material), 0.0 if φ < 0 (void).
    pub fn material_indicator(&self) -> Vec<f64> {
        self.phi
            .iter()
            .map(|&p| if p > 0.0 { 1.0 } else { 0.0 })
            .collect()
    }

    /// Volume fraction (fraction of grid points with φ > 0).
    pub fn volume_fraction(&self) -> f64 {
        let n_mat = self.phi.iter().filter(|&&p| p > 0.0).count();
        n_mat as f64 / self.phi.len() as f64
    }

    /// Advance level-set one step via upwind Hamilton-Jacobi scheme.
    ///
    /// v_n: normal velocity at each grid point (positive = expand).
    /// dt: time step (should satisfy CFL: dt < dx / max|v_n|).
    pub fn advance(&mut self, v_n: &[f64], dt: f64) {
        assert_eq!(v_n.len(), self.phi.len());
        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let old_phi = self.phi.clone();
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                let vn = v_n[idx];
                // Upwind differences for |∇φ|
                // x-direction
                let phi_xp = if i + 1 < nx {
                    old_phi[j * nx + i + 1]
                } else {
                    old_phi[idx]
                };
                let phi_xm = if i > 0 {
                    old_phi[j * nx + i - 1]
                } else {
                    old_phi[idx]
                };
                // y-direction
                let phi_yp = if j + 1 < ny {
                    old_phi[(j + 1) * nx + i]
                } else {
                    old_phi[idx]
                };
                let phi_ym = if j > 0 {
                    old_phi[(j - 1) * nx + i]
                } else {
                    old_phi[idx]
                };

                let dpx = (phi_xp - old_phi[idx]) / dx;
                let dmx = (old_phi[idx] - phi_xm) / dx;
                let dpy = (phi_yp - old_phi[idx]) / dx;
                let dmy = (old_phi[idx] - phi_ym) / dx;

                // Godunov upwind for HJ equation
                let grad_sq = if vn > 0.0 {
                    dmx.max(0.0).powi(2)
                        + dpx.min(0.0).powi(2)
                        + dmy.max(0.0).powi(2)
                        + dpy.min(0.0).powi(2)
                } else {
                    dpx.max(0.0).powi(2)
                        + dmx.min(0.0).powi(2)
                        + dpy.max(0.0).powi(2)
                        + dmy.min(0.0).powi(2)
                };
                self.phi[idx] = old_phi[idx] - dt * vn * grad_sq.sqrt();
            }
        }
    }

    /// Reinitialize φ to a signed distance function via fast marching (simplified).
    ///
    /// Uses Sussman reinitialization (PDE-based): ∂φ/∂τ + sign(φ₀)·(|∇φ| - 1) = 0.
    pub fn reinitialize(&mut self, n_steps: usize) {
        let phi0 = self.phi.clone();
        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let dt = 0.5 * dx;
        for _ in 0..n_steps {
            let old = self.phi.clone();
            for j in 0..ny {
                for i in 0..nx {
                    let idx = j * nx + i;
                    let s = phi0[idx].signum();
                    let phi_xp = if i + 1 < nx {
                        old[j * nx + i + 1]
                    } else {
                        old[idx]
                    };
                    let phi_xm = if i > 0 { old[j * nx + i - 1] } else { old[idx] };
                    let phi_yp = if j + 1 < ny {
                        old[(j + 1) * nx + i]
                    } else {
                        old[idx]
                    };
                    let phi_ym = if j > 0 {
                        old[(j - 1) * nx + i]
                    } else {
                        old[idx]
                    };
                    let dpx = (phi_xp - old[idx]) / dx;
                    let dmx = (old[idx] - phi_xm) / dx;
                    let dpy = (phi_yp - old[idx]) / dx;
                    let dmy = (old[idx] - phi_ym) / dx;
                    let g = if s > 0.0 {
                        (dmx.max(0.0).powi(2)
                            + dpx.min(0.0).powi(2)
                            + dmy.max(0.0).powi(2)
                            + dpy.min(0.0).powi(2))
                        .sqrt()
                    } else {
                        (dpx.max(0.0).powi(2)
                            + dmx.min(0.0).powi(2)
                            + dpy.max(0.0).powi(2)
                            + dmy.min(0.0).powi(2))
                        .sqrt()
                    };
                    self.phi[idx] = old[idx] - dt * s * (g - 1.0);
                }
            }
        }
    }

    /// Perimeter length (m): number of sign-change interfaces × dx.
    pub fn perimeter(&self) -> f64 {
        let nx = self.nx;
        let ny = self.ny;
        let mut count = 0usize;
        for j in 0..ny {
            for i in 0..nx {
                let v = self.phi[j * nx + i];
                if i + 1 < nx && v * self.phi[j * nx + i + 1] < 0.0 {
                    count += 1;
                }
                if j + 1 < ny && v * self.phi[(j + 1) * nx + i] < 0.0 {
                    count += 1;
                }
            }
        }
        count as f64 * self.dx
    }
}

// ---------------------------------------------------------------------------
// LevelSetField: signed-distance-aware level-set with density conversion
// ---------------------------------------------------------------------------

/// Level-set field that maintains a signed distance function φ on a 2-D grid.
///
/// Convention: φ > 0 = material, φ < 0 = void, φ = 0 = interface.
/// The field aims to satisfy |∇φ| = 1 after reinitialization.
#[derive(Debug, Clone)]
pub struct LevelSetField {
    /// Grid size in x
    pub nx: usize,
    /// Grid size in y
    pub ny: usize,
    /// Uniform grid spacing (m)
    pub dx: f64,
    /// Level-set values φ(i + j*nx)
    pub phi: Vec<f64>,
}

impl LevelSetField {
    /// Create a new zero-initialized level-set field.
    pub fn new(nx: usize, ny: usize, dx: f64) -> Self {
        Self {
            nx,
            ny,
            dx,
            phi: vec![0.0; nx * ny],
        }
    }

    /// Build a level-set from a density field by thresholding at 0.5.
    ///
    /// Grid points with ρ > 0.5 get φ = +dx, points with ρ ≤ 0.5 get φ = −dx.
    /// Call `reinitialize` afterwards to convert to a proper signed-distance field.
    pub fn from_density(rho: &[f64], nx: usize, ny: usize, dx: f64) -> Self {
        assert_eq!(rho.len(), nx * ny);
        let phi = rho
            .iter()
            .map(|&r| if r > 0.5 { dx } else { -dx })
            .collect();
        Self { nx, ny, dx, phi }
    }

    // ------------------------------------------------------------------
    // Inline helper: upwind finite differences at index (i, j)
    // ------------------------------------------------------------------
    #[inline]
    fn upwind_diffs(
        phi: &[f64],
        nx: usize,
        ny: usize,
        dx: f64,
        i: usize,
        j: usize,
    ) -> (f64, f64, f64, f64) {
        let idx = j * nx + i;
        let phi_xp = if i + 1 < nx {
            phi[j * nx + i + 1]
        } else {
            phi[idx]
        };
        let phi_xm = if i > 0 { phi[j * nx + i - 1] } else { phi[idx] };
        let phi_yp = if j + 1 < ny {
            phi[(j + 1) * nx + i]
        } else {
            phi[idx]
        };
        let phi_ym = if j > 0 {
            phi[(j - 1) * nx + i]
        } else {
            phi[idx]
        };

        let dpx = (phi_xp - phi[idx]) / dx; // forward x
        let dmx = (phi[idx] - phi_xm) / dx; // backward x
        let dpy = (phi_yp - phi[idx]) / dx; // forward y
        let dmy = (phi[idx] - phi_ym) / dx; // backward y
        (dpx, dmx, dpy, dmy)
    }

    /// Reinitialize φ to a signed distance function using pseudo-time Godunov upwind.
    ///
    /// Iterates the PDE: ∂φ/∂τ + sign(φ₀)·(|∇φ|_Godunov − 1) = 0
    /// for `n_iter` pseudo-time steps of size Δτ = 0.5·dx.
    pub fn reinitialize(&mut self, n_iter: usize) {
        let phi0 = self.phi.clone();
        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let dt = 0.5 * dx;

        for _ in 0..n_iter {
            let old = self.phi.clone();
            for j in 0..ny {
                for i in 0..nx {
                    let idx = j * nx + i;
                    let s = phi0[idx].signum();
                    if s == 0.0 {
                        continue;
                    }
                    let (dpx, dmx, dpy, dmy) = Self::upwind_diffs(&old, nx, ny, dx, i, j);

                    // Godunov: choose upwind based on sign(s)
                    let grad_sq = if s > 0.0 {
                        dmx.max(0.0).powi(2)
                            + dpx.min(0.0).powi(2)
                            + dmy.max(0.0).powi(2)
                            + dpy.min(0.0).powi(2)
                    } else {
                        dpx.max(0.0).powi(2)
                            + dmx.min(0.0).powi(2)
                            + dpy.max(0.0).powi(2)
                            + dmy.min(0.0).powi(2)
                    };
                    self.phi[idx] = old[idx] - dt * s * (grad_sq.sqrt() - 1.0);
                }
            }
        }
    }

    /// Convert φ to a smooth density field using a regularised Heaviside.
    ///
    /// H_ε(φ) = { 0                              if φ < −ε
    ///           { 0.5·(1 + φ/ε + sin(π·φ/ε)/π)  if |φ| ≤ ε
    ///           { 1                              if φ >  ε
    ///
    /// where ε = 1.5 · dx.
    pub fn to_density(&self) -> Vec<f64> {
        use std::f64::consts::PI;
        let eps = 1.5 * self.dx;
        self.phi
            .iter()
            .map(|&p| {
                if p < -eps {
                    0.0
                } else if p > eps {
                    1.0
                } else {
                    0.5 * (1.0 + p / eps + (PI * p / eps).sin() / PI)
                }
            })
            .collect()
    }

    /// Advance φ by one Hamilton-Jacobi upwind step.
    ///
    /// Solves ∂φ/∂t + v_n · |∇φ| = 0 with Godunov upwind for |∇φ|.
    /// `vel` is the normal velocity (positive = material expands).
    /// `dt` should satisfy CFL: dt ≤ dx / max|vel|.
    pub fn evolve(&mut self, vel: &[f64], dt: f64) {
        assert_eq!(vel.len(), self.phi.len(), "vel length must equal nx*ny");
        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let old = self.phi.clone();

        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                let vn = vel[idx];
                let (dpx, dmx, dpy, dmy) = Self::upwind_diffs(&old, nx, ny, dx, i, j);

                // Godunov: for φ_t + vn |∇φ| = 0
                //   vn > 0 → material grows → upwind from inside: use D⁻ for +, D⁺ for −
                let grad_sq = if vn > 0.0 {
                    dmx.max(0.0).powi(2)
                        + dpx.min(0.0).powi(2)
                        + dmy.max(0.0).powi(2)
                        + dpy.min(0.0).powi(2)
                } else {
                    dpx.max(0.0).powi(2)
                        + dmx.min(0.0).powi(2)
                        + dpy.max(0.0).powi(2)
                        + dmy.min(0.0).powi(2)
                };
                self.phi[idx] = old[idx] - dt * vn * grad_sq.sqrt();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Velocity extension by upwind scheme
// ---------------------------------------------------------------------------

/// Extend a velocity field off the zero level set by constant extension.
///
/// Given φ and a velocity defined on/near the interface, propagates the
/// velocity into the bulk by solving ∂F/∂τ + sign(φ)·(∇φ/|∇φ|)·∇F = 0
/// using a first-order upwind scheme for `n_iter = 5` pseudo-time steps.
///
/// This ensures vel·∇φ = 0 away from the interface, making the velocity
/// consistent with a level-set Hamilton-Jacobi evolution.
///
/// Returns the extended velocity field of length `nx * ny`.
pub fn extend_velocity_upwind(phi: &[f64], vel: &[f64], nx: usize, ny: usize, dx: f64) -> Vec<f64> {
    assert_eq!(phi.len(), nx * ny);
    assert_eq!(vel.len(), nx * ny);

    let dt = 0.5 * dx;
    let n_iter = 5usize;
    let mut f = vel.to_vec();

    for _ in 0..n_iter {
        let f_old = f.clone();
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                let s = phi[idx].signum();
                if s == 0.0 {
                    continue;
                }

                // Unit normal n̂ = ∇φ/|∇φ| from φ (central differences, clamped at boundary)
                let phi_xp = if i + 1 < nx {
                    phi[j * nx + i + 1]
                } else {
                    phi[idx]
                };
                let phi_xm = if i > 0 { phi[j * nx + i - 1] } else { phi[idx] };
                let phi_yp = if j + 1 < ny {
                    phi[(j + 1) * nx + i]
                } else {
                    phi[idx]
                };
                let phi_ym = if j > 0 {
                    phi[(j - 1) * nx + i]
                } else {
                    phi[idx]
                };

                let gx = (phi_xp - phi_xm) / (2.0 * dx);
                let gy = (phi_yp - phi_ym) / (2.0 * dx);
                let gmag = (gx * gx + gy * gy).sqrt().max(1e-30);
                let nx_hat = gx / gmag;
                let ny_hat = gy / gmag;

                // sign(φ) · n̂ gives advection direction
                let ax = s * nx_hat;
                let ay = s * ny_hat;

                // First-order upwind advection of F
                let df_dx = if ax > 0.0 {
                    if i > 0 {
                        (f_old[idx] - f_old[j * nx + i - 1]) / dx
                    } else {
                        0.0
                    }
                } else if i + 1 < nx {
                    (f_old[j * nx + i + 1] - f_old[idx]) / dx
                } else {
                    0.0
                };
                let df_dy = if ay > 0.0 {
                    if j > 0 {
                        (f_old[idx] - f_old[(j - 1) * nx + i]) / dx
                    } else {
                        0.0
                    }
                } else if j + 1 < ny {
                    (f_old[(j + 1) * nx + i] - f_old[idx]) / dx
                } else {
                    0.0
                };

                f[idx] = f_old[idx] - dt * (ax * df_dx + ay * df_dy);
            }
        }
    }
    f
}

// ---------------------------------------------------------------------------
// Curvature-based regularization
// ---------------------------------------------------------------------------

/// Compute the mean curvature κ = div(∇φ/|∇φ|) at every grid point.
///
/// Uses second-order central differences for the numerator and a regularised
/// denominator |∇φ|_ε = max(|∇φ|, ε) with ε = 1e-6 to avoid division by zero.
///
/// Returns a `Vec<f64>` of length `nx * ny`.
pub fn curvature_field(phi: &[f64], nx: usize, ny: usize, dx: f64) -> Vec<f64> {
    assert_eq!(phi.len(), nx * ny);
    let eps = 1e-6_f64;
    let mut kappa = vec![0.0_f64; nx * ny];

    for j in 0..ny {
        for i in 0..nx {
            let idx = j * nx + i;

            // Neighbour values (zero-Neumann BC at boundary)
            let phixp = if i + 1 < nx {
                phi[j * nx + i + 1]
            } else {
                phi[idx]
            };
            let phixm = if i > 0 { phi[j * nx + i - 1] } else { phi[idx] };
            let phiyp = if j + 1 < ny {
                phi[(j + 1) * nx + i]
            } else {
                phi[idx]
            };
            let phiym = if j > 0 {
                phi[(j - 1) * nx + i]
            } else {
                phi[idx]
            };

            // Diagonal neighbours for cross-terms
            let phixpyp = if i + 1 < nx && j + 1 < ny {
                phi[(j + 1) * nx + i + 1]
            } else if i + 1 < nx {
                phi[j * nx + i + 1]
            } else if j + 1 < ny {
                phi[(j + 1) * nx + i]
            } else {
                phi[idx]
            };
            let phixmyp = if i > 0 && j + 1 < ny {
                phi[(j + 1) * nx + i - 1]
            } else if i > 0 {
                phi[j * nx + i - 1]
            } else if j + 1 < ny {
                phi[(j + 1) * nx + i]
            } else {
                phi[idx]
            };
            let phixpym = if i + 1 < nx && j > 0 {
                phi[(j - 1) * nx + i + 1]
            } else if i + 1 < nx {
                phi[j * nx + i + 1]
            } else if j > 0 {
                phi[(j - 1) * nx + i]
            } else {
                phi[idx]
            };
            let phixmym = if i > 0 && j > 0 {
                phi[(j - 1) * nx + i - 1]
            } else if i > 0 {
                phi[j * nx + i - 1]
            } else if j > 0 {
                phi[(j - 1) * nx + i]
            } else {
                phi[idx]
            };

            let phi_c = phi[idx];

            // First derivatives (central)
            let phi_x = (phixp - phixm) / (2.0 * dx);
            let phi_y = (phiyp - phiym) / (2.0 * dx);

            // Second derivatives
            let phi_xx = (phixp - 2.0 * phi_c + phixm) / (dx * dx);
            let phi_yy = (phiyp - 2.0 * phi_c + phiym) / (dx * dx);
            let phi_xy = (phixpyp - phixmyp - phixpym + phixmym) / (4.0 * dx * dx);

            // |∇φ|²
            let grad_sq = phi_x * phi_x + phi_y * phi_y;
            let grad_mag = grad_sq.sqrt().max(eps);

            // κ = (φ_xx·φ_y² - 2·φ_x·φ_y·φ_xy + φ_yy·φ_x²) / |∇φ|³
            let numerator =
                phi_xx * phi_y * phi_y - 2.0 * phi_x * phi_y * phi_xy + phi_yy * phi_x * phi_x;
            kappa[idx] = numerator / (grad_mag * grad_mag * grad_mag);
        }
    }
    kappa
}

/// Regularization velocity = κ_weight · κ(φ).
///
/// Adding this to the normal velocity penalizes high curvature (i.e., rough
/// boundaries), smoothing the interface during level-set evolution.
///
/// Returns a `Vec<f64>` of length `nx * ny`.
pub fn regularization_velocity(
    phi: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    kappa_weight: f64,
) -> Vec<f64> {
    let kappa = curvature_field(phi, nx, ny, dx);
    kappa.into_iter().map(|k| kappa_weight * k).collect()
}

// ---------------------------------------------------------------------------
// Parametric shape
// ---------------------------------------------------------------------------

/// Parametric shape: a set of control points defining a smooth boundary.
#[derive(Debug, Clone)]
pub struct ParametricShape {
    /// Control point x-coordinates (m)
    pub x: Vec<f64>,
    /// Control point y-coordinates (m)
    pub y: Vec<f64>,
}

impl ParametricShape {
    /// Create a circular parametric shape with n control points.
    pub fn circle(cx: f64, cy: f64, radius: f64, n_pts: usize) -> Self {
        use std::f64::consts::PI;
        let x = (0..n_pts)
            .map(|i| cx + radius * (2.0 * PI * i as f64 / n_pts as f64).cos())
            .collect();
        let y = (0..n_pts)
            .map(|i| cy + radius * (2.0 * PI * i as f64 / n_pts as f64).sin())
            .collect();
        Self { x, y }
    }

    /// Perimeter (approximate, piecewise linear).
    pub fn perimeter(&self) -> f64 {
        let n = self.x.len();
        (0..n)
            .map(|i| {
                let j = (i + 1) % n;
                let dx = self.x[j] - self.x[i];
                let dy = self.y[j] - self.y[i];
                (dx * dx + dy * dy).sqrt()
            })
            .sum()
    }

    /// Area (shoelace formula).
    pub fn area(&self) -> f64 {
        let n = self.x.len();
        let sum: f64 = (0..n)
            .map(|i| {
                let j = (i + 1) % n;
                self.x[i] * self.y[j] - self.x[j] * self.y[i]
            })
            .sum();
        (sum / 2.0).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Existing tests (unchanged)
    // ------------------------------------------------------------------

    #[test]
    fn level_set_solid_volume_one() {
        let ls = LevelSet::solid(10, 10, 1e-6);
        assert!((ls.volume_fraction() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn level_set_void_volume_zero() {
        let ls = LevelSet::void(10, 10, 1e-6);
        assert!(ls.volume_fraction() == 0.0);
    }

    #[test]
    fn level_set_circle_volume_fraction() {
        let n = 50;
        let dx = 1e-6;
        let r = 15e-6;
        let c = n as f64 / 2.0 * dx;
        let ls = LevelSet::circle(n, n, dx, c, c, r);
        let vf = ls.volume_fraction();
        // Circle area fraction ≈ π*r² / (n*dx)² ≈ π*15²/50² ≈ 0.283
        assert!(vf > 0.1 && vf < 0.5, "vf={vf:.3}");
    }

    #[test]
    fn level_set_advance_does_not_panic() {
        let mut ls = LevelSet::circle(20, 20, 1e-6, 10e-6, 10e-6, 5e-6);
        let v_n = vec![1.0; ls.phi.len()];
        ls.advance(&v_n, 0.1e-6);
    }

    #[test]
    fn level_set_perimeter_positive() {
        let ls = LevelSet::circle(20, 20, 1e-6, 10e-6, 10e-6, 5e-6);
        assert!(ls.perimeter() > 0.0);
    }

    #[test]
    fn parametric_circle_area() {
        let shape = ParametricShape::circle(0.0, 0.0, 1.0, 360);
        let area = shape.area();
        // Area of unit circle ≈ π
        assert!((area - std::f64::consts::PI).abs() < 0.1, "area={area:.3}");
    }

    #[test]
    fn parametric_circle_perimeter() {
        let shape = ParametricShape::circle(0.0, 0.0, 1.0, 360);
        let perim = shape.perimeter();
        // 2πr ≈ 6.283
        assert!(
            (perim - 2.0 * std::f64::consts::PI).abs() < 0.1,
            "perim={perim:.3}"
        );
    }

    #[test]
    fn reinitialize_does_not_panic() {
        let mut ls = LevelSet::circle(10, 10, 1e-6, 5e-6, 5e-6, 3e-6);
        ls.reinitialize(3);
    }

    // ------------------------------------------------------------------
    // New tests: LevelSetField
    // ------------------------------------------------------------------

    /// A circular signed-distance field should be positive at the centre and
    /// negative well outside the radius.
    #[test]
    fn level_set_circle_distance() {
        let nx = 40;
        let ny = 40;
        let dx = 1.0; // unit grid for simplicity
        let cx = 20.0;
        let cy = 20.0;
        let radius = 8.0;

        // Build signed-distance φ = radius − dist(x,y)
        let mut phi = vec![0.0f64; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64;
                let y = j as f64;
                let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                phi[j * nx + i] = radius - d;
            }
        }
        let lsf = LevelSetField { nx, ny, dx, phi };

        // Centre point (20, 20): dist = 0 → φ = +radius > 0
        let phi_center = lsf.phi[20 * nx + 20];
        assert!(phi_center > 0.0, "phi at center = {phi_center}");

        // Far exterior point (0, 0): dist ≈ 28.3 → φ ≈ 8 − 28.3 < 0
        let phi_exterior = lsf.phi[0];
        assert!(phi_exterior < 0.0, "phi at exterior = {phi_exterior}");
    }

    /// `to_density` should return values near 1 well inside the circle and
    /// near 0 well outside it.
    #[test]
    fn level_set_to_density() {
        let nx = 40;
        let ny = 40;
        let dx = 1.0;
        let cx = 20.0;
        let cy = 20.0;
        let radius = 8.0;

        let mut phi = vec![0.0f64; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64;
                let y = j as f64;
                let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                phi[j * nx + i] = radius - d;
            }
        }
        let lsf = LevelSetField { nx, ny, dx, phi };
        let density = lsf.to_density();

        // Deep inside: φ ≫ ε → density ≈ 1
        let rho_center = density[20 * nx + 20];
        assert!(rho_center > 0.99, "density at center = {rho_center}");

        // Deep outside: φ ≪ −ε → density ≈ 0
        let rho_exterior = density[0];
        assert!(rho_exterior < 0.01, "density at exterior = {rho_exterior}");
    }

    /// A uniform φ field has zero gradient everywhere, so curvature should
    /// be exactly zero (or within floating-point noise).
    #[test]
    fn curvature_uniform_zero() {
        let nx = 10;
        let ny = 10;
        let dx = 1e-6;
        // Constant φ = 1.0 → ∇φ = 0 → κ = 0
        let phi = vec![1.0f64; nx * ny];
        let kappa = curvature_field(&phi, nx, ny, dx);
        for &k in &kappa {
            assert!(
                k.abs() < 1e-10,
                "expected zero curvature for uniform phi, got {k}"
            );
        }
    }

    /// `extend_velocity_upwind` must return a vector of the same length.
    #[test]
    fn extend_velocity_length() {
        let nx = 12;
        let ny = 12;
        let dx = 1e-6;
        let mut phi = vec![0.0f64; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 * dx - 6e-6;
                let y = j as f64 * dx - 6e-6;
                phi[j * nx + i] = 3e-6 - (x * x + y * y).sqrt();
            }
        }
        let vel = vec![1.0f64; nx * ny];
        let ext = extend_velocity_upwind(&phi, &vel, nx, ny, dx);
        assert_eq!(ext.len(), nx * ny);
    }

    /// `regularization_velocity` values should be proportional to kappa_weight.
    #[test]
    fn regularization_velocity_scaling() {
        let nx = 20;
        let ny = 20;
        let dx = 1.0;
        let cx = 10.0;
        let cy = 10.0;
        let radius = 5.0;
        let phi: Vec<f64> = (0..ny)
            .flat_map(|j| {
                (0..nx).map(move |i| {
                    let d = ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2)).sqrt();
                    radius - d
                })
            })
            .collect();

        let v1 = regularization_velocity(&phi, nx, ny, dx, 1.0);
        let v2 = regularization_velocity(&phi, nx, ny, dx, 2.0);

        for (a, b) in v1.iter().zip(v2.iter()) {
            let ratio = if a.abs() > 1e-12 { b / a } else { 2.0 };
            assert!((ratio - 2.0).abs() < 1e-10, "ratio={ratio}");
        }
    }

    /// `LevelSetField::evolve` must not panic and must update φ when vel ≠ 0.
    #[test]
    fn level_set_field_evolve_changes_phi() {
        let nx = 10;
        let ny = 10;
        let dx = 1e-6;
        let mut lsf = LevelSetField::new(nx, ny, dx);
        // Set a simple linear φ so |∇φ| ≠ 0 and evolve will actually change values.
        for j in 0..ny {
            for i in 0..nx {
                lsf.phi[j * nx + i] = (i as f64 - 5.0) * dx;
            }
        }
        let phi_before = lsf.phi.clone();
        let vel = vec![1.0f64; nx * ny];
        lsf.evolve(&vel, 0.1e-6);
        let changed = phi_before
            .iter()
            .zip(lsf.phi.iter())
            .any(|(a, b)| (a - b).abs() > 1e-20);
        assert!(changed, "evolve did not change phi");
    }

    /// `from_density` with all 1.0 should produce all-positive φ.
    #[test]
    fn level_set_field_from_density_all_material() {
        let nx = 5;
        let ny = 5;
        let dx = 1e-6;
        let rho = vec![1.0f64; nx * ny];
        let lsf = LevelSetField::from_density(&rho, nx, ny, dx);
        assert!(lsf.phi.iter().all(|&p| p > 0.0));
    }

    /// `from_density` with all 0.0 should produce all-negative φ.
    #[test]
    fn level_set_field_from_density_all_void() {
        let nx = 5;
        let ny = 5;
        let dx = 1e-6;
        let rho = vec![0.0f64; nx * ny];
        let lsf = LevelSetField::from_density(&rho, nx, ny, dx);
        assert!(lsf.phi.iter().all(|&p| p < 0.0));
    }

    /// After reinitialization, |∇φ| should be close to 1 in the interior
    /// (away from the boundary and domain edges).
    #[test]
    fn level_set_field_reinitialize_grad_near_one() {
        let nx = 30;
        let ny = 30;
        let dx = 1.0;
        let cx = 15.0;
        let cy = 15.0;
        let radius = 8.0;

        // Start from a non-unit-gradient function: φ = (radius² − dist²)
        let mut phi: Vec<f64> = (0..ny)
            .flat_map(|j| {
                (0..nx).map(move |i| {
                    let d2 = (i as f64 - cx).powi(2) + (j as f64 - cy).powi(2);
                    radius * radius - d2
                })
            })
            .collect();

        let mut lsf = LevelSetField {
            nx,
            ny,
            dx,
            phi: phi.clone(),
        };
        lsf.reinitialize(30);
        phi = lsf.phi.clone();

        // Check |∇φ| ≈ 1 at a ring of interior points near the interface
        let mut max_err = 0.0_f64;
        for j in 2..ny - 2 {
            for i in 2..nx - 2 {
                let idx = j * nx + i;
                // Only test points near the interface (|φ| < 3*dx)
                if phi[idx].abs() > 3.0 * dx {
                    continue;
                }
                let gx = (phi[j * nx + i + 1] - phi[j * nx + i - 1]) / (2.0 * dx);
                let gy = (phi[(j + 1) * nx + i] - phi[(j - 1) * nx + i]) / (2.0 * dx);
                let grad_mag = (gx * gx + gy * gy).sqrt();
                let err = (grad_mag - 1.0).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        assert!(
            max_err < 0.25,
            "max |∇φ| error near interface = {max_err:.4}"
        );
    }
}
