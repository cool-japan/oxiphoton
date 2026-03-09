//! PIC Waveguide Routing Primitives.
//!
//! Provides analytical models for common PIC routing elements:
//! - Circular and Euler bends
//! - S-bend offset connectors
//! - Waveguide crossings
//! - Adiabatic and linear tapers
//! - Manhattan layout router

use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// WaveguideBend
// ─────────────────────────────────────────────────────────────────────────────

/// Circular arc waveguide bend.
///
/// The bend loss model uses the Marcatili–Schmeltzer radiation-mode coupling
/// formula, which gives loss that decays exponentially with bend radius.
#[derive(Debug, Clone)]
pub struct WaveguideBend {
    /// Bend radius (µm)
    pub radius_um: f64,
    /// Subtended angle (degrees)
    pub angle_deg: f64,
    /// Waveguide width (nm)
    pub width_nm: f64,
    /// Precomputed bend loss (dB) – updated by `bend_loss_db()`
    pub loss_db: f64,
}

impl WaveguideBend {
    /// Construct a new bend.  `loss_db` is initialised to 0; call
    /// \[`bend_loss_db`\] to compute it.
    ///
    /// # Arguments
    /// * `radius_um`  – Bend radius (µm)
    /// * `angle_deg`  – Subtended angle (degrees)
    /// * `width_nm`   – Waveguide width (nm)
    pub fn new(radius_um: f64, angle_deg: f64, width_nm: f64) -> Self {
        Self {
            radius_um,
            angle_deg,
            width_nm,
            loss_db: 0.0,
        }
    }

    /// Bend radiation loss (dB) using the exponential decay model.
    ///
    /// α_bend = C₁ · exp(−C₂ · R)
    ///
    /// where C₁ and C₂ depend on the numerical aperture (n_eff − n_clad).
    ///
    /// # Arguments
    /// * `n_eff`      – Effective index of guided mode
    /// * `n_clad`     – Cladding refractive index
    /// * `wavelength` – Free-space wavelength (m)
    pub fn bend_loss_db(&self, n_eff: f64, n_clad: f64, wavelength: f64) -> f64 {
        let delta_n = (n_eff - n_clad).max(1.0e-6);
        let k0 = 2.0 * PI / wavelength;
        // Decay constant γ = sqrt(k0² (n_eff² - n_clad²))
        let gamma = k0 * (n_eff * n_eff - n_clad * n_clad).sqrt().max(1.0e-10);
        let r_m = self.radius_um * 1.0e-6;
        // Loss coefficient (1/m): α = (π k0²) / (γ * R) * exp(-2/3 * γ * R * (δn/n_eff)^(3/2))
        // Simplified Marcatili form (per radian of arc)
        let exponent = -2.0 / 3.0 * gamma * r_m * (delta_n / n_eff).powf(1.5);
        let alpha_per_rad = (PI * k0 * k0) / (gamma * r_m) * exponent.exp();
        let arc_length_m = r_m * self.angle_deg * PI / 180.0;
        // Convert to dB: loss = α_per_rad * (arc_length_m / r_m) * 10/ln(10)
        let loss_nepers = alpha_per_rad * self.angle_deg * PI / 180.0 * arc_length_m;
        loss_nepers * 10.0 / (10.0_f64).ln()
    }

    /// Length of the Euler (clothoid) transition section that smoothly
    /// ramps from infinite radius down to `self.radius_um`.
    ///
    /// L_euler ≈ 0.4 * R  (empirical, keeps mode mismatch < 0.05 dB)
    pub fn euler_transition_length_um(&self) -> f64 {
        0.4 * self.radius_um
    }

    /// Find the minimum bend radius (µm) that achieves `target_db` loss or less.
    ///
    /// Uses bisection over the range \[1 µm, 1000 µm\].
    ///
    /// # Arguments
    /// * `target_db`  – Maximum acceptable bend loss (dB)
    /// * `n_eff`      – Effective index
    /// * `n_clad`     – Cladding index
    /// * `wavelength` – Free-space wavelength (m)
    pub fn min_radius_for_loss(
        &self,
        target_db: f64,
        n_eff: f64,
        n_clad: f64,
        wavelength: f64,
    ) -> f64 {
        let mut lo = 1.0_f64;
        let mut hi = 1000.0_f64;
        let test_angle = self.angle_deg;
        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            let b = WaveguideBend::new(mid, test_angle, self.width_nm);
            let loss = b.bend_loss_db(n_eff, n_clad, wavelength);
            if loss > target_db {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hi
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SBend
// ─────────────────────────────────────────────────────────────────────────────

/// S-bend waveguide: a pair of opposite circular arcs connecting two
/// parallel waveguides laterally offset by `offset_um`.
#[derive(Debug, Clone)]
pub struct SBend {
    /// Total S-bend length along the propagation axis (µm)
    pub length_um: f64,
    /// Lateral offset between input and output ports (µm)
    pub offset_um: f64,
    /// Waveguide width (nm)
    pub width_nm: f64,
}

impl SBend {
    /// Create a new S-bend.
    ///
    /// # Arguments
    /// * `length_um` – Total axial length (µm)
    /// * `offset_um` – Lateral (y) displacement (µm)
    /// * `width_nm`  – Waveguide width (nm)
    pub fn new(length_um: f64, offset_um: f64, width_nm: f64) -> Self {
        Self {
            length_um,
            offset_um,
            width_nm,
        }
    }

    /// Minimum bend radius at the S-bend inflection point (µm).
    ///
    /// For a sinusoidal S-bend: r_min = L² / (2π² · offset).
    pub fn minimum_radius_um(&self) -> f64 {
        let l = self.length_um;
        let d = self.offset_um.abs().max(1.0e-6);
        l * l / (2.0 * PI * PI * d)
    }

    /// Total insertion loss (dB) considering bend radiation loss.
    ///
    /// # Arguments
    /// * `n_eff`      – Effective index
    /// * `n_clad`     – Cladding index
    /// * `wavelength` – Wavelength (m)
    pub fn insertion_loss_db(&self, n_eff: f64, n_clad: f64, wavelength: f64) -> f64 {
        let r_min = self.minimum_radius_um().max(0.1);
        // Approximate by two 90° bends at r_min radius (conservative)
        let bend = WaveguideBend::new(r_min, 90.0, self.width_nm);
        2.0 * bend.bend_loss_db(n_eff, n_clad, wavelength)
    }

    /// Minimum adiabatic length (µm) for mode mismatch loss < 0.01 dB.
    ///
    /// Adiabatic criterion: dR/dz < R / L_beat, where L_beat = λ / (2 Δn_eff).
    ///
    /// # Arguments
    /// * `n_eff`      – Effective index
    /// * `wavelength` – Wavelength (m)
    pub fn adiabatic_length_um(&self, n_eff: f64, wavelength: f64) -> f64 {
        let lambda_um = wavelength * 1.0e6;
        let delta_n = 0.01 * n_eff; // typical Δn_eff for ±10% width change
        let l_beat_um = lambda_um / (2.0 * delta_n);
        // Need ≥ 5 beat lengths along the S-bend for adiabaticity
        5.0 * l_beat_um * self.offset_um.abs() / lambda_um
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WaveguideCrossing
// ─────────────────────────────────────────────────────────────────────────────

/// 90° waveguide crossing element.
///
/// Two design variants are supported:
/// 1. Width-broadened crossing (Gaussian broadening reduces crosstalk)
/// 2. MMI-based crossing (multimode self-imaging)
#[derive(Debug, Clone)]
pub struct WaveguideCrossing {
    /// Waveguide width at the crossing centre (nm)
    pub width_nm: f64,
    /// Through-port insertion loss (dB)
    pub insertion_loss_db: f64,
    /// Cross-port crosstalk (dB, should be < −30 dB)
    pub crosstalk_db: f64,
    /// Back-reflection into the input port (dB)
    pub back_reflection_db: f64,
}

impl WaveguideCrossing {
    /// Width-broadened crossing.  The waveguide is adiabatically widened
    /// to a Gaussian profile at the intersection to reduce diffraction.
    ///
    /// # Arguments
    /// * `width_nm` – Nominal waveguide width (nm) outside the crossing
    pub fn new_broadened(width_nm: f64) -> Self {
        // Broadened width ≈ 3× for 220 nm SOI (typical)
        let broadened = width_nm * 3.0;
        // Insertion loss decreases with broader crossing
        let il = 0.05 + 1000.0 / broadened;
        // Crosstalk improves approximately quadratically with broadening factor
        let xt = -20.0 - 10.0 * (broadened / 1000.0).log10();
        Self {
            width_nm,
            insertion_loss_db: il.clamp(0.01, 1.0),
            crosstalk_db: xt.clamp(-60.0, -10.0),
            back_reflection_db: -40.0,
        }
    }

    /// MMI-based crossing.  Uses a square multimode section for
    /// near-lossless and low-crosstalk crossing via self-imaging.
    ///
    /// # Arguments
    /// * `width_nm` – Nominal waveguide width (nm)
    pub fn new_mmi_based(width_nm: f64) -> Self {
        Self {
            width_nm,
            insertion_loss_db: 0.1,
            crosstalk_db: -40.0,
            back_reflection_db: -45.0,
        }
    }

    /// Total loss, counting both through loss and crosstalk leakage.
    pub fn total_loss_db(&self) -> f64 {
        // Power leaked to crosstalk port degrades through transmission
        let crosstalk_linear = 10.0_f64.powf(self.crosstalk_db / 10.0);
        let il_linear = 10.0_f64.powf(-self.insertion_loss_db / 10.0);
        // Total = insertion loss + crosstalk penalty
        -10.0 * (il_linear * (1.0 - crosstalk_linear)).log10()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TaperShape / WaveguideTaper
// ─────────────────────────────────────────────────────────────────────────────

/// Taper profile shape.
#[derive(Debug, Clone)]
pub enum TaperShape {
    /// Linear width variation: w(z) = w_start + (w_end − w_start) * z/L
    Linear,
    /// Parabolic width variation: w(z) = w_start + (w_end − w_start) * (z/L)²
    Parabolic,
    /// Exponential width variation: w(z) = w_start * exp(z/L * ln(w_end/w_start))
    Exponential,
    /// Adiabatic taper designed to keep mode-mismatch below the given threshold.
    Adiabatic {
        /// Target mode overlap loss (dB)
        mode_overlap_db: f64,
    },
}

/// Waveguide width taper for mode conversion or spot-size matching.
#[derive(Debug, Clone)]
pub struct WaveguideTaper {
    /// Taper length (µm)
    pub length_um: f64,
    /// Width at z = 0 (nm)
    pub width_start_nm: f64,
    /// Width at z = length_um (nm)
    pub width_end_nm: f64,
    /// Profile shape
    pub taper_type: TaperShape,
}

impl WaveguideTaper {
    /// Construct a linear taper.
    ///
    /// # Arguments
    /// * `length`  – Taper length (µm)
    /// * `w_start` – Starting width (nm)
    /// * `w_end`   – Ending width (nm)
    pub fn new_linear(length: f64, w_start: f64, w_end: f64) -> Self {
        Self {
            length_um: length,
            width_start_nm: w_start,
            width_end_nm: w_end,
            taper_type: TaperShape::Linear,
        }
    }

    /// Construct an adiabatic taper sized to keep loss below `target_loss_db`.
    ///
    /// The adiabatic criterion sets the minimum taper length:
    /// L ≥ (w_end − w_start) * n_eff / (n_clad * λ/w_avg)
    ///
    /// # Arguments
    /// * `w_start`       – Starting width (nm)
    /// * `w_end`         – Ending width (nm)
    /// * `target_loss_db` – Maximum allowed mode-mismatch loss (dB)
    /// * `n_eff`         – Effective index of the guided mode
    /// * `wavelength`    – Free-space wavelength (m)
    pub fn new_adiabatic(
        w_start: f64,
        w_end: f64,
        target_loss_db: f64,
        n_eff: f64,
        wavelength: f64,
    ) -> Self {
        let lambda_nm = wavelength * 1.0e9;
        let dw = (w_end - w_start).abs();
        let w_avg = (w_start + w_end) / 2.0;
        // Safety factor: smaller target loss → longer taper
        let safety = 1.0 + 1.0 / target_loss_db.max(0.01);
        let length_um = safety * dw * n_eff / (lambda_nm / w_avg) / 1000.0;
        Self {
            length_um: length_um.max(5.0),
            width_start_nm: w_start,
            width_end_nm: w_end,
            taper_type: TaperShape::Adiabatic {
                mode_overlap_db: target_loss_db,
            },
        }
    }

    /// Width (nm) at axial position z (µm).
    ///
    /// Returns `width_start_nm` for z ≤ 0 and `width_end_nm` for z ≥ length_um.
    pub fn width_at(&self, z_um: f64) -> f64 {
        let t = (z_um / self.length_um).clamp(0.0, 1.0);
        let w0 = self.width_start_nm;
        let w1 = self.width_end_nm;
        match &self.taper_type {
            TaperShape::Linear => w0 + (w1 - w0) * t,
            TaperShape::Parabolic => w0 + (w1 - w0) * t * t,
            TaperShape::Exponential => {
                if (w1 / w0.max(1.0e-9)).abs() < 1.0e-12 {
                    w0
                } else {
                    w0 * (w1 / w0.max(1.0e-9)).ln().exp() * t.exp()
                }
            }
            TaperShape::Adiabatic { .. } => w0 + (w1 - w0) * t,
        }
    }

    /// Insertion loss (dB) from mode overlap mismatch at input and output facets.
    ///
    /// Uses the approximate overlap integral for Gaussian modes:
    /// η = 4 (w₁/w₂ + w₂/w₁)⁻²  where wᵢ ∝ 1/√n_eff_i
    ///
    /// # Arguments
    /// * `n_eff_start` – Effective index at the narrow end
    /// * `n_eff_end`   – Effective index at the wide end
    pub fn insertion_loss_db(&self, n_eff_start: f64, n_eff_end: f64) -> f64 {
        if (n_eff_start - n_eff_end).abs() < 1.0e-9 {
            return 0.0;
        }
        // Mode field radius ∝ 1/sqrt(n_eff * k0)
        let r1 = 1.0 / n_eff_start.sqrt();
        let r2 = 1.0 / n_eff_end.sqrt();
        let overlap = 4.0 * r1 * r2 / ((r1 + r2) * (r1 + r2));
        -10.0 * overlap.log10()
    }

    /// Check whether the taper satisfies the adiabatic criterion.
    ///
    /// Condition: dw/dz (nm/µm) < λ_nm * n_eff / (n_clad * w_avg_nm)
    ///
    /// # Arguments
    /// * `n_eff`      – Effective index
    /// * `n_clad`     – Cladding index
    /// * `wavelength` – Free-space wavelength (m)
    pub fn is_adiabatic(&self, n_eff: f64, n_clad: f64, wavelength: f64) -> bool {
        let dw_dz = (self.width_end_nm - self.width_start_nm).abs() / self.length_um;
        let w_avg = (self.width_start_nm + self.width_end_nm) / 2.0;
        let lambda_nm = wavelength * 1.0e9;
        let threshold = lambda_nm * n_eff / (n_clad * w_avg);
        dw_dz < threshold
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RouteSegment / WaveguideRoute
// ─────────────────────────────────────────────────────────────────────────────

/// A single segment of a PIC route.
#[derive(Debug, Clone)]
pub enum RouteSegment {
    /// Straight waveguide run.
    Straight {
        /// Length (µm)
        length_um: f64,
        /// Propagation direction (degrees from +x axis)
        angle_deg: f64,
    },
    /// Circular arc bend.
    Bend {
        /// Bend radius (µm)
        radius_um: f64,
        /// Subtended angle (degrees, signed: positive = CCW)
        angle_deg: f64,
    },
    /// Width taper.
    Taper {
        /// Taper length (µm)
        length_um: f64,
        /// Starting width (nm)
        w_start_nm: f64,
        /// Ending width (nm)
        w_end_nm: f64,
    },
}

impl RouteSegment {
    /// Returns the segment length in µm.
    pub fn length_um(&self) -> f64 {
        match self {
            Self::Straight { length_um, .. } => *length_um,
            Self::Bend {
                radius_um,
                angle_deg,
            } => radius_um * angle_deg.abs() * PI / 180.0,
            Self::Taper { length_um, .. } => *length_um,
        }
    }
}

/// A complete routed waveguide path between two ports.
#[derive(Debug, Clone)]
pub struct WaveguideRoute {
    /// Ordered list of route segments
    pub segments: Vec<RouteSegment>,
    /// Total physical length (µm)
    pub total_length_um: f64,
    /// Number of bend elements
    pub n_bends: usize,
}

impl WaveguideRoute {
    fn from_segments(segments: Vec<RouteSegment>) -> Self {
        let total_length_um = segments.iter().map(|s| s.length_um()).sum();
        let n_bends = segments
            .iter()
            .filter(|s| matches!(s, RouteSegment::Bend { .. }))
            .count();
        Self {
            segments,
            total_length_um,
            n_bends,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PicRouter
// ─────────────────────────────────────────────────────────────────────────────

/// Manhattan-style waveguide router for PIC layouts.
///
/// Generates L-routes and U-routes using straight segments and 90° bends.
/// Ports are described by their (x, y) position (µm) and exit direction
/// (degrees from +x axis, multiples of 90°).
#[derive(Debug, Clone)]
pub struct PicRouter {
    /// Routing grid resolution (µm)
    pub grid_size_um: f64,
    /// Minimum waveguide-to-waveguide spacing (µm)
    pub min_spacing_um: f64,
    /// Bend radius for 90° turns (µm)
    pub bend_radius_um: f64,
    /// Waveguide width (nm)
    pub waveguide_width_nm: f64,
}

impl PicRouter {
    /// Create a new router.
    ///
    /// # Arguments
    /// * `grid_um`      – Grid pitch (µm)
    /// * `min_spacing`  – Minimum spacing (µm)
    /// * `bend_radius`  – 90° bend radius (µm)
    /// * `width_nm`     – Waveguide width (nm)
    pub fn new(grid_um: f64, min_spacing: f64, bend_radius: f64, width_nm: f64) -> Self {
        Self {
            grid_size_um: grid_um,
            min_spacing_um: min_spacing,
            bend_radius_um: bend_radius,
            waveguide_width_nm: width_nm,
        }
    }

    /// Snap a coordinate to the routing grid.
    fn snap(&self, v: f64) -> f64 {
        (v / self.grid_size_um).round() * self.grid_size_um
    }

    /// Route between two ports using an L-route (one 90° bend) or a
    /// U-route (three 90° bends) when ports face each other.
    ///
    /// # Arguments
    /// * `port_a` – (x, y) of port A (µm)
    /// * `dir_a`  – Exit direction of port A (degrees)
    /// * `port_b` – (x, y) of port B (µm)
    /// * `dir_b`  – Exit direction of port B (degrees, toward the route)
    pub fn route(
        &self,
        port_a: [f64; 2],
        dir_a: f64,
        port_b: [f64; 2],
        _dir_b: f64,
    ) -> WaveguideRoute {
        let r = self.bend_radius_um;
        let dx = self.snap(port_b[0] - port_a[0]);
        let dy = self.snap(port_b[1] - port_a[1]);
        let dir_a_rad = dir_a * PI / 180.0;
        let is_horizontal = dir_a_rad.cos().abs() > dir_a_rad.sin().abs();

        let mut segs: Vec<RouteSegment> = Vec::new();

        if is_horizontal {
            // Exit horizontally → corner → run vertically to destination
            let horiz = dx.abs() - r;
            let vert = dy.abs() - r;
            if horiz > 0.0 {
                segs.push(RouteSegment::Straight {
                    length_um: horiz,
                    angle_deg: dir_a,
                });
            }
            let turn = if dy >= 0.0 { 90.0 } else { -90.0 };
            segs.push(RouteSegment::Bend {
                radius_um: r,
                angle_deg: turn,
            });
            if vert > 0.0 {
                segs.push(RouteSegment::Straight {
                    length_um: vert,
                    angle_deg: dir_a + turn,
                });
            }
            segs.push(RouteSegment::Bend {
                radius_um: r,
                angle_deg: -turn,
            });
        } else {
            // Exit vertically → corner → run horizontally
            let vert = dy.abs() - r;
            let horiz = dx.abs() - r;
            if vert > 0.0 {
                segs.push(RouteSegment::Straight {
                    length_um: vert,
                    angle_deg: dir_a,
                });
            }
            let turn = if dx >= 0.0 { -90.0 } else { 90.0 };
            segs.push(RouteSegment::Bend {
                radius_um: r,
                angle_deg: turn,
            });
            if horiz > 0.0 {
                segs.push(RouteSegment::Straight {
                    length_um: horiz,
                    angle_deg: dir_a + turn,
                });
            }
            segs.push(RouteSegment::Bend {
                radius_um: r,
                angle_deg: -turn,
            });
        }

        WaveguideRoute::from_segments(segs)
    }

    /// Estimate the insertion loss (dB) for a routed path.
    ///
    /// # Arguments
    /// * `route`          – The route to evaluate
    /// * `loss_db_per_cm` – Straight waveguide propagation loss (dB/cm)
    /// * `bend_loss_db`   – Loss per 90° bend (dB)
    pub fn route_loss_db(
        &self,
        route: &WaveguideRoute,
        loss_db_per_cm: f64,
        bend_loss_db: f64,
    ) -> f64 {
        let straight_cm = route.total_length_um * 1.0e-4; // µm → cm
        let propagation_loss = straight_cm * loss_db_per_cm;
        let bend_loss = route.n_bends as f64 * bend_loss_db;
        propagation_loss + bend_loss
    }

    /// Return the total physical route length (µm).
    pub fn route_length_um(&self, route: &WaveguideRoute) -> f64 {
        route.total_length_um
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bend_euler_transition_proportional() {
        let b = WaveguideBend::new(10.0, 90.0, 450.0);
        assert_abs_diff_eq!(b.euler_transition_length_um(), 4.0, epsilon = 1.0e-10);
    }

    #[test]
    fn test_bend_loss_increases_for_smaller_radius() {
        let n_eff = 2.45;
        let n_clad = 1.44;
        let wl = 1.55e-6;
        let b_small = WaveguideBend::new(2.0, 90.0, 450.0);
        let b_large = WaveguideBend::new(20.0, 90.0, 450.0);
        let loss_small = b_small.bend_loss_db(n_eff, n_clad, wl);
        let loss_large = b_large.bend_loss_db(n_eff, n_clad, wl);
        assert!(
            loss_small >= loss_large,
            "Smaller radius should have >= loss"
        );
    }

    #[test]
    fn test_sbend_minimum_radius_formula() {
        let sb = SBend::new(100.0, 10.0, 450.0);
        let r = sb.minimum_radius_um();
        // r = L² / (2 π² D) = 10000 / (2 * 9.87 * 10) ≈ 50.7
        assert_abs_diff_eq!(r, 100.0_f64.powi(2) / (2.0 * PI * PI * 10.0), epsilon = 0.1);
    }

    #[test]
    fn test_taper_width_at_endpoints() {
        let t = WaveguideTaper::new_linear(50.0, 300.0, 900.0);
        assert_abs_diff_eq!(t.width_at(0.0), 300.0, epsilon = 1.0e-6);
        assert_abs_diff_eq!(t.width_at(50.0), 900.0, epsilon = 1.0e-6);
    }

    #[test]
    fn test_taper_midpoint_linear() {
        let t = WaveguideTaper::new_linear(100.0, 400.0, 1000.0);
        assert_abs_diff_eq!(t.width_at(50.0), 700.0, epsilon = 1.0e-6);
    }

    #[test]
    fn test_adiabatic_taper_length_positive() {
        let t = WaveguideTaper::new_adiabatic(300.0, 1000.0, 0.1, 2.45, 1.55e-6);
        assert!(t.length_um > 0.0, "Adiabatic taper length must be positive");
    }

    #[test]
    fn test_crossing_total_loss_mmi() {
        let c = WaveguideCrossing::new_mmi_based(450.0);
        let total = c.total_loss_db();
        assert!(
            total > 0.0 && total < 2.0,
            "Unexpected crossing loss: {total} dB"
        );
    }

    #[test]
    fn test_router_route_has_segments() {
        let router = PicRouter::new(1.0, 3.0, 5.0, 450.0);
        let route = router.route([0.0, 0.0], 0.0, [100.0, 50.0], 180.0);
        assert!(!route.segments.is_empty());
        assert!(route.total_length_um > 0.0);
    }

    #[test]
    fn test_router_loss_positive() {
        let router = PicRouter::new(1.0, 3.0, 5.0, 450.0);
        let route = router.route([0.0, 0.0], 0.0, [200.0, 100.0], 180.0);
        let loss = router.route_loss_db(&route, 2.0, 0.05);
        assert!(loss > 0.0, "Route loss must be positive");
    }
}
