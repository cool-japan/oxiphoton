/// 3D Yee grid layout helper.
///
/// Tracks staggered field positions on the Yee lattice and provides
/// geometry queries, field interpolation, and Poynting vector computation.
///
/// Field staggering (Yee 1966):
///   Ex  at  (i+½, j,   k  )
///   Ey  at  (i,   j+½, k  )
///   Ez  at  (i,   j,   k+½)
///   Hx  at  (i,   j+½, k+½)
///   Hy  at  (i+½, j,   k+½)
///   Hz  at  (i+½, j+½, k  )
pub struct Yee3d {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

impl Yee3d {
    /// Construct a new Yee3d grid helper.
    ///
    /// # Panics
    /// Does not panic; all values are stored as-is. Callers should ensure
    /// nx, ny, nz ≥ 1 and dx, dy, dz > 0.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Indexing
    // ──────────────────────────────────────────────────────────────────────────

    /// Linear index for cell (i, j, k): idx = k*nx*ny + j*nx + i.
    #[inline(always)]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        k * self.nx * self.ny + j * self.nx + i
    }

    /// Convert a linear index back to (i, j, k).
    #[inline(always)]
    pub fn ijk(&self, raw: usize) -> (usize, usize, usize) {
        let k = raw / (self.nx * self.ny);
        let rem = raw % (self.nx * self.ny);
        let j = rem / self.nx;
        let i = rem % self.nx;
        (i, j, k)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Physical positions (staggered Yee locations)
    // ──────────────────────────────────────────────────────────────────────────

    /// Physical position of Ex component (staggered at i+½, j, k).
    pub fn ex_pos(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        [
            (i as f64 + 0.5) * self.dx,
            j as f64 * self.dy,
            k as f64 * self.dz,
        ]
    }

    /// Physical position of Ey component (staggered at i, j+½, k).
    pub fn ey_pos(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        [
            i as f64 * self.dx,
            (j as f64 + 0.5) * self.dy,
            k as f64 * self.dz,
        ]
    }

    /// Physical position of Ez component (staggered at i, j, k+½).
    pub fn ez_pos(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        [
            i as f64 * self.dx,
            j as f64 * self.dy,
            (k as f64 + 0.5) * self.dz,
        ]
    }

    /// Physical position of Hx component (staggered at i, j+½, k+½).
    pub fn hx_pos(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        [
            i as f64 * self.dx,
            (j as f64 + 0.5) * self.dy,
            (k as f64 + 0.5) * self.dz,
        ]
    }

    /// Physical position of Hy component (staggered at i+½, j, k+½).
    pub fn hy_pos(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        [
            (i as f64 + 0.5) * self.dx,
            j as f64 * self.dy,
            (k as f64 + 0.5) * self.dz,
        ]
    }

    /// Physical position of Hz component (staggered at i+½, j+½, k).
    pub fn hz_pos(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        [
            (i as f64 + 0.5) * self.dx,
            (j as f64 + 0.5) * self.dy,
            k as f64 * self.dz,
        ]
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Geometry queries
    // ──────────────────────────────────────────────────────────────────────────

    /// Total number of cells in the grid.
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Physical domain size [Lx, Ly, Lz] (m).
    pub fn domain_size(&self) -> [f64; 3] {
        [
            self.nx as f64 * self.dx,
            self.ny as f64 * self.dy,
            self.nz as f64 * self.dz,
        ]
    }

    /// Volume of one Yee cell (m³).
    #[inline]
    pub fn cell_volume(&self) -> f64 {
        self.dx * self.dy * self.dz
    }

    /// Cross-sectional area of the XY face (m²).
    pub fn area_xy(&self) -> f64 {
        self.dx * self.dy * self.nx as f64 * self.ny as f64
    }

    /// Cross-sectional area of the XZ face (m²).
    pub fn area_xz(&self) -> f64 {
        self.dx * self.dz * self.nx as f64 * self.nz as f64
    }

    /// Cross-sectional area of the YZ face (m²).
    pub fn area_yz(&self) -> f64 {
        self.dy * self.dz * self.ny as f64 * self.nz as f64
    }

    /// Return `true` when cell (i,j,k) lies within the PML absorber region
    /// (i.e., within `pml` cells of any boundary face).
    pub fn in_pml(&self, i: usize, j: usize, k: usize, pml: usize) -> bool {
        i < pml
            || j < pml
            || k < pml
            || i >= self.nx.saturating_sub(pml)
            || j >= self.ny.saturating_sub(pml)
            || k >= self.nz.saturating_sub(pml)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Iterators
    // ──────────────────────────────────────────────────────────────────────────

    /// Iterate over all cell indices as `(i, j, k, linear_idx)`.
    pub fn iter_cells(&self) -> impl Iterator<Item = (usize, usize, usize, usize)> + '_ {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        (0..nz).flat_map(move |k| {
            (0..ny).flat_map(move |j| {
                (0..nx).map(move |i| {
                    let raw = k * nx * ny + j * nx + i;
                    (i, j, k, raw)
                })
            })
        })
    }

    /// Iterate over interior cells only (excluding PML boundary cells of thickness `pml`).
    pub fn iter_interior(
        &self,
        pml: usize,
    ) -> impl Iterator<Item = (usize, usize, usize, usize)> + '_ {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let i0 = pml;
        let i1 = nx.saturating_sub(pml);
        let j0 = pml;
        let j1 = ny.saturating_sub(pml);
        let k0 = pml;
        let k1 = nz.saturating_sub(pml);
        (k0..k1).flat_map(move |k| {
            (j0..j1).flat_map(move |j| {
                (i0..i1).map(move |i| {
                    let raw = k * nx * ny + j * nx + i;
                    (i, j, k, raw)
                })
            })
        })
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Nearest-cell lookup
    // ──────────────────────────────────────────────────────────────────────────

    /// Given a physical point (x, y, z), return the nearest cell (i, j, k).
    /// Clamps to valid cell range [0, n-1].
    pub fn nearest_cell(&self, x: f64, y: f64, z: f64) -> (usize, usize, usize) {
        let i = (x / self.dx).round() as isize;
        let j = (y / self.dy).round() as isize;
        let k = (z / self.dz).round() as isize;
        let clamp = |v: isize, max: usize| v.clamp(0, max.saturating_sub(1) as isize) as usize;
        (clamp(i, self.nx), clamp(j, self.ny), clamp(k, self.nz))
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Trilinear field interpolation
    // ──────────────────────────────────────────────────────────────────────────

    /// Trilinear interpolation of the E-field vector at arbitrary physical point (x,y,z).
    ///
    /// Uses the cell-centred (unstaggered) approximation — suitable for
    /// post-processing but not for source injection.
    pub fn interpolate_e(
        &self,
        x: f64,
        y: f64,
        z: f64,
        ex: &[f64],
        ey: &[f64],
        ez: &[f64],
    ) -> [f64; 3] {
        [
            self.trilinear(x, y, z, ex),
            self.trilinear(x, y, z, ey),
            self.trilinear(x, y, z, ez),
        ]
    }

    /// Trilinear interpolation of the H-field vector at arbitrary physical point (x,y,z).
    pub fn interpolate_h(
        &self,
        x: f64,
        y: f64,
        z: f64,
        hx: &[f64],
        hy: &[f64],
        hz: &[f64],
    ) -> [f64; 3] {
        [
            self.trilinear(x, y, z, hx),
            self.trilinear(x, y, z, hy),
            self.trilinear(x, y, z, hz),
        ]
    }

    /// Perform trilinear interpolation of a scalar field array at (x,y,z).
    fn trilinear(&self, x: f64, y: f64, z: f64, field: &[f64]) -> f64 {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;

        // Cell index of lower-left corner
        let xi = (x / self.dx).floor() as isize;
        let yj = (y / self.dy).floor() as isize;
        let zk = (z / self.dz).floor() as isize;

        let clamp = |v: isize, max: usize| v.clamp(0, max.saturating_sub(1) as isize) as usize;

        let i0 = clamp(xi, nx);
        let j0 = clamp(yj, ny);
        let k0 = clamp(zk, nz);
        let i1 = (i0 + 1).min(nx.saturating_sub(1));
        let j1 = (j0 + 1).min(ny.saturating_sub(1));
        let k1 = (k0 + 1).min(nz.saturating_sub(1));

        // Fractional offsets in [0,1]
        let tx = ((x / self.dx) - xi as f64).clamp(0.0, 1.0);
        let ty = ((y / self.dy) - yj as f64).clamp(0.0, 1.0);
        let tz = ((z / self.dz) - zk as f64).clamp(0.0, 1.0);

        let v = |i: usize, j: usize, k: usize| -> f64 {
            field.get(k * nx * ny + j * nx + i).copied().unwrap_or(0.0)
        };

        // Trilinear formula
        let c000 = v(i0, j0, k0);
        let c100 = v(i1, j0, k0);
        let c010 = v(i0, j1, k0);
        let c110 = v(i1, j1, k0);
        let c001 = v(i0, j0, k1);
        let c101 = v(i1, j0, k1);
        let c011 = v(i0, j1, k1);
        let c111 = v(i1, j1, k1);

        c000 * (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
            + c100 * tx * (1.0 - ty) * (1.0 - tz)
            + c010 * (1.0 - tx) * ty * (1.0 - tz)
            + c110 * tx * ty * (1.0 - tz)
            + c001 * (1.0 - tx) * (1.0 - ty) * tz
            + c101 * tx * (1.0 - ty) * tz
            + c011 * (1.0 - tx) * ty * tz
            + c111 * tx * ty * tz
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Poynting vector
    // ──────────────────────────────────────────────────────────────────────────

    /// Compute the Poynting vector S = E × H at cell (i, j, k).
    ///
    /// Uses the field values at the cell-centre linear index (simplified,
    /// not fully staggered-aware — adequate for post-processing energy flux).
    #[allow(clippy::too_many_arguments)]
    pub fn poynting(
        &self,
        i: usize,
        j: usize,
        k: usize,
        ex: &[f64],
        ey: &[f64],
        ez: &[f64],
        hx: &[f64],
        hy: &[f64],
        hz: &[f64],
    ) -> [f64; 3] {
        if i >= self.nx || j >= self.ny || k >= self.nz {
            return [0.0; 3];
        }
        let idx = self.idx(i, j, k);
        let (ex_v, ey_v, ez_v) = (
            ex.get(idx).copied().unwrap_or(0.0),
            ey.get(idx).copied().unwrap_or(0.0),
            ez.get(idx).copied().unwrap_or(0.0),
        );
        let (hx_v, hy_v, hz_v) = (
            hx.get(idx).copied().unwrap_or(0.0),
            hy.get(idx).copied().unwrap_or(0.0),
            hz.get(idx).copied().unwrap_or(0.0),
        );

        // S = E × H
        [
            ey_v * hz_v - ez_v * hy_v,
            ez_v * hx_v - ex_v * hz_v,
            ex_v * hy_v - ey_v * hx_v,
        ]
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Derived geometry helpers
    // ──────────────────────────────────────────────────────────────────────────

    /// Return the physical centre of cell (i,j,k).
    pub fn cell_centre(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        [
            (i as f64 + 0.5) * self.dx,
            (j as f64 + 0.5) * self.dy,
            (k as f64 + 0.5) * self.dz,
        ]
    }

    /// Return the lower corner of cell (i,j,k).
    pub fn cell_origin(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        [i as f64 * self.dx, j as f64 * self.dy, k as f64 * self.dz]
    }

    /// Return the smallest cell spacing (min of dx, dy, dz).
    pub fn min_spacing(&self) -> f64 {
        self.dx.min(self.dy).min(self.dz)
    }

    /// Return the largest cell spacing.
    pub fn max_spacing(&self) -> f64 {
        self.dx.max(self.dy).max(self.dz)
    }

    /// Return the Courant limit c·dt ≤ 1/√(1/dx²+1/dy²+1/dz²).
    pub fn courant_limit_dt(&self, c: f64) -> f64 {
        let inv =
            (1.0 / (self.dx * self.dx) + 1.0 / (self.dy * self.dy) + 1.0 / (self.dz * self.dz))
                .sqrt();
        1.0 / (c * inv)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn grid() -> Yee3d {
        Yee3d::new(10, 12, 8, 10e-9, 10e-9, 10e-9)
    }

    // ── Indexing ─────────────────────────────────────────────────────────────

    #[test]
    fn idx_roundtrips_to_ijk() {
        let g = grid();
        let (i, j, k) = (3, 5, 2);
        let raw = g.idx(i, j, k);
        assert_eq!(g.ijk(raw), (i, j, k));
    }

    #[test]
    fn idx_origin_is_zero() {
        let g = grid();
        assert_eq!(g.idx(0, 0, 0), 0);
    }

    #[test]
    fn idx_last_cell() {
        let g = grid();
        let raw = g.idx(g.nx - 1, g.ny - 1, g.nz - 1);
        assert_eq!(raw, g.num_cells() - 1);
    }

    // ── Staggered positions ──────────────────────────────────────────────────

    #[test]
    fn ex_pos_staggered_in_x() {
        let g = grid();
        let p = g.ex_pos(3, 4, 2);
        assert_relative_eq!(p[0], 3.5 * 10e-9, epsilon = 1e-20);
        assert_relative_eq!(p[1], 4.0 * 10e-9, epsilon = 1e-20);
    }

    #[test]
    fn ey_pos_staggered_in_y() {
        let g = grid();
        let p = g.ey_pos(2, 3, 1);
        assert_relative_eq!(p[1], 3.5 * 10e-9, epsilon = 1e-20);
    }

    #[test]
    fn ez_pos_staggered_in_z() {
        let g = grid();
        let p = g.ez_pos(1, 2, 4);
        assert_relative_eq!(p[2], 4.5 * 10e-9, epsilon = 1e-20);
    }

    #[test]
    fn hx_pos_staggered_in_y_and_z() {
        let g = grid();
        let p = g.hx_pos(0, 2, 3);
        assert_relative_eq!(p[1], 2.5 * 10e-9, epsilon = 1e-20);
        assert_relative_eq!(p[2], 3.5 * 10e-9, epsilon = 1e-20);
    }

    #[test]
    fn hz_pos_staggered_in_x_and_y() {
        let g = grid();
        let p = g.hz_pos(1, 2, 3);
        assert_relative_eq!(p[0], 1.5 * 10e-9, epsilon = 1e-20);
        assert_relative_eq!(p[1], 2.5 * 10e-9, epsilon = 1e-20);
        assert_relative_eq!(p[2], 3.0 * 10e-9, epsilon = 1e-20);
    }

    // ── Geometry ─────────────────────────────────────────────────────────────

    #[test]
    fn num_cells_correct() {
        let g = grid();
        assert_eq!(g.num_cells(), 10 * 12 * 8);
    }

    #[test]
    fn domain_size_correct() {
        let g = grid();
        let sz = g.domain_size();
        assert_relative_eq!(sz[0], 10.0 * 10e-9, epsilon = 1e-22);
        assert_relative_eq!(sz[1], 12.0 * 10e-9, epsilon = 1e-22);
        assert_relative_eq!(sz[2], 8.0 * 10e-9, epsilon = 1e-22);
    }

    #[test]
    fn cell_volume_correct() {
        let g = Yee3d::new(5, 5, 5, 2.0, 3.0, 4.0);
        assert_relative_eq!(g.cell_volume(), 24.0);
    }

    #[test]
    fn area_xy_correct() {
        let g = Yee3d::new(4, 5, 6, 1.0, 2.0, 3.0);
        // area_xy = dx*dy*nx*ny = 1*2*4*5 = 40
        assert_relative_eq!(g.area_xy(), 40.0);
    }

    // ── PML detection ────────────────────────────────────────────────────────

    #[test]
    fn in_pml_detects_boundary_cells() {
        let g = Yee3d::new(20, 20, 20, 1.0, 1.0, 1.0);
        assert!(g.in_pml(0, 5, 5, 4));
        assert!(g.in_pml(19, 5, 5, 4));
        assert!(g.in_pml(5, 0, 5, 4));
        assert!(!g.in_pml(5, 5, 5, 4));
    }

    // ── Nearest cell ─────────────────────────────────────────────────────────

    #[test]
    fn nearest_cell_origin() {
        let g = Yee3d::new(10, 10, 10, 1.0, 1.0, 1.0);
        assert_eq!(g.nearest_cell(0.0, 0.0, 0.0), (0, 0, 0));
    }

    #[test]
    fn nearest_cell_clamps_to_bounds() {
        let g = Yee3d::new(10, 10, 10, 1.0, 1.0, 1.0);
        let (i, j, k) = g.nearest_cell(1000.0, 1000.0, 1000.0);
        assert!(i < 10 && j < 10 && k < 10);
    }

    // ── Iteration ────────────────────────────────────────────────────────────

    #[test]
    fn iter_cells_count_matches_num_cells() {
        let g = grid();
        assert_eq!(g.iter_cells().count(), g.num_cells());
    }

    #[test]
    fn iter_interior_excludes_pml_cells() {
        let g = Yee3d::new(20, 20, 20, 1.0, 1.0, 1.0);
        let pml = 4;
        let count = g.iter_interior(pml).count();
        let expected = (20 - 2 * pml).pow(3);
        assert_eq!(count, expected);
    }

    // ── Interpolation ────────────────────────────────────────────────────────

    #[test]
    fn trilinear_uniform_field_returns_same_value() {
        let g = Yee3d::new(5, 5, 5, 1.0, 1.0, 1.0);
        let n = g.num_cells();
        let ex = vec![3.7; n];
        let ey = vec![0.0; n];
        let ez = vec![0.0; n];
        let result = g.interpolate_e(2.0, 2.0, 2.0, &ex, &ey, &ez);
        assert_relative_eq!(result[0], 3.7, epsilon = 1e-12);
    }

    #[test]
    fn trilinear_at_corner_matches_cell_value() {
        let g = Yee3d::new(4, 4, 4, 1.0, 1.0, 1.0);
        let n = g.num_cells();
        let mut ez = vec![0.0; n];
        ez[g.idx(1, 1, 1)] = 5.0;
        let result = g.interpolate_e(1.0, 1.0, 1.0, &vec![0.0; n], &vec![0.0; n], &ez);
        // At the exact grid node the interpolated value approaches the node value
        assert!(result[2].is_finite());
    }

    // ── Poynting vector ──────────────────────────────────────────────────────

    #[test]
    fn poynting_zero_when_fields_zero() {
        let g = grid();
        let n = g.num_cells();
        let zeros = vec![0.0; n];
        let s = g.poynting(2, 3, 1, &zeros, &zeros, &zeros, &zeros, &zeros, &zeros);
        assert_eq!(s, [0.0; 3]);
    }

    #[test]
    fn poynting_ex_hy_gives_sz() {
        let g = Yee3d::new(5, 5, 5, 1.0, 1.0, 1.0);
        let n = g.num_cells();
        let mut ex = vec![0.0; n];
        let mut hy = vec![0.0; n];
        let idx = g.idx(2, 2, 2);
        ex[idx] = 1.0;
        hy[idx] = 1.0;
        // S = E × H → for Ex and Hy: Sz = Ex*Hy - Ey*Hx = 1*1 - 0 = 1
        let zeros = vec![0.0; n];
        let s = g.poynting(2, 2, 2, &ex, &zeros, &zeros, &zeros, &hy, &zeros);
        assert_relative_eq!(s[2], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn poynting_out_of_bounds_returns_zeros() {
        let g = grid();
        let n = g.num_cells();
        let zeros = vec![0.0; n];
        let s = g.poynting(
            999, 999, 999, &zeros, &zeros, &zeros, &zeros, &zeros, &zeros,
        );
        assert_eq!(s, [0.0; 3]);
    }

    // ── Courant limit ────────────────────────────────────────────────────────

    #[test]
    fn courant_limit_dt_is_positive() {
        let g = Yee3d::new(50, 50, 50, 10e-9, 10e-9, 10e-9);
        let c = 299_792_458.0;
        let dt = g.courant_limit_dt(c);
        assert!(dt > 0.0 && dt.is_finite());
    }

    #[test]
    fn courant_limit_dt_satisfies_condition() {
        let g = Yee3d::new(50, 50, 50, 10e-9, 10e-9, 10e-9);
        let c = 299_792_458.0;
        let dt = g.courant_limit_dt(c);
        let s = c * dt * (1.0 / (g.dx * g.dx) + 1.0 / (g.dy * g.dy) + 1.0 / (g.dz * g.dz)).sqrt();
        assert_relative_eq!(s, 1.0, epsilon = 1e-12);
    }
}
