//! Structured grid specifications for FDTD and mode solvers.
//!
//! Provides uniform, non-uniform, and adaptive grid types in 1D, 2D, and 3D,
//! along with Yee cell helpers and subpixel averaging weights for material
//! interfaces.

/// 1D grid specification.
#[derive(Debug, Clone)]
pub struct GridSpec1d {
    /// Cell edge positions (length = n_cells + 1)
    pub edges: Vec<f64>,
    /// Cell center positions (length = n_cells)
    pub centers: Vec<f64>,
}

impl GridSpec1d {
    /// Uniform grid from `start` to `end` with `n` cells.
    pub fn uniform(start: f64, end: f64, n: usize) -> Self {
        assert!(n >= 1, "Need at least 1 cell");
        let dx = (end - start) / n as f64;
        let edges: Vec<f64> = (0..=n).map(|i| start + i as f64 * dx).collect();
        let centers: Vec<f64> = (0..n).map(|i| start + (i as f64 + 0.5) * dx).collect();
        Self { edges, centers }
    }

    /// Non-uniform grid from explicit edge positions.
    pub fn nonuniform(edges: Vec<f64>) -> Self {
        assert!(edges.len() >= 2, "Need at least 2 edges");
        let centers: Vec<f64> = edges.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
        Self { edges, centers }
    }

    /// Adaptive grid: uniform base + refined region `[x0, x1]` with `n_fine` cells.
    pub fn adaptive(
        start: f64,
        end: f64,
        n_coarse: usize,
        x0: f64,
        x1: f64,
        n_fine: usize,
    ) -> Self {
        let mut edges: Vec<f64> = Vec::new();
        // Coarse region left
        let dx_l = (x0 - start) / n_coarse as f64;
        for i in 0..=n_coarse {
            let e = start + i as f64 * dx_l;
            if e <= x0 + 1e-14 * (end - start) {
                edges.push(e);
            }
        }
        // Fine region
        let dx_f = (x1 - x0) / n_fine as f64;
        for i in 1..=n_fine {
            edges.push(x0 + i as f64 * dx_f);
        }
        // Coarse region right
        let dx_r = (end - x1) / n_coarse as f64;
        for i in 1..=n_coarse {
            edges.push(x1 + i as f64 * dx_r);
        }
        edges.dedup_by(|a, b| (*a - *b).abs() < 1e-20);
        let centers: Vec<f64> = edges.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
        Self { edges, centers }
    }

    /// Number of cells.
    pub fn n_cells(&self) -> usize {
        self.centers.len()
    }

    /// Cell spacings (dx for each cell).
    pub fn spacings(&self) -> Vec<f64> {
        self.edges.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Minimum cell size.
    pub fn dx_min(&self) -> f64 {
        self.spacings()
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    }

    /// Maximum cell size.
    pub fn dx_max(&self) -> f64 {
        self.spacings().iter().cloned().fold(0.0_f64, f64::max)
    }

    /// Find the cell index containing position `x` (clamped to valid range).
    pub fn find_cell(&self, x: f64) -> usize {
        if x <= self.edges[0] {
            return 0;
        }
        if x >= self.edges[self.edges.len() - 1] {
            return self.n_cells() - 1;
        }
        match self
            .edges
            .binary_search_by(|e| e.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(i) => i.min(self.n_cells() - 1),
            Err(i) => (i - 1).min(self.n_cells() - 1),
        }
    }

    /// Subpixel averaging weight for a material interface at position `x_interface`.
    ///
    /// Returns (weight_left, weight_right) for cell `cell_idx` assuming
    /// the interface lies within the cell.
    pub fn subpixel_weights(&self, cell_idx: usize, x_interface: f64) -> (f64, f64) {
        let x_lo = self.edges[cell_idx];
        let x_hi = self.edges[cell_idx + 1];
        let dx = x_hi - x_lo;
        let frac = ((x_interface - x_lo) / dx).clamp(0.0, 1.0);
        (1.0 - frac, frac)
    }
}

/// 2D grid specification (tensor product of two 1D grids).
#[derive(Debug, Clone)]
pub struct GridSpec2d {
    pub x: GridSpec1d,
    pub y: GridSpec1d,
}

impl GridSpec2d {
    /// Uniform 2D grid.
    pub fn uniform(x0: f64, x1: f64, nx: usize, y0: f64, y1: f64, ny: usize) -> Self {
        Self {
            x: GridSpec1d::uniform(x0, x1, nx),
            y: GridSpec1d::uniform(y0, y1, ny),
        }
    }

    /// Number of cells in x.
    pub fn nx(&self) -> usize {
        self.x.n_cells()
    }

    /// Number of cells in y.
    pub fn ny(&self) -> usize {
        self.y.n_cells()
    }

    /// Flat index (row-major: i*ny + j) for cell (i, j).
    pub fn idx(&self, i: usize, j: usize) -> usize {
        i * self.ny() + j
    }

    /// Yee cell: E-field at cell centers, H-field at cell edges.
    ///
    /// Returns the E-field (cell center) positions as (x, y) pairs.
    pub fn yee_e_positions(&self) -> Vec<(f64, f64)> {
        let mut out = Vec::with_capacity(self.nx() * self.ny());
        for i in 0..self.nx() {
            for j in 0..self.ny() {
                out.push((self.x.centers[i], self.y.centers[j]));
            }
        }
        out
    }

    /// Yee cell: H-field (Hz) positions at cell corners (edges in both x and y).
    pub fn yee_h_positions(&self) -> Vec<(f64, f64)> {
        let nx_h = self.x.edges.len();
        let ny_h = self.y.edges.len();
        let mut out = Vec::with_capacity(nx_h * ny_h);
        for i in 0..nx_h {
            for j in 0..ny_h {
                out.push((self.x.edges[i], self.y.edges[j]));
            }
        }
        out
    }

    /// Subpixel averaging weight for a circular interface of radius `r` centered at (`cx`, `cy`).
    ///
    /// Returns the fill fraction (0.0 = all outside, 1.0 = all inside) for cell `(i, j)`.
    /// Uses a simple Monte Carlo estimation with `n_samples` random points.
    pub fn circle_fill_fraction(&self, i: usize, j: usize, cx: f64, cy: f64, r: f64) -> f64 {
        let x0 = self.x.edges[i];
        let x1 = self.x.edges[i + 1];
        let y0 = self.y.edges[j];
        let y1 = self.y.edges[j + 1];
        let n = 16usize;
        let mut inside = 0usize;
        for ix in 0..n {
            for iy in 0..n {
                let xp = x0 + (ix as f64 + 0.5) / n as f64 * (x1 - x0);
                let yp = y0 + (iy as f64 + 0.5) / n as f64 * (y1 - y0);
                let dx = xp - cx;
                let dy = yp - cy;
                if dx * dx + dy * dy <= r * r {
                    inside += 1;
                }
            }
        }
        inside as f64 / (n * n) as f64
    }
}

/// 3D grid specification.
#[derive(Debug, Clone)]
pub struct GridSpec3d {
    pub x: GridSpec1d,
    pub y: GridSpec1d,
    pub z: GridSpec1d,
}

impl GridSpec3d {
    /// Uniform 3D grid.
    #[allow(clippy::too_many_arguments)]
    pub fn uniform(
        x0: f64,
        x1: f64,
        nx: usize,
        y0: f64,
        y1: f64,
        ny: usize,
        z0: f64,
        z1: f64,
        nz: usize,
    ) -> Self {
        Self {
            x: GridSpec1d::uniform(x0, x1, nx),
            y: GridSpec1d::uniform(y0, y1, ny),
            z: GridSpec1d::uniform(z0, z1, nz),
        }
    }

    /// nx × ny × nz grid.
    pub fn nx(&self) -> usize {
        self.x.n_cells()
    }
    pub fn ny(&self) -> usize {
        self.y.n_cells()
    }
    pub fn nz(&self) -> usize {
        self.z.n_cells()
    }

    /// Flat index (i*ny*nz + j*nz + k).
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.ny() + j) * self.nz() + k
    }

    /// Total number of cells.
    pub fn n_total(&self) -> usize {
        self.nx() * self.ny() * self.nz()
    }

    /// Check if a box region `[xa,xb] × [ya,yb] × [za,zb]` overlaps cell `(i,j,k)`.
    #[allow(clippy::too_many_arguments)]
    pub fn box_overlaps_cell(
        &self,
        i: usize,
        j: usize,
        k: usize,
        xa: f64,
        xb: f64,
        ya: f64,
        yb: f64,
        za: f64,
        zb: f64,
    ) -> bool {
        let xi = self.x.edges[i];
        let xi1 = self.x.edges[i + 1];
        let yj = self.y.edges[j];
        let yj1 = self.y.edges[j + 1];
        let zk = self.z.edges[k];
        let zk1 = self.z.edges[k + 1];
        xi1 > xa && xi < xb && yj1 > ya && yj < yb && zk1 > za && zk < zb
    }

    /// Fill a material index map for a box region.
    ///
    /// Sets `map[idx(i,j,k)] = material_id` for all cells overlapping the box.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_box_material(
        &self,
        map: &mut [usize],
        xa: f64,
        xb: f64,
        ya: f64,
        yb: f64,
        za: f64,
        zb: f64,
        material_id: usize,
    ) {
        for i in 0..self.nx() {
            for j in 0..self.ny() {
                for k in 0..self.nz() {
                    if self.box_overlaps_cell(i, j, k, xa, xb, ya, yb, za, zb) {
                        map[self.idx(i, j, k)] = material_id;
                    }
                }
            }
        }
    }
}

/// Yee cell helper for 1D grids.
///
/// In 1D FDTD (TEM wave), E-field is at integer positions and H-field at half-integer positions.
pub struct YeeCellHelper1d {
    /// Grid spacing
    pub dz: f64,
    /// Number of cells
    pub n: usize,
}

impl YeeCellHelper1d {
    pub fn new(n: usize, dz: f64) -> Self {
        Self { dz, n }
    }

    /// E-field position at index `i` (i * dz).
    pub fn e_pos(&self, i: usize) -> f64 {
        i as f64 * self.dz
    }

    /// H-field position at index `i` ((i + 0.5) * dz).
    pub fn h_pos(&self, i: usize) -> f64 {
        (i as f64 + 0.5) * self.dz
    }

    /// Courant limit dt_max = dz / c.
    pub fn courant_limit(&self) -> f64 {
        self.dz / 2.998e8
    }

    /// Subpixel-averaged permittivity at E-cell `i` given an interface at position `x_int`.
    ///
    /// Left material has `eps_left`, right has `eps_right`.
    /// Returns harmonically averaged eps (appropriate for normal component).
    pub fn harmonic_avg_eps(&self, i: usize, x_int: f64, eps_left: f64, eps_right: f64) -> f64 {
        let (w_left, w_right) =
            GridSpec1d::uniform(0.0, self.n as f64 * self.dz, self.n).subpixel_weights(i, x_int);
        1.0 / (w_left / eps_left + w_right / eps_right)
    }

    /// Subpixel-averaged permittivity using arithmetic average (tangential component).
    pub fn arithmetic_avg_eps(&self, i: usize, x_int: f64, eps_left: f64, eps_right: f64) -> f64 {
        let (w_left, w_right) =
            GridSpec1d::uniform(0.0, self.n as f64 * self.dz, self.n).subpixel_weights(i, x_int);
        w_left * eps_left + w_right * eps_right
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn uniform_1d_grid_edges_and_centers() {
        let g = GridSpec1d::uniform(0.0, 1.0, 4);
        assert_eq!(g.n_cells(), 4);
        assert_relative_eq!(g.edges[0], 0.0);
        assert_relative_eq!(g.edges[4], 1.0);
        assert_relative_eq!(g.centers[0], 0.125);
        assert_relative_eq!(g.centers[3], 0.875);
    }

    #[test]
    fn uniform_1d_spacings_constant() {
        let g = GridSpec1d::uniform(0.0, 2.0, 10);
        let sp = g.spacings();
        for s in &sp {
            assert_relative_eq!(*s, 0.2, epsilon = 1e-14);
        }
    }

    #[test]
    fn nonuniform_grid_centers() {
        let g = GridSpec1d::nonuniform(vec![0.0, 1.0, 3.0, 6.0]);
        assert_eq!(g.n_cells(), 3);
        assert_relative_eq!(g.centers[0], 0.5);
        assert_relative_eq!(g.centers[1], 2.0);
        assert_relative_eq!(g.centers[2], 4.5);
    }

    #[test]
    fn find_cell_clamps_to_bounds() {
        let g = GridSpec1d::uniform(0.0, 1.0, 10);
        assert_eq!(g.find_cell(-0.5), 0);
        assert_eq!(g.find_cell(1.5), 9);
    }

    #[test]
    fn find_cell_interior() {
        let g = GridSpec1d::uniform(0.0, 1.0, 10);
        assert_eq!(g.find_cell(0.25), 2);
        assert_eq!(g.find_cell(0.55), 5);
    }

    #[test]
    fn subpixel_weights_sum_to_one() {
        let g = GridSpec1d::uniform(0.0, 1.0, 10);
        let (wl, wr) = g.subpixel_weights(3, 0.35);
        assert_relative_eq!(wl + wr, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn grid2d_uniform() {
        let g = GridSpec2d::uniform(0.0, 1.0, 4, 0.0, 2.0, 8);
        assert_eq!(g.nx(), 4);
        assert_eq!(g.ny(), 8);
        assert_eq!(g.idx(1, 2), 10);
    }

    #[test]
    fn grid2d_circle_fill_interior() {
        let g = GridSpec2d::uniform(-1.0, 1.0, 20, -1.0, 1.0, 20);
        // Cell at center should be almost entirely inside circle r=0.9
        let frac = g.circle_fill_fraction(10, 10, 0.0, 0.0, 0.9);
        assert!(frac > 0.5, "Center cell fill={frac:.3}");
    }

    #[test]
    fn grid2d_circle_fill_exterior() {
        let g = GridSpec2d::uniform(-1.0, 1.0, 20, -1.0, 1.0, 20);
        // Corner cell at (0,0) far from center should be mostly outside r=0.5
        let frac = g.circle_fill_fraction(0, 0, 0.0, 0.0, 0.5);
        assert!(frac < 0.5, "Corner cell fill={frac:.3}");
    }

    #[test]
    fn grid3d_idx_and_fill() {
        // Use a 10×10×10 grid so cells at the edges (e.g. [0.0,0.1]) lie entirely
        // outside the box [0.2,0.8] and are not filled.
        let g = GridSpec3d::uniform(0.0, 1.0, 10, 0.0, 1.0, 10, 0.0, 1.0, 10);
        let mut map = vec![0usize; g.n_total()];
        g.fill_box_material(&mut map, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 1);
        let inside = map.iter().filter(|&&m| m == 1).count();
        assert!(inside > 0, "Should fill some cells");
        assert!(inside < g.n_total(), "Should not fill all cells");
    }

    #[test]
    fn yee_cell_helper_positions() {
        let yee = YeeCellHelper1d::new(100, 10e-9);
        assert_relative_eq!(yee.e_pos(5), 50e-9, epsilon = 1e-20);
        assert_relative_eq!(yee.h_pos(5), 55e-9, epsilon = 1e-20);
    }

    #[test]
    fn yee_courant_limit() {
        let yee = YeeCellHelper1d::new(100, 10e-9);
        let dt_max = yee.courant_limit();
        assert!(dt_max > 0.0);
        assert!(dt_max < 1e-16, "Courant dt = {dt_max:.3e}");
    }

    #[test]
    fn adaptive_grid_has_fine_region() {
        let g = GridSpec1d::adaptive(0.0, 10e-6, 5, 4e-6, 6e-6, 20);
        let sp = g.spacings();
        let dx_min = sp.iter().cloned().fold(f64::INFINITY, f64::min);
        let dx_max = sp.iter().cloned().fold(0.0_f64, f64::max);
        assert!(dx_min < dx_max, "Fine region should have smaller cells");
    }
}
