use crate::geometry::Shape2d;
use crate::material::DispersiveMaterial;
use crate::units::Wavelength;

/// 1D Yee grid: TEM wave propagating in z, Ex and Hy fields
#[derive(Debug, Clone)]
pub struct Yee1d {
    /// Number of cells
    pub nz: usize,
    /// Cell spacing (m)
    pub dz: f64,
    /// Ex field at integer z positions \[0..nz\]
    pub ex: Vec<f64>,
    /// Hy field at half-integer z positions \[0..nz\] (Hy\[i\] is at z=(i+0.5)*dz)
    pub hy: Vec<f64>,
    /// Relative permittivity at E-field positions
    pub eps_r: Vec<f64>,
    /// Relative permeability at H-field positions
    pub mu_r: Vec<f64>,
}

impl Yee1d {
    pub fn new(nz: usize, dz: f64) -> Self {
        Self {
            nz,
            dz,
            ex: vec![0.0; nz],
            hy: vec![0.0; nz],
            eps_r: vec![1.0; nz],
            mu_r: vec![1.0; nz],
        }
    }

    /// Fill a slab region [z_start, z_end) with a material
    pub fn fill_material(
        &mut self,
        z_start: f64,
        z_end: f64,
        material: &dyn DispersiveMaterial,
        wavelength: Wavelength,
    ) {
        let ri = material.refractive_index(wavelength);
        let eps = ri.n * ri.n - ri.k * ri.k;
        let i_start = (z_start / self.dz).floor() as usize;
        let i_end = ((z_end / self.dz).ceil() as usize).min(self.nz);
        for i in i_start..i_end {
            self.eps_r[i] = eps.max(1.0);
        }
    }
}

/// 2D Yee grid for TE mode: Hz, Ex, Ey fields
///
/// Grid layout (nx x ny cells):
/// - Hz\[i,j\] at (i+0.5, j+0.5) — cell centers
/// - Ex\[i,j\] at (i,   j+0.5) — y-edge midpoints
/// - Ey\[i,j\] at (i+0.5, j  ) — x-edge midpoints
#[derive(Debug, Clone)]
pub struct Yee2dTe {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    /// Ex field: size (nx+1) x ny — indexed as \[j*(nx+1) + i\]
    pub ex: Vec<f64>,
    /// Ey field: size nx x (ny+1) — indexed as \[j*nx + i\]
    pub ey: Vec<f64>,
    /// Hz field: size nx x ny — indexed as \[j*nx + i\]
    pub hz: Vec<f64>,
    /// Relative permittivity at Ex positions (nx+1)*ny
    pub eps_ex: Vec<f64>,
    /// Relative permittivity at Ey positions nx*(ny+1)
    pub eps_ey: Vec<f64>,
    /// Relative permeability at Hz positions nx*ny
    pub mu_hz: Vec<f64>,
}

impl Yee2dTe {
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        Self {
            nx,
            ny,
            dx,
            dy,
            ex: vec![0.0; (nx + 1) * ny],
            ey: vec![0.0; nx * (ny + 1)],
            hz: vec![0.0; nx * ny],
            eps_ex: vec![1.0; (nx + 1) * ny],
            eps_ey: vec![1.0; nx * (ny + 1)],
            mu_hz: vec![1.0; nx * ny],
        }
    }

    pub fn ex_idx(&self, i: usize, j: usize) -> usize {
        j * (self.nx + 1) + i
    }

    pub fn ey_idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    pub fn hz_idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Fill a shape with material properties
    pub fn fill_shape(
        &mut self,
        shape: &dyn Shape2d,
        material: &dyn DispersiveMaterial,
        wavelength: Wavelength,
    ) {
        let ri = material.refractive_index(wavelength);
        let eps = (ri.n * ri.n - ri.k * ri.k).max(1.0);

        // Fill Ex positions: (i, j+0.5) for i in [0..nx+1], j in [0..ny]
        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let dy = self.dy;
        for j in 0..ny {
            for i in 0..=nx {
                let x = i as f64 * dx;
                let y = (j as f64 + 0.5) * dy;
                if shape.contains(x, y) {
                    let idx = j * (nx + 1) + i;
                    self.eps_ex[idx] = eps;
                }
            }
        }

        // Fill Ey positions: (i+0.5, j) for i in [0..nx], j in [0..ny+1]
        for j in 0..=ny {
            for i in 0..nx {
                let x = (i as f64 + 0.5) * dx;
                let y = j as f64 * dy;
                if shape.contains(x, y) {
                    let idx = j * nx + i;
                    self.eps_ey[idx] = eps;
                }
            }
        }
    }
}
