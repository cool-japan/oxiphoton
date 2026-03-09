use crate::units::Wavelength;
use serde::{Deserialize, Serialize};

/// FDTD spatial and temporal grid parameters
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GridSpacing {
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

impl GridSpacing {
    pub fn uniform(d: f64) -> Self {
        Self {
            dx: d,
            dy: d,
            dz: d,
        }
    }

    /// Auto-select grid spacing: lambda_min / (n_max * ppw)
    pub fn auto(min_wavelength: Wavelength, max_index: f64, points_per_wavelength: usize) -> Self {
        let d = min_wavelength.0 / (max_index * points_per_wavelength as f64);
        Self::uniform(d)
    }
}

/// FDTD simulation dimensions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Dimensions {
    OneD { nz: usize },
    TwoD { nx: usize, ny: usize },
    ThreeD { nx: usize, ny: usize, nz: usize },
}

/// Boundary configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryConfig {
    pub pml_cells: usize,
    pub pml_m: f64,
    pub pml_r0: f64,
}

impl BoundaryConfig {
    /// Create PML boundary with given number of cells
    pub fn pml(cells: usize) -> Self {
        Self {
            pml_cells: cells,
            pml_m: 3.5,
            pml_r0: 1e-8,
        }
    }
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        Self::pml(20)
    }
}

/// Simulation result data
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub steps_completed: usize,
    pub time_elapsed_s: f64,
}

/// Full simulation configuration builder.
///
/// Combines spatial grid, boundary, and source configuration for a complete
/// FDTD setup. Provides auto-Courant selection and basic validation.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub grid: GridSpacing,
    pub dimensions: Dimensions,
    pub boundary: BoundaryConfig,
    /// Courant safety factor (default 0.99)
    pub courant_factor: f64,
    /// Total simulation time (s); 0 = run for `n_steps` steps
    pub total_time: f64,
    /// Number of steps (used if total_time == 0)
    pub n_steps: usize,
    /// Maximum refractive index in the domain (for auto-Courant)
    pub n_max: f64,
    /// Minimum wavelength (m) for auto grid spacing
    pub lambda_min: f64,
    /// Points per wavelength
    pub points_per_wavelength: usize,
}

impl SimulationConfig {
    /// Create a config from explicit grid spacing and dimensions.
    pub fn new(grid: GridSpacing, dimensions: Dimensions) -> Self {
        Self {
            grid,
            dimensions,
            boundary: BoundaryConfig::default(),
            courant_factor: 0.99,
            total_time: 0.0,
            n_steps: 1000,
            n_max: 1.0,
            lambda_min: 1550e-9,
            points_per_wavelength: 20,
        }
    }

    /// Auto-configure grid spacing from lambda_min and n_max.
    pub fn auto(dimensions: Dimensions, lambda_min: Wavelength, n_max: f64, ppw: usize) -> Self {
        let grid = GridSpacing::auto(lambda_min, n_max, ppw);
        Self {
            grid,
            dimensions,
            boundary: BoundaryConfig::default(),
            courant_factor: 0.99,
            total_time: 0.0,
            n_steps: 1000,
            n_max,
            lambda_min: lambda_min.0,
            points_per_wavelength: ppw,
        }
    }

    /// Set the boundary configuration.
    pub fn with_boundary(mut self, boundary: BoundaryConfig) -> Self {
        self.boundary = boundary;
        self
    }

    /// Set the Courant safety factor.
    pub fn with_courant(mut self, factor: f64) -> Self {
        self.courant_factor = factor.clamp(0.1, 1.0);
        self
    }

    /// Set total simulation time in seconds.
    pub fn with_total_time(mut self, t: f64) -> Self {
        self.total_time = t;
        self
    }

    /// Set number of steps (used if total_time == 0).
    pub fn with_n_steps(mut self, n: usize) -> Self {
        self.n_steps = n;
        self
    }

    /// Compute the optimal time step dt (s) using Courant condition.
    pub fn optimal_dt(&self) -> f64 {
        use crate::fdtd::courant::courant_dt;
        self.courant_factor * courant_dt(self.dimensions, self.grid, self.n_max)
    }

    /// Number of steps needed to reach `total_time`.
    pub fn steps_for_time(&self, dt: f64) -> usize {
        if self.total_time > 0.0 {
            (self.total_time / dt).ceil() as usize
        } else {
            self.n_steps
        }
    }

    /// Validate configuration; returns error string if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.grid.dx <= 0.0 || self.grid.dy <= 0.0 || self.grid.dz <= 0.0 {
            return Err("Grid spacing must be positive".to_string());
        }
        if self.n_max < 1.0 {
            return Err("n_max must be >= 1.0".to_string());
        }
        if self.boundary.pml_cells == 0 {
            return Err("PML must have at least 1 cell".to_string());
        }
        Ok(())
    }
}

/// Checkpoint data for resuming a simulation.
#[derive(Debug, Clone)]
pub struct SimulationCheckpoint {
    pub step: usize,
    pub time: f64,
    pub field_data: Vec<f64>,
}

impl SimulationCheckpoint {
    pub fn new(step: usize, time: f64, field_data: Vec<f64>) -> Self {
        Self {
            step,
            time,
            field_data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulation_config_auto() {
        let dims = Dimensions::TwoD { nx: 100, ny: 100 };
        let cfg = SimulationConfig::auto(dims, Wavelength::from_nm(1550.0), 3.5, 20);
        assert!(cfg.grid.dx > 0.0);
        assert!(cfg.grid.dx < 1550e-9 / 3.5 / 10.0, "Grid too coarse");
        cfg.validate().expect("Config should be valid");
    }

    #[test]
    fn simulation_config_optimal_dt() {
        let dims = Dimensions::OneD { nz: 100 };
        let cfg = SimulationConfig::auto(dims, Wavelength::from_nm(1000.0), 1.5, 15);
        let dt = cfg.optimal_dt();
        assert!(dt > 0.0);
        assert!(dt < 1e-12, "dt should be sub-ps");
    }

    #[test]
    fn steps_for_time_calculation() {
        let dims = Dimensions::OneD { nz: 100 };
        let cfg = SimulationConfig::auto(dims, Wavelength::from_nm(1550.0), 1.0, 20)
            .with_total_time(1e-12);
        let dt = cfg.optimal_dt();
        let n = cfg.steps_for_time(dt);
        assert!(n > 0);
    }

    #[test]
    fn config_validate_fails_zero_dx() {
        let dims = Dimensions::OneD { nz: 100 };
        let mut cfg = SimulationConfig::new(GridSpacing::uniform(0.0), dims);
        cfg.grid.dx = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_with_boundary_setter() {
        let dims = Dimensions::OneD { nz: 100 };
        let cfg = SimulationConfig::new(GridSpacing::uniform(10e-9), dims)
            .with_boundary(BoundaryConfig::pml(30));
        assert_eq!(cfg.boundary.pml_cells, 30);
    }

    #[test]
    fn config_courant_clamped() {
        let dims = Dimensions::OneD { nz: 100 };
        let cfg = SimulationConfig::new(GridSpacing::uniform(10e-9), dims).with_courant(2.0); // should clamp to 1.0
        assert!(cfg.courant_factor <= 1.0);
    }
}
