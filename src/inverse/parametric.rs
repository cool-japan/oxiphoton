//! Parametric optimization for photonic devices.
//!
//! Optimises a small number of continuous design parameters (e.g., waveguide
//! widths, gaps, coupler lengths) using gradient-free or gradient-based methods.
//!
//! Supported algorithms:
//!   - Nelder-Mead simplex (gradient-free, robust for < 20 parameters)
//!   - Gradient descent with momentum
//!   - Particle swarm (for global optimization)
//!   - Multi-start wrapper (runs optimizer from many random starts)

// ─── Convergence history ──────────────────────────────────────────────────────

/// Records iteration numbers and objective values during optimization.
pub struct ConvergenceHistory {
    /// Iteration indices at which values were recorded.
    pub iters: Vec<usize>,
    /// Objective (FOM) values at each recorded iteration.
    pub values: Vec<f64>,
}

impl ConvergenceHistory {
    /// Create an empty history.
    pub fn new() -> Self {
        Self {
            iters: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Append an iteration/value pair.
    pub fn push(&mut self, iter: usize, value: f64) {
        self.iters.push(iter);
        self.values.push(value);
    }

    /// Return the best (maximum) value seen, or `f64::NEG_INFINITY` if empty.
    pub fn best(&self) -> f64 {
        self.values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Return `true` if the range of the last few recorded values is less than `tol`.
    ///
    /// Uses the last `window` entries; if fewer entries exist returns `false`.
    pub fn converged(&self, tol: f64) -> bool {
        let window = 10;
        let n = self.values.len();
        if n < window {
            return false;
        }
        let recent = &self.values[n - window..];
        let hi = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lo = recent.iter().cloned().fold(f64::INFINITY, f64::min);
        (hi - lo).abs() < tol
    }
}

impl Default for ConvergenceHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Parametric problem ───────────────────────────────────────────────────────

/// Parametric optimization problem specification.
pub struct ParametricProblem {
    /// Number of parameters
    pub n_params: usize,
    /// Lower bounds for each parameter
    pub lower: Vec<f64>,
    /// Upper bounds for each parameter
    pub upper: Vec<f64>,
    /// Current parameter vector
    pub params: Vec<f64>,
    /// Best FOM seen
    pub best_fom: f64,
    /// Best parameters seen
    pub best_params: Vec<f64>,
    /// History of FOM values
    pub history: Vec<f64>,
}

impl ParametricProblem {
    /// Create a parametric problem with bounds.
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Self {
        let n = lower.len();
        assert_eq!(lower.len(), upper.len());
        let params: Vec<f64> = lower
            .iter()
            .zip(upper.iter())
            .map(|(l, u)| (l + u) / 2.0)
            .collect();
        Self {
            n_params: n,
            best_fom: f64::NEG_INFINITY,
            best_params: params.clone(),
            params,
            lower,
            upper,
            history: Vec::new(),
        }
    }

    /// Set initial parameters.
    pub fn set_params(&mut self, params: Vec<f64>) {
        assert_eq!(params.len(), self.n_params);
        self.params = params;
    }

    /// Clip parameters to bounds.
    pub fn clip(&self, params: &mut [f64]) {
        for (i, p) in params.iter_mut().enumerate() {
            *p = p.clamp(self.lower[i], self.upper[i]);
        }
    }

    /// Record FOM evaluation; update best if improved.
    pub fn record(&mut self, params: Vec<f64>, fom: f64) {
        self.history.push(fom);
        if fom > self.best_fom {
            self.best_fom = fom;
            self.best_params = params;
        }
    }

    /// Number of function evaluations so far.
    pub fn n_evals(&self) -> usize {
        self.history.len()
    }

    /// Convergence: FOM improvement < tol over last `window` evaluations.
    pub fn is_converged(&self, tol: f64, window: usize) -> bool {
        let n = self.history.len();
        if n < window {
            return false;
        }
        let recent = &self.history[n - window..];
        let max = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = recent.iter().cloned().fold(f64::INFINITY, f64::min);
        (max - min).abs() < tol
    }
}

// ─── Nelder-Mead ─────────────────────────────────────────────────────────────

/// Nelder-Mead simplex optimizer.
///
/// Gradient-free method suitable for ≤ 20 continuous parameters.
///
/// # Two APIs
///
/// * **Simple**: `NelderMead::new(x0)` → `.run(f, max_iter, tol)` minimises `f`.
/// * **Manual**: `NelderMead::default()` + `optimize()` / `step()` maximises `f` (legacy).
pub struct NelderMead {
    /// Initial point (used by `run`).
    pub x0: Vec<f64>,
    /// Reflection coefficient α
    pub alpha: f64,
    /// Expansion coefficient γ
    pub gamma: f64,
    /// Contraction coefficient ρ
    pub rho: f64,
    /// Shrink coefficient σ
    pub sigma: f64,
    /// Convergence history (populated by `run`).
    pub history: ConvergenceHistory,
}

impl NelderMead {
    /// Create an optimizer starting from `x0`.
    ///
    /// The `run` method **minimises** the objective `f`.
    pub fn new(x0: Vec<f64>) -> Self {
        Self {
            x0,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
            history: ConvergenceHistory::new(),
        }
    }

    /// Minimise `f` for up to `max_iter` iterations or until simplex diameter < `tol`.
    ///
    /// Returns `(best_params, best_value)`.
    pub fn run(&mut self, f: impl Fn(&[f64]) -> f64, max_iter: usize, tol: f64) -> (Vec<f64>, f64) {
        let n = self.x0.len();
        // Build initial simplex: x0 + unit perturbations
        let step = if self.x0.iter().any(|&v| v.abs() > 1e-12) {
            self.x0
                .iter()
                .cloned()
                .map(|v| {
                    if v.abs() > 1e-12 {
                        v.abs() * 0.05
                    } else {
                        0.00025
                    }
                })
                .collect::<Vec<_>>()
        } else {
            vec![0.00025; n]
        };
        let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        simplex.push(self.x0.clone());
        for i in 0..n {
            let mut v = self.x0.clone();
            v[i] += step[i];
            simplex.push(v);
        }
        // Evaluate (minimisation: negate for internal sort, keep raw)
        let mut fvals: Vec<f64> = simplex.iter().map(|x| f(x)).collect();

        for iter in 0..max_iter {
            // Sort: best (lowest) first
            let mut order: Vec<usize> = (0..n + 1).collect();
            order.sort_by(|&a, &b| {
                fvals[a]
                    .partial_cmp(&fvals[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            simplex = order.iter().map(|&i| simplex[i].clone()).collect();
            fvals = order.iter().map(|&i| fvals[i]).collect();

            self.history.push(iter, fvals[0]);

            // Convergence: diameter of simplex
            let diam = simplex[1..].iter().fold(0.0_f64, |acc, v| {
                let d: f64 = v
                    .iter()
                    .zip(&simplex[0])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                acc.max(d)
            });
            if diam < tol {
                break;
            }

            // Centroid of all but worst
            let mut centroid = vec![0.0_f64; n];
            for v in simplex.iter().take(n) {
                for (j, c) in centroid.iter_mut().enumerate() {
                    *c += v[j];
                }
            }
            for c in centroid.iter_mut() {
                *c /= n as f64;
            }

            let worst = &simplex[n];
            let f_best = fvals[0];
            let f_worst = fvals[n];
            let f_second_worst = fvals[n - 1];

            // Reflect
            let xr: Vec<f64> = (0..n)
                .map(|j| centroid[j] + self.alpha * (centroid[j] - worst[j]))
                .collect();
            let fr = f(&xr);

            if fr < f_best {
                // Expand
                let xe: Vec<f64> = (0..n)
                    .map(|j| centroid[j] + self.gamma * (xr[j] - centroid[j]))
                    .collect();
                let fe = f(&xe);
                if fe < fr {
                    simplex[n] = xe;
                    fvals[n] = fe;
                } else {
                    simplex[n] = xr;
                    fvals[n] = fr;
                }
            } else if fr < f_second_worst {
                simplex[n] = xr;
                fvals[n] = fr;
            } else {
                // Contract
                let xc: Vec<f64> = if fr < f_worst {
                    (0..n)
                        .map(|j| centroid[j] + self.rho * (xr[j] - centroid[j]))
                        .collect()
                } else {
                    (0..n)
                        .map(|j| centroid[j] + self.rho * (worst[j] - centroid[j]))
                        .collect()
                };
                let fc = f(&xc);
                if fc < f_worst {
                    simplex[n] = xc;
                    fvals[n] = fc;
                } else {
                    // Shrink
                    let best = simplex[0].clone();
                    for i in 1..=n {
                        for (j, v) in simplex[i].iter_mut().enumerate() {
                            *v = best[j] + self.sigma * (*v - best[j]);
                        }
                        fvals[i] = f(&simplex[i]);
                    }
                }
            }
        }

        // Return best
        let best_idx = fvals
            .iter()
            .enumerate()
            .fold(0, |bi, (i, &fv)| if fv < fvals[bi] { i } else { bi });
        (simplex[best_idx].clone(), fvals[best_idx])
    }

    // ── Legacy maximisation API ──────────────────────────────────────────────

    /// One step of the Nelder-Mead algorithm (maximises `fom`).
    ///
    /// `simplex` and `fom_vals` are sorted best-first (highest value first).
    pub fn step<F>(&self, simplex: &mut Vec<Vec<f64>>, fom_vals: &mut Vec<f64>, mut fom: F)
    where
        F: FnMut(&[f64]) -> f64,
    {
        let n = simplex[0].len();
        let n1 = simplex.len();

        // Sort descending (best first)
        let mut idx: Vec<usize> = (0..n1).collect();
        idx.sort_by(|&a, &b| {
            fom_vals[b]
                .partial_cmp(&fom_vals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let sorted_s: Vec<Vec<f64>> = idx.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_f: Vec<f64> = idx.iter().map(|&i| fom_vals[i]).collect();
        *simplex = sorted_s;
        *fom_vals = sorted_f;

        let mut centroid = vec![0.0_f64; n];
        for vertex in simplex.iter().take(n1 - 1) {
            for (j, c) in centroid.iter_mut().enumerate() {
                *c += vertex[j];
            }
        }
        for c in centroid.iter_mut() {
            *c /= (n1 - 1) as f64;
        }

        let worst = simplex.last().expect("simplex is non-empty (n+1 vertices)");
        let f_worst = *fom_vals.last().expect("fom_vals is non-empty (n+1 values)");
        let f_best = fom_vals[0];
        let f_second_worst = fom_vals[n1 - 2];

        let xr: Vec<f64> = (0..n)
            .map(|j| centroid[j] + self.alpha * (centroid[j] - worst[j]))
            .collect();
        let fr = fom(&xr);

        if fr > f_best {
            let xe: Vec<f64> = (0..n)
                .map(|j| centroid[j] + self.gamma * (xr[j] - centroid[j]))
                .collect();
            let fe = fom(&xe);
            if fe > fr {
                *simplex.last_mut().expect("simplex is non-empty") = xe;
                *fom_vals.last_mut().expect("fom_vals is non-empty") = fe;
            } else {
                *simplex.last_mut().expect("simplex is non-empty") = xr;
                *fom_vals.last_mut().expect("fom_vals is non-empty") = fr;
            }
        } else if fr >= f_second_worst {
            *simplex.last_mut().expect("simplex is non-empty") = xr;
            *fom_vals.last_mut().expect("fom_vals is non-empty") = fr;
        } else {
            let xc: Vec<f64> = if fr > f_worst {
                (0..n)
                    .map(|j| centroid[j] + self.rho * (xr[j] - centroid[j]))
                    .collect()
            } else {
                (0..n)
                    .map(|j| centroid[j] + self.rho * (worst[j] - centroid[j]))
                    .collect()
            };
            let fc = fom(&xc);
            if fc > f_worst {
                *simplex.last_mut().expect("simplex is non-empty") = xc;
                *fom_vals.last_mut().expect("fom_vals is non-empty") = fc;
            } else {
                let best = simplex[0].clone();
                for i in 1..n1 {
                    for (j, v) in simplex[i].iter_mut().enumerate() {
                        *v = best[j] + self.sigma * (*v - best[j]);
                    }
                    fom_vals[i] = fom(&simplex[i]);
                }
            }
        }
    }

    /// Initialise a simplex of n+1 vertices around `x0`.
    pub fn init_simplex(&self, x0: &[f64], step: f64) -> Vec<Vec<f64>> {
        let n = x0.len();
        let mut s = vec![x0.to_vec()];
        for i in 0..n {
            let mut v = x0.to_vec();
            v[i] += step;
            s.push(v);
        }
        s
    }

    /// Run Nelder-Mead for `max_iter` iterations maximising `fom`.
    pub fn optimize<F>(&self, x0: &[f64], step: f64, max_iter: usize, mut fom: F) -> (Vec<f64>, f64)
    where
        F: FnMut(&[f64]) -> f64,
    {
        let mut simplex = self.init_simplex(x0, step);
        let mut fom_vals: Vec<f64> = simplex.iter().map(|x| fom(x)).collect();
        for _ in 0..max_iter {
            self.step(&mut simplex, &mut fom_vals, &mut fom);
        }
        let best_idx = fom_vals.iter().enumerate().fold(
            0,
            |bi, (i, &fv)| {
                if fv > fom_vals[bi] {
                    i
                } else {
                    bi
                }
            },
        );
        (simplex[best_idx].clone(), fom_vals[best_idx])
    }
}

impl Default for NelderMead {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

// ─── PSO ──────────────────────────────────────────────────────────────────────

/// Particle Swarm Optimizer (minimisation).
///
/// Classic inertia-weight PSO with personal and global best tracking.
pub struct Pso {
    /// Number of particles.
    pub n_particles: usize,
    /// Dimension of the search space.
    pub dim: usize,
    /// Per-dimension bounds `(lo, hi)`.
    pub bounds: Vec<(f64, f64)>,
    /// Inertia weight ω.
    pub omega: f64,
    /// Cognitive coefficient c₁.
    pub c1: f64,
    /// Social coefficient c₂.
    pub c2: f64,
    /// Convergence history (populated by `run`).
    pub history: ConvergenceHistory,
    /// Internal PRNG state (xorshift64).
    rng: u64,
}

impl Pso {
    /// Create a new PSO with `n_particles` particles in a `dim`-dimensional space
    /// bounded by `bounds`.
    pub fn new(n_particles: usize, dim: usize, bounds: Vec<(f64, f64)>) -> Self {
        assert_eq!(bounds.len(), dim, "bounds length must equal dim");
        Self {
            n_particles,
            dim,
            bounds,
            omega: 0.729,
            c1: 1.494,
            c2: 1.494,
            history: ConvergenceHistory::new(),
            rng: 0x123456789ABCDEF1,
        }
    }

    /// Seed the internal RNG.
    pub fn seed(&mut self, seed: u64) {
        self.rng = seed | 1; // ensure non-zero
    }

    // xorshift64
    fn rand_f64(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        // Map to [0, 1)
        (x >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Minimise `f` for up to `max_iter` iterations.
    ///
    /// Returns `(best_params, best_value)`.
    pub fn run(&mut self, f: impl Fn(&[f64]) -> f64, max_iter: usize) -> (Vec<f64>, f64) {
        let np = self.n_particles;
        let d = self.dim;

        // Initialise positions and velocities uniformly within bounds
        let mut pos: Vec<Vec<f64>> = (0..np)
            .map(|_| {
                (0..d)
                    .map(|j| {
                        let (lo, hi) = self.bounds[j];
                        lo + self.rand_f64() * (hi - lo)
                    })
                    .collect()
            })
            .collect();

        let mut vel: Vec<Vec<f64>> = (0..np)
            .map(|_| {
                (0..d)
                    .map(|j| {
                        let (lo, hi) = self.bounds[j];
                        let span = hi - lo;
                        -span * 0.5 + self.rand_f64() * span
                    })
                    .collect()
            })
            .collect();

        let mut pbest_pos: Vec<Vec<f64>> = pos.clone();
        let mut pbest_val: Vec<f64> = (0..np).map(|i| f(&pos[i])).collect();

        // Global best
        let gbest_idx =
            pbest_val
                .iter()
                .enumerate()
                .fold(0, |bi, (i, &v)| if v < pbest_val[bi] { i } else { bi });
        let mut gbest_pos: Vec<f64> = pbest_pos[gbest_idx].clone();
        let mut gbest_val: f64 = pbest_val[gbest_idx];

        for iter in 0..max_iter {
            for i in 0..np {
                // Update velocity and position
                let r1s: Vec<f64> = (0..d).map(|_| self.rand_f64()).collect();
                let r2s: Vec<f64> = (0..d).map(|_| self.rand_f64()).collect();
                for (j, ((((v, p), pb), gb), (r1, r2))) in vel[i]
                    .iter_mut()
                    .zip(pos[i].iter_mut())
                    .zip(pbest_pos[i].iter())
                    .zip(gbest_pos.iter())
                    .zip(r1s.iter().zip(r2s.iter()))
                    .enumerate()
                {
                    *v = self.omega * *v + self.c1 * r1 * (*pb - *p) + self.c2 * r2 * (*gb - *p);
                    *p = (*p + *v).clamp(self.bounds[j].0, self.bounds[j].1);
                }
                // Evaluate
                let fv = f(&pos[i]);
                if fv < pbest_val[i] {
                    pbest_val[i] = fv;
                    pbest_pos[i] = pos[i].clone();
                    if fv < gbest_val {
                        gbest_val = fv;
                        gbest_pos = pos[i].clone();
                    }
                }
            }
            self.history.push(iter, gbest_val);
        }

        (gbest_pos, gbest_val)
    }
}

// ─── Multi-start ──────────────────────────────────────────────────────────────

/// Multi-start wrapper: runs an optimizer from multiple random starting points
/// and returns the globally best result.
pub struct MultiStart;

impl MultiStart {
    /// Run `optimizer_factory(x0)` from `n_starts` random starts within
    /// `bounds`, seeded deterministically from `rng_seed`.
    ///
    /// `optimizer_factory` receives an initial point `x0` and returns
    /// `(best_params, best_value)` (minimisation convention).
    ///
    /// Returns the `(params, value)` pair with the lowest objective value.
    pub fn run<F, O>(
        bounds: &[(f64, f64)],
        optimizer_factory: F,
        n_starts: usize,
        rng_seed: u64,
    ) -> (Vec<f64>, f64)
    where
        F: Fn(Vec<f64>) -> O,
        O: FnOnce() -> (Vec<f64>, f64),
    {
        let dim = bounds.len();
        let mut rng = rng_seed | 1;
        let mut best_params = vec![0.0_f64; dim];
        let mut best_val = f64::INFINITY;

        for _ in 0..n_starts {
            let x0: Vec<f64> = bounds
                .iter()
                .map(|&(lo, hi)| {
                    // xorshift64
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let r = (rng >> 11) as f64 / (1u64 << 53) as f64;
                    lo + r * (hi - lo)
                })
                .collect();

            let (params, val) = optimizer_factory(x0)();
            if val < best_val {
                best_val = val;
                best_params = params;
            }
        }

        (best_params, best_val)
    }
}

// ─── Gradient descent with momentum ──────────────────────────────────────────

/// Gradient descent with momentum (for differentiable objectives).
pub struct MomentumGD {
    /// Learning rate
    pub lr: f64,
    /// Momentum coefficient β
    pub beta: f64,
}

impl MomentumGD {
    pub fn new(lr: f64, beta: f64) -> Self {
        Self { lr, beta }
    }

    /// One gradient-descent step with momentum.
    ///
    /// - `params`: current parameters (modified in place)
    /// - `grad`: gradient ∇FOM (positive = increasing FOM)
    /// - `velocity`: momentum buffer (modified in place)
    /// - `lower`, `upper`: bounds for clipping
    pub fn step(
        &self,
        params: &mut [f64],
        grad: &[f64],
        velocity: &mut [f64],
        lower: &[f64],
        upper: &[f64],
    ) {
        for i in 0..params.len() {
            velocity[i] = self.beta * velocity[i] + (1.0 - self.beta) * grad[i];
            params[i] = (params[i] + self.lr * velocity[i]).clamp(lower[i], upper[i]);
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ConvergenceHistory ────────────────────────────────────────────────────

    #[test]
    fn convergence_history_works() {
        let mut h = ConvergenceHistory::new();
        assert_eq!(h.best(), f64::NEG_INFINITY);
        assert!(!h.converged(1e-6));

        h.push(0, 5.0);
        h.push(1, 3.0);
        h.push(2, 7.0);
        assert!((h.best() - 7.0).abs() < 1e-12);

        // Fill 10 entries with the same value → should converge
        for i in 3..13 {
            h.push(i, 7.0);
        }
        assert!(h.converged(1e-6));
    }

    // ── NelderMead (run / minimisation API) ───────────────────────────────────

    #[test]
    fn nelder_mead_quadratic() {
        // Minimise (x-1)² + (y-2)², start at (0,0)
        let mut nm = NelderMead::new(vec![0.0, 0.0]);
        let (best, val) = nm.run(|x| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2), 500, 1e-10);
        assert!((best[0] - 1.0).abs() < 0.02, "x={:.6}", best[0]);
        assert!((best[1] - 2.0).abs() < 0.02, "y={:.6}", best[1]);
        assert!(val < 1e-4, "val={:.2e}", val);
        assert!(!nm.history.iters.is_empty());
    }

    // ── PSO ───────────────────────────────────────────────────────────────────

    #[test]
    fn pso_quadratic() {
        // Minimise (x-1)² + (y-2)², bounds [-5,5]²
        let bounds = vec![(-5.0_f64, 5.0_f64), (-5.0_f64, 5.0_f64)];
        let mut pso = Pso::new(30, 2, bounds);
        pso.seed(42);
        let (best, val) = pso.run(|x| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2), 300);
        assert!((best[0] - 1.0).abs() < 0.1, "x={:.6}", best[0]);
        assert!((best[1] - 2.0).abs() < 0.1, "y={:.6}", best[1]);
        assert!(val < 0.02, "val={:.4}", val);
        assert!(!pso.history.iters.is_empty());
    }

    // ── MultiStart ────────────────────────────────────────────────────────────

    #[test]
    fn multi_start_finds_optimum() {
        // Minimise (x-1)² + (y-2)² from 5 random starts in [-5,5]²
        let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let (best, val) = MultiStart::run(
            &bounds,
            |x0| {
                move || {
                    let mut nm = NelderMead::new(x0);
                    nm.run(|x| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2), 300, 1e-10)
                }
            },
            5,
            12345,
        );
        assert!((best[0] - 1.0).abs() < 0.05, "x={:.6}", best[0]);
        assert!((best[1] - 2.0).abs() < 0.05, "y={:.6}", best[1]);
        assert!(val < 1e-4, "val={:.2e}", val);
    }

    // ── Legacy maximisation API ───────────────────────────────────────────────

    #[test]
    fn parametric_problem_init() {
        let p = ParametricProblem::new(vec![0.0; 3], vec![1.0; 3]);
        assert_eq!(p.n_params, 3);
        assert!(p.params.iter().all(|&v| (v - 0.5).abs() < 1e-10));
    }

    #[test]
    fn nelder_mead_optimizes_quadratic() {
        // Maximise -(x-0.3)² → optimal x=0.3
        let nm = NelderMead::new(vec![]);
        let (best, fom) = nm.optimize(&[0.0], 0.1, 100, |x| -(x[0] - 0.3).powi(2));
        assert!((best[0] - 0.3).abs() < 0.01, "best={:.4}", best[0]);
        assert!(fom > -1e-4);
    }

    #[test]
    fn nelder_mead_2d_quadratic() {
        let nm = NelderMead::new(vec![]);
        let (best, _fom) = nm.optimize(&[0.0, 0.0], 0.2, 200, |x| {
            -(x[0] - 0.5).powi(2) - (x[1] - 0.7).powi(2)
        });
        assert!((best[0] - 0.5).abs() < 0.05, "x0={:.4}", best[0]);
        assert!((best[1] - 0.7).abs() < 0.05, "x1={:.4}", best[1]);
    }

    #[test]
    fn parametric_convergence_check() {
        let mut p = ParametricProblem::new(vec![0.0], vec![1.0]);
        for i in 0..10 {
            p.record(vec![i as f64 / 10.0], 0.5);
        }
        assert!(p.is_converged(0.01, 5));
    }

    #[test]
    fn momentum_gd_moves_in_gradient_direction() {
        let mgd = MomentumGD::new(0.1, 0.9);
        let mut params = vec![0.0];
        let mut velocity = vec![0.0];
        mgd.step(&mut params, &[1.0], &mut velocity, &[-10.0], &[10.0]);
        assert!(params[0] > 0.0, "params={:.4}", params[0]);
    }
}
