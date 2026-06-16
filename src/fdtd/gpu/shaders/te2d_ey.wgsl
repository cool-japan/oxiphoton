// Ey field update with CPML corrections.
// Ey[i,j] at (i+0.5, j), indexed j*nx+i  for i in [0, nx), j in [0, ny]
// dEy/dt = -(1/eps) * dHz/dx
//
// Backward diff: (Hz[i,j] - Hz[i-1,j]) / dx; Hz[-1,j] = 0 at i=0 (PEC).
// OUTER ROW GUARD: at j=ny (the (ny+1)-th row), dhz_dx = 0.

@group(0) @binding(0) var<storage, read_write> ey      : array<f32>; // nx*(ny+1)
@group(0) @binding(1) var<storage, read>       hz      : array<f32>; // nx*ny
@group(0) @binding(2) var<storage, read_write> psi_ey_x: array<f32>; // nx*(ny+1)
@group(0) @binding(3) var<storage, read>       eps_ey  : array<f32>; // nx*(ny+1)
@group(0) @binding(4) var<storage, read>       pml_e_x : array<f32>; // 3*nx: [b_e, c_e, kappa_e]

struct Dims { nx: u32, ny: u32, dx: f32, dy: f32, dt: f32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(5) var<uniform> dims: Dims;

const EPS0: f32 = 8.854187817620389e-12;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let nx = dims.nx;
    let ny = dims.ny;

    // Dispatch covers i in [0, nx); j in [0, ny] = ny+1 rows
    if (i >= nx || j > ny) { return; }

    let idx_ey = j * nx + i;

    // Backward diff: Hz[i,j] - Hz[i-1,j]; Hz[-1,j] = 0 at i=0
    // Outer row guard: j == ny -> dhz_dx = 0 (no Hz cell at j=ny)
    var dhz_dx: f32 = 0.0;
    if (j < ny) {
        if (i > 0u) {
            dhz_dx = (hz[j * nx + i] - hz[j * nx + i - 1u]) / dims.dx;
        } else {
            dhz_dx = hz[j * nx + i] / dims.dx;
        }
    }

    let b_ex     = pml_e_x[i];
    let c_ex     = pml_e_x[nx + i];
    let kappa_ex = pml_e_x[2u * nx + i];

    let psi_new = b_ex * psi_ey_x[idx_ey] + c_ex * dhz_dx;
    psi_ey_x[idx_ey] = psi_new;

    let eps = EPS0 * eps_ey[idx_ey];
    ey[idx_ey] = ey[idx_ey] - dims.dt / eps * (dhz_dx / kappa_ex + psi_new);
}
