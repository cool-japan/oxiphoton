// Ex field update with CPML corrections.
// Ex[i,j] at (i, j+0.5), indexed j*(nx+1)+i  for i in [0, nx], j in [0, ny)
// dEx/dt = (1/eps) * dHz/dy
//
// Backward diff: (Hz[i,j] - Hz[i,j-1]) / dy; Hz[i,-1] = 0 at j=0 (PEC).
// OUTER COLUMN GUARD: at i=nx (the (nx+1)-th column), dhz_dy = 0.

@group(0) @binding(0) var<storage, read_write> ex      : array<f32>; // (nx+1)*ny
@group(0) @binding(1) var<storage, read>       hz      : array<f32>; // nx*ny
@group(0) @binding(2) var<storage, read_write> psi_ex_y: array<f32>; // (nx+1)*ny
@group(0) @binding(3) var<storage, read>       eps_ex  : array<f32>; // (nx+1)*ny
@group(0) @binding(4) var<storage, read>       pml_e_y : array<f32>; // 3*ny: [b_e, c_e, kappa_e]

struct Dims { nx: u32, ny: u32, dx: f32, dy: f32, dt: f32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(5) var<uniform> dims: Dims;

const EPS0: f32 = 8.854187817620389e-12;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let nx = dims.nx;
    let ny = dims.ny;

    // Dispatch covers i in [0, nx] = nx+1 columns; j in [0, ny)
    if (i > nx || j >= ny) { return; }

    let idx_ex = j * (nx + 1u) + i;

    // Backward diff: Hz[i,j] - Hz[i,j-1]; Hz[i,-1] = 0 at j=0
    // Outer column guard: i == nx -> dhz_dy = 0 (no Hz cell at i=nx)
    var dhz_dy: f32 = 0.0;
    if (i < nx) {
        if (j > 0u) {
            dhz_dy = (hz[j * nx + i] - hz[(j - 1u) * nx + i]) / dims.dy;
        } else {
            dhz_dy = hz[j * nx + i] / dims.dy;
        }
    }

    let b_ey     = pml_e_y[j];
    let c_ey     = pml_e_y[ny + j];
    let kappa_ey = pml_e_y[2u * ny + j];

    let psi_new = b_ey * psi_ex_y[idx_ex] + c_ey * dhz_dy;
    psi_ex_y[idx_ex] = psi_new;

    let eps = EPS0 * eps_ex[idx_ex];
    ex[idx_ex] = ex[idx_ex] + dims.dt / eps * (dhz_dy / kappa_ey + psi_new);
}
