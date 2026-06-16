// Hz field update with CPML corrections.
// Hz[i,j] at (i+0.5, j+0.5), indexed j*nx+i  (nx*ny cells)
// dHz/dt = (1/mu) * (dEx/dy - dEy/dx)
//
// Forward differences (zeroing the FAR out-of-domain neighbour):
//   dEx/dy: (Ex[i,j+1] - Ex[i,j]) / dy;  Ex[i,ny] = 0 when j+1 == ny
//   dEy/dx: (Ey[i+1,j] - Ey[i,j]) / dx;  Ey[nx,j] = 0 when i+1 == nx
//
// PML packed layout (pml_h_x, pml_h_y): [b_h[0..n], c_h[0..n], kappa_h[0..n]]

@group(0) @binding(0) var<storage, read_write> hz      : array<f32>; // nx*ny
@group(0) @binding(1) var<storage, read>       ex      : array<f32>; // (nx+1)*ny, idx j*(nx+1)+i
@group(0) @binding(2) var<storage, read>       ey      : array<f32>; // nx*(ny+1), idx j*nx+i
@group(0) @binding(3) var<storage, read_write> psi_hz_x: array<f32>; // nx*ny
@group(0) @binding(4) var<storage, read_write> psi_hz_y: array<f32>; // nx*ny
@group(0) @binding(5) var<storage, read>       mu_hz   : array<f32>; // nx*ny (relative perm)
@group(0) @binding(6) var<storage, read>       pml_h_x : array<f32>; // 3*nx: [b_h, c_h, kappa_h]
@group(0) @binding(7) var<storage, read>       pml_h_y : array<f32>; // 3*ny: [b_h, c_h, kappa_h]

struct Dims { nx: u32, ny: u32, dx: f32, dy: f32, dt: f32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(8) var<uniform> dims: Dims;

const MU0: f32 = 1.2566370614359173e-6;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let nx = dims.nx;
    let ny = dims.ny;

    if (i >= nx || j >= ny) { return; }

    let idx_hz    = j * nx + i;
    let idx_ex_b  = j * (nx + 1u) + i; // Ex[i, j]

    // dEx/dy: forward diff; far neighbour Ex[i, j+1] = 0 when j+1 == ny
    var ex_top: f32 = 0.0;
    if (j + 1u < ny) {
        ex_top = ex[(j + 1u) * (nx + 1u) + i];
    }
    let dex_dy = (ex_top - ex[idx_ex_b]) / dims.dy;

    // dEy/dx: forward diff; far neighbour Ey[i+1, j] = 0 when i+1 == nx
    let idx_ey_l = j * nx + i; // Ey[i, j]
    var ey_right: f32 = 0.0;
    if (i + 1u < nx) {
        ey_right = ey[j * nx + i + 1u];
    }
    let dey_dx = (ey_right - ey[idx_ey_l]) / dims.dx;

    // CPML coefficients (packed layout)
    let b_hx     = pml_h_x[i];
    let c_hx     = pml_h_x[nx + i];
    let kappa_hx = pml_h_x[2u * nx + i];
    let b_hy     = pml_h_y[j];
    let c_hy     = pml_h_y[ny + j];
    let kappa_hy = pml_h_y[2u * ny + j];

    // Update psi — store new value, then use it in field update (matches CPU order)
    let psi_x_new = b_hx * psi_hz_x[idx_hz] + c_hx * dey_dx;
    let psi_y_new = b_hy * psi_hz_y[idx_hz] + c_hy * dex_dy;
    psi_hz_x[idx_hz] = psi_x_new;
    psi_hz_y[idx_hz] = psi_y_new;

    let mu = MU0 * mu_hz[idx_hz];
    hz[idx_hz] = hz[idx_hz]
        + dims.dt / mu * (dex_dy / kappa_hy - dey_dx / kappa_hx + psi_y_new - psi_x_new);
}
