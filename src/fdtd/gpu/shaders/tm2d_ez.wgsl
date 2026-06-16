// Ez update: dEz/dt = (1/eps) * ((dHy/dx) - (dHx/dy))
// Ez[i,j] idx j*nx+i, range i in [0,nx), j in [0,ny)
// Backward diffs: PEC gives Hy[-1,j]=0 at i=0; Hx[i,-1]=0 at j=0.

@group(0) @binding(0) var<storage, read_write> ez      : array<f32>; // nx*ny
@group(0) @binding(1) var<storage, read>       hx      : array<f32>; // nx*(ny+1), idx j*nx+i
@group(0) @binding(2) var<storage, read>       hy      : array<f32>; // (nx+1)*ny, idx j*(nx+1)+i
@group(0) @binding(3) var<storage, read_write> psi_ez_x: array<f32>; // nx*ny
@group(0) @binding(4) var<storage, read_write> psi_ez_y: array<f32>; // nx*ny
@group(0) @binding(5) var<storage, read>       eps_r   : array<f32>; // nx*ny
// pml_e_x: [b_e[0..nx], c_e[0..nx], kappa_e[0..nx]]
@group(0) @binding(6) var<storage, read>       pml_e_x : array<f32>; // 3*nx
// pml_e_y: [b_e[0..ny], c_e[0..ny], kappa_e[0..ny]]
@group(0) @binding(7) var<storage, read>       pml_e_y : array<f32>; // 3*ny

struct Dims { nx: u32, ny: u32, dx: f32, dy: f32, dt: f32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(8) var<uniform> dims: Dims;

const EPS0: f32 = 8.854187817620389e-12;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let nx = dims.nx;
    let ny = dims.ny;

    if (i >= nx || j >= ny) { return; }

    let idx = j * nx + i;

    // dHy/dx: backward diff; Hy[-1,j]=0 at i=0 (PEC)
    // Hy idx: j*(nx+1)+i
    let hy_cur = hy[j * (nx + 1u) + i];
    var dhy_dx: f32;
    if (i > 0u) {
        dhy_dx = (hy_cur - hy[j * (nx + 1u) + i - 1u]) / dims.dx;
    } else {
        dhy_dx = hy_cur / dims.dx;
    }

    // dHx/dy: backward diff; Hx[i,-1]=0 at j=0 (PEC)
    // Hx idx: j*nx+i
    let hx_cur = hx[j * nx + i];
    var dhx_dy: f32;
    if (j > 0u) {
        dhx_dy = (hx_cur - hx[(j - 1u) * nx + i]) / dims.dy;
    } else {
        dhx_dy = hx_cur / dims.dy;
    }

    let b_ex     = pml_e_x[i];
    let c_ex     = pml_e_x[nx + i];
    let kappa_ex = pml_e_x[2u * nx + i];
    let b_ey     = pml_e_y[j];
    let c_ey     = pml_e_y[ny + j];
    let kappa_ey = pml_e_y[2u * ny + j];

    let psi_x_new = b_ex * psi_ez_x[idx] + c_ex * dhy_dx;
    let psi_y_new = b_ey * psi_ez_y[idx] + c_ey * dhx_dy;
    psi_ez_x[idx] = psi_x_new;
    psi_ez_y[idx] = psi_y_new;

    let eps = EPS0 * eps_r[idx];
    ez[idx] = ez[idx]
        + dims.dt / eps * ((dhy_dx / kappa_ex + psi_x_new) - (dhx_dy / kappa_ey + psi_y_new));
}
