// Hx update: dHx/dt = -(1/mu) * dEz/dy
// Hx[i,j] idx j*nx+i, update range i in [0,nx), j in [0,ny)
// Forward diff in y; FAR zero at j+1==ny.

@group(0) @binding(0) var<storage, read_write> hx      : array<f32>; // nx*(ny+1)
@group(0) @binding(1) var<storage, read>       ez      : array<f32>; // nx*ny
@group(0) @binding(2) var<storage, read_write> psi_hx_y: array<f32>; // nx*(ny+1)
// pml_h_y: [b_h[0..ny], c_h[0..ny], kappa_h[0..ny]]
@group(0) @binding(3) var<storage, read>       pml_h_y : array<f32>; // 3*ny

struct Dims { nx: u32, ny: u32, dx: f32, dy: f32, dt: f32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(4) var<uniform> dims: Dims;

const MU0: f32 = 1.2566370614359173e-6;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let nx = dims.nx;
    let ny = dims.ny;

    if (i >= nx || j >= ny) { return; }

    let idx_hx = j * nx + i;

    // Forward diff Ez in y: FAR zero at j+1==ny
    var ez_top: f32 = 0.0;
    if (j + 1u < ny) {
        ez_top = ez[(j + 1u) * nx + i];
    }
    let dez_dy = (ez_top - ez[j * nx + i]) / dims.dy;

    let b_hy     = pml_h_y[j];
    let c_hy     = pml_h_y[ny + j];
    let kappa_hy = pml_h_y[2u * ny + j];

    let psi_new = b_hy * psi_hx_y[idx_hx] + c_hy * dez_dy;
    psi_hx_y[idx_hx] = psi_new;

    hx[idx_hx] = hx[idx_hx] - dims.dt / MU0 * (dez_dy / kappa_hy + psi_new);
}
