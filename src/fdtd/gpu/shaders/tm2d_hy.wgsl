// Hy update: dHy/dt = (1/mu) * dEz/dx
// Hy[i,j] idx j*(nx+1)+i, update range i in [0,nx), j in [0,ny)
// Forward diff in x; FAR zero at i+1==nx.

@group(0) @binding(0) var<storage, read_write> hy      : array<f32>; // (nx+1)*ny
@group(0) @binding(1) var<storage, read>       ez      : array<f32>; // nx*ny
@group(0) @binding(2) var<storage, read_write> psi_hy_x: array<f32>; // (nx+1)*ny
// pml_h_x: [b_h[0..nx], c_h[0..nx], kappa_h[0..nx]]
@group(0) @binding(3) var<storage, read>       pml_h_x : array<f32>; // 3*nx

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

    let idx_hy = j * (nx + 1u) + i;

    // Forward diff Ez in x: FAR zero at i+1==nx
    var ez_right: f32 = 0.0;
    if (i + 1u < nx) {
        ez_right = ez[j * nx + i + 1u];
    }
    let dez_dx = (ez_right - ez[j * nx + i]) / dims.dx;

    let b_hx     = pml_h_x[i];
    let c_hx     = pml_h_x[nx + i];
    let kappa_hx = pml_h_x[2u * nx + i];

    let psi_new = b_hx * psi_hy_x[idx_hy] + c_hx * dez_dx;
    psi_hy_x[idx_hy] = psi_new;

    hy[idx_hy] = hy[idx_hy] + dims.dt / MU0 * (dez_dx / kappa_hx + psi_new);
}
