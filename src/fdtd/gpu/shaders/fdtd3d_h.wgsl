// 3D H-field update with CPML. All 6 fields same size n=nx*ny*nz.
// idx = (k*ny + j)*nx + i. Loop range: i,j,k in 0..dim-1 (guard: exit at dim-1).
// Forward differences: neighbor at +1 is always valid because loop stops before last.
// H uses − coeff_curl.
// PML (stride-6): pml_x[6*i + {0=b_e,1=c_e,2=k_e,3=b_h,4=c_h,5=k_h}]
// PSI (stride-6 per cell): psi_h[6*idx + {0=hx_y,1=hx_z,2=hy_x,3=hy_z,4=hz_x,5=hz_y}]

@group(0) @binding(0) var<storage, read_write> buf_h   : array<vec4<f32>>; // n vec4s
@group(0) @binding(1) var<storage, read>       buf_e   : array<vec4<f32>>; // n vec4s
@group(0) @binding(2) var<storage, read_write> psi_h   : array<f32>;       // 6n f32
@group(0) @binding(3) var<storage, read>       buf_mat : array<vec4<f32>>; // n: (eps,mu,se,sm)
@group(0) @binding(4) var<storage, read>       pml_x   : array<f32>;       // 6*nx
@group(0) @binding(5) var<storage, read>       pml_y   : array<f32>;       // 6*ny
@group(0) @binding(6) var<storage, read>       pml_z   : array<f32>;       // 6*nz

struct Dims3d { nx: u32, ny: u32, nz: u32, dx: f32, dy: f32, dz: f32, dt: f32, _p0: u32 }
@group(0) @binding(7) var<uniform> dims: Dims3d;

const MU0: f32 = 1.2566370614359173e-6;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;
    let nx = dims.nx;
    let ny = dims.ny;
    let nz = dims.nz;

    // H loop runs 0..n-1 on each axis; the last plane stays at 0 (PEC-like)
    if (i >= nx - 1u || j >= ny - 1u || k >= nz - 1u) { return; }

    let idx      = (k * ny + j) * nx + i;
    let idx_pi   = idx + 1u;         // i+1
    let idx_pj   = idx + nx;         // j+1
    let idx_pk   = idx + nx * ny;    // k+1

    let e0  = buf_e[idx];
    let e_i = buf_e[idx_pi];
    let e_j = buf_e[idx_pj];
    let e_k = buf_e[idx_pk];

    // forward differences: (neighbor - self) / spacing
    // e0.x=ex, e0.y=ey, e0.z=ez
    let dez_dy = (e_j.z - e0.z) / dims.dy;
    let dey_dz = (e_k.y - e0.y) / dims.dz;
    let dex_dz = (e_k.x - e0.x) / dims.dz;
    let dez_dx = (e_i.z - e0.z) / dims.dx;
    let dey_dx = (e_i.y - e0.y) / dims.dx;
    let dex_dy = (e_j.x - e0.x) / dims.dy;

    // PML coefficients (derivative axis determines which axis's PML to use)
    let b_hx = pml_x[6u * i + 3u];  let c_hx = pml_x[6u * i + 4u];  let k_hx = pml_x[6u * i + 5u];
    let b_hy = pml_y[6u * j + 3u];  let c_hy = pml_y[6u * j + 4u];  let k_hy = pml_y[6u * j + 5u];
    let b_hz = pml_z[6u * k + 3u];  let c_hz = pml_z[6u * k + 4u];  let k_hz = pml_z[6u * k + 5u];

    // ψ (stride-6 per cell): store new value, then use it
    let pb = 6u * idx;
    let phx_y_new = b_hy * psi_h[pb + 0u] + c_hy * dez_dy;
    let phx_z_new = b_hz * psi_h[pb + 1u] + c_hz * dey_dz;
    let phy_x_new = b_hx * psi_h[pb + 2u] + c_hx * dez_dx;
    let phy_z_new = b_hz * psi_h[pb + 3u] + c_hz * dex_dz;
    let phz_x_new = b_hx * psi_h[pb + 4u] + c_hx * dey_dx;
    let phz_y_new = b_hy * psi_h[pb + 5u] + c_hy * dex_dy;
    psi_h[pb + 0u] = phx_y_new;
    psi_h[pb + 1u] = phx_z_new;
    psi_h[pb + 2u] = phy_x_new;
    psi_h[pb + 3u] = phy_z_new;
    psi_h[pb + 4u] = phz_x_new;
    psi_h[pb + 5u] = phz_y_new;

    // Lossy magnetic coefficients
    let mu_r  = buf_mat[idx].y;
    let sig_m = buf_mat[idx].w;
    let mu    = MU0 * mu_r;
    let half_sig_dt_mu = sig_m * dims.dt / (2.0 * mu);
    let den   = 1.0 + half_sig_dt_mu;
    let coeff_h    = (1.0 - half_sig_dt_mu) / den;
    let coeff_curl = (dims.dt / mu) / den;

    var h = buf_h[idx];
    h.x = coeff_h * h.x - coeff_curl * (dez_dy / k_hy + phx_y_new - dey_dz / k_hz - phx_z_new);
    h.y = coeff_h * h.y - coeff_curl * (dex_dz / k_hz + phy_z_new - dez_dx / k_hx - phy_x_new);
    h.z = coeff_h * h.z - coeff_curl * (dey_dx / k_hx + phz_x_new - dex_dy / k_hy - phz_y_new);
    buf_h[idx] = h;
}
