// 3D E-field update with CPML. All 6 fields same size n=nx*ny*nz.
// idx = (k*ny + j)*nx + i. Loop range: i,j,k in 1..dim-1 (guard: exit at 0 or dim-1).
// Backward differences: neighbor at -1 is always valid because loop starts at 1.
// E uses + coeff_curl.
// PML (stride-6): pml_x[6*i + {0=b_e,1=c_e,2=k_e,3=b_h,4=c_h,5=k_h}]
// PSI (stride-6 per cell): psi_e[6*idx + {0=ex_y,1=ex_z,2=ey_x,3=ey_z,4=ez_x,5=ez_y}]

@group(0) @binding(0) var<storage, read_write> buf_e   : array<vec4<f32>>; // n vec4s
@group(0) @binding(1) var<storage, read>       buf_h   : array<vec4<f32>>; // n vec4s
@group(0) @binding(2) var<storage, read_write> psi_e   : array<f32>;       // 6n f32
@group(0) @binding(3) var<storage, read>       buf_mat : array<vec4<f32>>; // n: (eps,mu,se,sm)
@group(0) @binding(4) var<storage, read>       pml_x   : array<f32>;       // 6*nx
@group(0) @binding(5) var<storage, read>       pml_y   : array<f32>;       // 6*ny
@group(0) @binding(6) var<storage, read>       pml_z   : array<f32>;       // 6*nz

struct Dims3d { nx: u32, ny: u32, nz: u32, dx: f32, dy: f32, dz: f32, dt: f32, _p0: u32 }
@group(0) @binding(7) var<uniform> dims: Dims3d;

const EPS0: f32 = 8.854187817620389e-12;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;
    let nx = dims.nx;
    let ny = dims.ny;
    let nz = dims.nz;

    // E loop runs 1..n-1 on each axis
    if (i < 1u || j < 1u || k < 1u || i >= nx - 1u || j >= ny - 1u || k >= nz - 1u) { return; }

    let idx      = (k * ny + j) * nx + i;
    let idx_mi   = idx - 1u;         // i-1
    let idx_mj   = idx - nx;         // j-1
    let idx_mk   = idx - nx * ny;    // k-1

    let h0  = buf_h[idx];
    let h_i = buf_h[idx_mi];   // i-1
    let h_j = buf_h[idx_mj];   // j-1
    let h_k = buf_h[idx_mk];   // k-1

    // backward differences: (self - neighbor_at_minus1) / spacing
    // h0.x=hx, h0.y=hy, h0.z=hz
    let dhz_dy = (h0.z - h_j.z) / dims.dy;
    let dhy_dz = (h0.y - h_k.y) / dims.dz;
    let dhx_dz = (h0.x - h_k.x) / dims.dz;
    let dhz_dx = (h0.z - h_i.z) / dims.dx;
    let dhy_dx = (h0.y - h_i.y) / dims.dx;
    let dhx_dy = (h0.x - h_j.x) / dims.dy;

    // PML coefficients (use b_e/c_e/kappa_e, offsets 0,1,2)
    let b_ex = pml_x[6u * i + 0u];  let c_ex = pml_x[6u * i + 1u];  let k_ex = pml_x[6u * i + 2u];
    let b_ey = pml_y[6u * j + 0u];  let c_ey = pml_y[6u * j + 1u];  let k_ey = pml_y[6u * j + 2u];
    let b_ez = pml_z[6u * k + 0u];  let c_ez = pml_z[6u * k + 1u];  let k_ez = pml_z[6u * k + 2u];

    // ψ (stride-6 per cell): store new value, then use it
    // layout: [psi_ex_y=0, psi_ex_z=1, psi_ey_x=2, psi_ey_z=3, psi_ez_x=4, psi_ez_y=5]
    let pb = 6u * idx;
    let pex_y_new = b_ey * psi_e[pb + 0u] + c_ey * dhz_dy;
    let pex_z_new = b_ez * psi_e[pb + 1u] + c_ez * dhy_dz;
    let pey_x_new = b_ex * psi_e[pb + 2u] + c_ex * dhz_dx;
    let pey_z_new = b_ez * psi_e[pb + 3u] + c_ez * dhx_dz;
    let pez_x_new = b_ex * psi_e[pb + 4u] + c_ex * dhy_dx;
    let pez_y_new = b_ey * psi_e[pb + 5u] + c_ey * dhx_dy;
    psi_e[pb + 0u] = pex_y_new;
    psi_e[pb + 1u] = pex_z_new;
    psi_e[pb + 2u] = pey_x_new;
    psi_e[pb + 3u] = pey_z_new;
    psi_e[pb + 4u] = pez_x_new;
    psi_e[pb + 5u] = pez_y_new;

    // Lossy electric coefficients
    let eps_r  = buf_mat[idx].x;
    let sig_e  = buf_mat[idx].z;
    let eps    = EPS0 * eps_r;
    let half_sig_dt_eps = sig_e * dims.dt / (2.0 * eps);
    let den    = 1.0 + half_sig_dt_eps;
    let coeff_e    = (1.0 - half_sig_dt_eps) / den;
    let coeff_curl = (dims.dt / eps) / den;

    var e = buf_e[idx];
    e.x = coeff_e * e.x + coeff_curl * (dhz_dy / k_ey + pex_y_new - dhy_dz / k_ez - pex_z_new);
    e.y = coeff_e * e.y + coeff_curl * (dhx_dz / k_ez + pey_z_new - dhz_dx / k_ex - pey_x_new);
    e.z = coeff_e * e.z + coeff_curl * (dhy_dx / k_ex + pez_x_new - dhx_dy / k_ey - pez_y_new);
    buf_e[idx] = e;
}
