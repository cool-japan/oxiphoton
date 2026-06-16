// Additive Ez point-source injection (1×1 dispatch, before Hx update).

@group(0) @binding(0) var<storage, read_write> ez: array<f32>;

struct SrcUniform {
    flat_idx : u32,
    _p0      : u32,
    _p1      : u32,
    val      : f32,
}
@group(0) @binding(1) var<uniform> src: SrcUniform;

@compute @workgroup_size(1)
fn main() {
    ez[src.flat_idx] = ez[src.flat_idx] + src.val;
}
