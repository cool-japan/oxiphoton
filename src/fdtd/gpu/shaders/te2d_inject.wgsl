// Additive Hz point-source injection.
// Dispatched as (1, 1, 1) once per step, before the Hz update pass.

@group(0) @binding(0) var<storage, read_write> hz: array<f32>;

struct SrcUniform {
    flat_idx : u32,
    _p0      : u32,
    _p1      : u32,
    val      : f32,
}
@group(0) @binding(1) var<uniform> src: SrcUniform;

@compute @workgroup_size(1)
fn main() {
    hz[src.flat_idx] = hz[src.flat_idx] + src.val;
}
