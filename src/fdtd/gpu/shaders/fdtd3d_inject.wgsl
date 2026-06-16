// Inject a source into Ez (component .z of the packed E vec4).
@group(0) @binding(0) var<storage, read_write> e_field: array<vec4<f32>>;

struct SrcUniform { flat_idx: u32, _p0: u32, _p1: u32, val: f32 }
@group(0) @binding(1) var<uniform> src: SrcUniform;

@compute @workgroup_size(1)
fn main() {
    var e = e_field[src.flat_idx];
    e.z = e.z + src.val;
    e_field[src.flat_idx] = e;
}
