/// Geometry integration tests — GridSpec, Aabb2d, voxelize, SDF operations,
/// mesh quality metrics.
use oxiphoton::geometry::{
    aspect_ratio, df_difference, df_intersection, df_union, skewness, voxelize, Aabb2d, Circle2d,
    GridSpec1d, GridSpec2d, GridSpec3d, Rect2d,
};

// ── GridSpec1d ────────────────────────────────────────────────────────────────

#[test]
fn grid1d_uniform_100_cells() {
    let g = GridSpec1d::uniform(0.0, 10e-6, 100);
    assert_eq!(g.n_cells(), 100);
}

#[test]
fn grid1d_uniform_dx_exactly_100nm() {
    let g = GridSpec1d::uniform(0.0, 10e-6, 100);
    for &d in g.spacings().iter() {
        assert!((d - 100e-9).abs() < 1e-18, "dx should be 100 nm: {d:.3e}");
    }
}

#[test]
fn grid1d_uniform_first_last_edge() {
    let start = 1e-6_f64;
    let end = 5e-6_f64;
    let g = GridSpec1d::uniform(start, end, 40);
    assert!((g.edges[0] - start).abs() < 1e-20);
    assert!((g.edges[g.edges.len() - 1] - end).abs() < 1e-20);
}

#[test]
fn grid1d_nonuniform_from_edges() {
    let edges: Vec<f64> = (0..=5).map(|i| i as f64 * 200e-9).collect();
    let g = GridSpec1d::nonuniform(edges);
    assert_eq!(g.n_cells(), 5);
    assert!((g.centers[0] - 100e-9).abs() < 1e-15);
}

#[test]
fn grid1d_xcoord_centers_inside_domain() {
    let g = GridSpec1d::uniform(0.0, 1e-6, 20);
    for &c in &g.centers {
        assert!(c > 0.0 && c < 1e-6, "center out of domain: {c:.3e}");
    }
}

#[test]
fn grid1d_dx_min_max_equal_for_uniform() {
    let g = GridSpec1d::uniform(0.0, 1e-6, 10);
    let dmin = g.dx_min();
    let dmax = g.dx_max();
    assert!(
        (dmin - dmax).abs() < 1e-18,
        "Uniform grid: dx_min should equal dx_max"
    );
}

#[test]
fn grid1d_find_cell_clamped_below() {
    let g = GridSpec1d::uniform(1.0, 2.0, 10);
    assert_eq!(g.find_cell(0.5), 0);
}

#[test]
fn grid1d_find_cell_clamped_above() {
    let g = GridSpec1d::uniform(0.0, 1.0, 10);
    assert_eq!(g.find_cell(1.5), 9);
}

// ── GridSpec2d ────────────────────────────────────────────────────────────────

#[test]
fn grid2d_nx_ny_correct() {
    let g = GridSpec2d::uniform(0.0, 5e-6, 50, 0.0, 4e-6, 40);
    assert_eq!(g.nx(), 50);
    assert_eq!(g.ny(), 40);
}

#[test]
fn grid2d_yee_e_positions_count() {
    let g = GridSpec2d::uniform(0.0, 1e-6, 8, 0.0, 1e-6, 8);
    let pos = g.yee_e_positions();
    assert_eq!(pos.len(), 64);
}

// ── GridSpec3d fill_box_material ──────────────────────────────────────────────

#[test]
fn grid3d_fill_box_material_center_labeled() {
    let g = GridSpec3d::uniform(0.0, 10.0, 10, 0.0, 10.0, 10, 0.0, 10.0, 10);
    let mut map = vec![0_usize; g.n_total()];
    g.fill_box_material(&mut map, 3.0, 7.0, 3.0, 7.0, 3.0, 7.0, 2);
    let center = g.idx(5, 5, 5);
    assert_eq!(map[center], 2, "center should be labeled 2");
}

#[test]
fn grid3d_fill_box_material_boundary_unlabeled() {
    let g = GridSpec3d::uniform(0.0, 10.0, 10, 0.0, 10.0, 10, 0.0, 10.0, 10);
    let mut map = vec![0_usize; g.n_total()];
    g.fill_box_material(&mut map, 3.0, 7.0, 3.0, 7.0, 3.0, 7.0, 2);
    assert_eq!(map[g.idx(0, 0, 0)], 0, "corner should be unlabeled");
    assert_eq!(map[g.idx(9, 9, 9)], 0, "far corner should be unlabeled");
}

// ── GridSpec3d basic ──────────────────────────────────────────────────────────

#[test]
fn grid3d_n_total() {
    let g = GridSpec3d::uniform(0.0, 1.0, 5, 0.0, 1.0, 6, 0.0, 1.0, 7);
    assert_eq!(g.n_total(), 210);
}

#[test]
fn grid3d_dimensions_reported_correctly() {
    let g = GridSpec3d::uniform(0.0, 1.0, 3, 0.0, 1.0, 4, 0.0, 1.0, 5);
    assert_eq!(g.nx(), 3);
    assert_eq!(g.ny(), 4);
    assert_eq!(g.nz(), 5);
}

// ── Aabb2d ────────────────────────────────────────────────────────────────────

#[test]
fn aabb2d_contains_interior_point() {
    let box_ = Aabb2d::new([0.0, 0.0], [2.0, 2.0]);
    assert!(box_.contains([1.0, 1.0]));
}

#[test]
fn aabb2d_does_not_contain_exterior_point() {
    let box_ = Aabb2d::new([0.0, 0.0], [2.0, 2.0]);
    assert!(!box_.contains([3.0, 1.0]));
}

#[test]
fn aabb2d_contains_boundary_point() {
    let box_ = Aabb2d::new([0.0, 0.0], [2.0, 2.0]);
    assert!(box_.contains([0.0, 0.0]));
    assert!(box_.contains([2.0, 2.0]));
}

#[test]
fn aabb2d_intersects_overlapping() {
    let a = Aabb2d::new([0.0, 0.0], [2.0, 2.0]);
    let b = Aabb2d::new([1.0, 1.0], [3.0, 3.0]);
    assert!(a.intersects(&b));
}

#[test]
fn aabb2d_does_not_intersect_disjoint() {
    let a = Aabb2d::new([0.0, 0.0], [1.0, 1.0]);
    let b = Aabb2d::new([2.0, 2.0], [3.0, 3.0]);
    assert!(!a.intersects(&b));
}

// ── voxelize ──────────────────────────────────────────────────────────────────

#[test]
fn voxelize_circle_center_is_solid() {
    let nx = 20_usize;
    let ny = 20_usize;
    let size = 2000e-9_f64; // 2 μm domain
    let cx = size / 2.0;
    let cy = size / 2.0;
    let r = 400e-9_f64;
    let circle = Circle2d::new(cx, cy, r);
    let bounds = [[0.0, 0.0], [size, size]];
    let grid = voxelize(&circle, nx, ny, bounds);
    // Center cell (10, 10) → row-major index 10*20 + 10 = 210
    let center_idx = 10 * nx + 10;
    assert!(
        grid[center_idx],
        "Circle center cell should be inside the circle"
    );
}

#[test]
fn voxelize_rectangle_corner_outside() {
    let nx = 20_usize;
    let ny = 20_usize;
    let size = 2000e-9_f64;
    // Rectangle occupying only the center quarter
    let margin = size * 0.3;
    let rect = Rect2d::new(margin, size - margin, margin, size - margin);
    let bounds = [[0.0, 0.0], [size, size]];
    let grid = voxelize(&rect, nx, ny, bounds);
    // Cell (0, 0) at bottom-left corner should be outside
    assert!(
        !grid[0],
        "Rectangle corner cell should be outside the rectangle"
    );
}

#[test]
fn voxelize_full_rect_all_inside() {
    let nx = 10_usize;
    let ny = 10_usize;
    // Rectangle covering entire domain
    let rect = Rect2d::new(0.0, 1.0, 0.0, 1.0);
    let bounds = [[0.0, 0.0], [1.0, 1.0]];
    let grid = voxelize(&rect, nx, ny, bounds);
    assert!(
        grid.iter().all(|&v| v),
        "Full domain rectangle: all cells should be inside"
    );
}

// ── df_union / df_intersection / df_difference ────────────────────────────────

#[test]
fn df_union_inside_either_is_negative() {
    let d1 = -1.0_f64; // inside A
    let d2 = 2.0_f64; // outside B
    let u = df_union(d1, d2);
    assert!(u < 0.0, "Union: inside A → inside union: {u}");
}

#[test]
fn df_union_outside_both_is_positive() {
    let u = df_union(1.0, 2.0);
    assert!(u > 0.0, "Union: outside both → outside union: {u}");
}

#[test]
fn df_intersection_inside_both_is_negative() {
    let inter = df_intersection(-1.0, -2.0);
    assert!(
        inter < 0.0,
        "Intersection: inside both → inside intersection: {inter}"
    );
}

#[test]
fn df_intersection_outside_either_is_positive() {
    let inter = df_intersection(1.0, -2.0);
    assert!(
        inter > 0.0,
        "Intersection: outside A → outside intersection: {inter}"
    );
}

#[test]
fn df_difference_inside_a_inside_b_is_positive() {
    // A - B: inside A (da < 0) and inside B (db < 0) → removed by B → outside
    let diff = df_difference(-1.0, -2.0);
    assert!(diff > 0.0, "A\\B: inside both → outside difference: {diff}");
}

#[test]
fn df_difference_inside_a_outside_b_is_negative() {
    // Inside A (da < 0), outside B (db > 0) → inside difference
    let diff = df_difference(-2.0, 1.0);
    assert!(
        diff < 0.0,
        "A\\B: inside A, outside B → inside difference: {diff}"
    );
}

// ── aspect_ratio ──────────────────────────────────────────────────────────────

#[test]
fn aspect_ratio_equilateral_triangle_about_1_15() {
    let v0 = [0.0_f64, 0.0_f64];
    let v1 = [1.0_f64, 0.0_f64];
    let v2 = [0.5_f64, 3.0_f64.sqrt() / 2.0_f64];
    let ar = aspect_ratio(v0, v1, v2);
    // For equilateral: longest_edge = 1, shortest_altitude = sqrt(3)/2
    // ar = 1 / (sqrt(3)/2) ≈ 1.155
    assert!(
        ar > 0.9 && ar < 2.0,
        "Equilateral aspect ratio should be ~1.15: {ar}"
    );
}

#[test]
fn aspect_ratio_flat_triangle_much_higher() {
    let v0 = [0.0_f64, 0.0_f64];
    let v1 = [10.0_f64, 0.0_f64];
    let v2 = [0.0_f64, 0.1_f64]; // very flat
    let ar_flat = aspect_ratio(v0, v1, v2);
    let eq0 = [0.0_f64, 0.0_f64];
    let eq1 = [1.0_f64, 0.0_f64];
    let eq2 = [0.5_f64, 3.0_f64.sqrt() / 2.0_f64];
    let ar_eq = aspect_ratio(eq0, eq1, eq2);
    assert!(
        ar_flat > ar_eq,
        "Flat triangle should have worse aspect ratio: {ar_flat} vs {ar_eq}"
    );
}

// ── skewness ──────────────────────────────────────────────────────────────────

#[test]
fn skewness_equilateral_is_zero() {
    let v0 = [0.0_f64, 0.0_f64];
    let v1 = [1.0_f64, 0.0_f64];
    let v2 = [0.5_f64, 3.0_f64.sqrt() / 2.0_f64];
    let s = skewness(v0, v1, v2);
    assert!(
        s < 1e-10,
        "Equilateral triangle should have zero skewness: {s}"
    );
}

#[test]
fn skewness_near_degenerate_is_near_one() {
    let v0 = [0.0_f64, 0.0_f64];
    let v1 = [1.0_f64, 0.0_f64];
    let v2 = [0.5_f64, 1e-6_f64]; // almost collinear
    let s = skewness(v0, v1, v2);
    assert!(
        s > 0.9,
        "Nearly degenerate triangle should have skewness near 1: {s}"
    );
}

#[test]
fn skewness_always_in_range_0_1() {
    let v0 = [0.0_f64, 0.0_f64];
    let v1 = [1.0_f64, 0.0_f64];
    let v2 = [0.0_f64, 1.0_f64]; // right triangle
    let s = skewness(v0, v1, v2);
    assert!((0.0..=1.0).contains(&s), "Skewness out of [0,1]: {s}");
}

#[test]
fn skewness_right_triangle_between_0_and_1() {
    let v0 = [0.0_f64, 0.0_f64];
    let v1 = [1.0_f64, 0.0_f64];
    let v2 = [0.0_f64, 1.0_f64];
    let s = skewness(v0, v1, v2);
    // Right triangle has one 90° angle (> 60°) → skewness > 0
    assert!(
        s > 0.0 && s < 0.5,
        "Right triangle skewness should be in (0, 0.5): {s}"
    );
}
