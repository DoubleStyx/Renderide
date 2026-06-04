//! Tracy plots for CPU-side world-mesh forward preparation.

use super::tracy_plot::tracy_plot;

/// Records the size of one prepared world-mesh forward view.
pub fn plot_world_mesh_prepare(draws: usize, material_packets: usize, primary_groups: usize) {
    tracy_plot!("world_mesh_prepare::draws", draws as f64);
    tracy_plot!(
        "world_mesh_prepare::material_packets",
        material_packets as f64
    );
    tracy_plot!("world_mesh_prepare::primary_groups", primary_groups as f64);
}
