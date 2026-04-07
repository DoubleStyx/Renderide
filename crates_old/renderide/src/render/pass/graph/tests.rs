//! Unit tests for graph build errors, execution metadata, and pass helpers.

use crate::render::pass::clustered_light::ClusteredLightPass;
use crate::render::pass::composite::CompositePass;
use crate::render::pass::error::RenderPassError;
use crate::render::pass::mesh_pass::MeshRenderPass;
use crate::render::pass::overlay_pass::OverlayRenderPass;
use crate::render::pass::rtao_blur::RtaoBlurPass;
use crate::render::pass::rtao_compute::RtaoComputePass;

use super::{
    GraphBuildError, GraphBuilder, PassResources, RenderPass, RenderPassContext, ResourceSlot,
    build_main_render_graph,
};

use super::views::texture_read_target_uses;

struct TestPass {
    name: String,
}

impl RenderPass for TestPass {
    fn name(&self) -> &str {
        &self.name
    }
    fn execute(&mut self, _ctx: &mut RenderPassContext) -> Result<(), RenderPassError> {
        Ok(())
    }
}

#[test]
fn graph_builder_add_pass_if_skips_when_false() {
    let mut builder = GraphBuilder::new();
    let _a = builder.add_pass(Box::new(TestPass {
        name: "a".to_string(),
    }));
    let opt = builder.add_pass_if(
        false,
        Box::new(TestPass {
            name: "b".to_string(),
        }),
    );
    assert!(opt.is_none());
    let graph = builder.build().expect("single pass");
    assert_eq!(graph.pass_names(), &["a"]);
}

#[test]
fn graph_builder_add_pass_if_adds_when_true() {
    let mut builder = GraphBuilder::new();
    let a = builder.add_pass(Box::new(TestPass {
        name: "a".to_string(),
    }));
    let b = builder
        .add_pass_if(
            true,
            Box::new(TestPass {
                name: "b".to_string(),
            }),
        )
        .expect("condition true");
    builder.add_edge(a, b);
    let graph = builder.build().expect("chain");
    assert_eq!(graph.pass_names(), &["a", "b"]);
}

#[test]
fn main_render_graph_no_rtao_mesh_writes_color_depth() {
    let graph = build_main_render_graph(false, false, false).expect("graph");
    assert_eq!(graph.pass_names(), &["clustered_light", "mesh", "overlay"]);
    let (comp, overlay) = graph.special_pass_ids();
    assert!(comp.is_none());
    assert!(overlay.is_some());
    let res = graph.pass_resources();
    assert!(res[1].writes.contains(&ResourceSlot::Color));
    assert!(res[1].writes.contains(&ResourceSlot::Depth));
    assert!(!res[1].writes.contains(&ResourceSlot::Position));
}

#[test]
fn main_render_graph_fullscreen_filter_hook_inserts_placeholder() {
    let graph = build_main_render_graph(false, false, true).expect("graph");
    assert_eq!(
        graph.pass_names(),
        &[
            "clustered_light",
            "mesh",
            "fullscreen_filter_placeholder",
            "overlay",
        ]
    );
}

#[test]
fn main_render_graph_rtao_includes_compute_blur_composite() {
    let graph = build_main_render_graph(true, false, false).expect("graph");
    let names = graph.pass_names();
    assert_eq!(names.len(), 6);
    assert!(names.iter().any(|n| n == "rtao_compute"));
    assert!(names.iter().any(|n| n == "rtao_blur"));
    assert!(names.iter().any(|n| n == "composite"));
    let (comp, overlay) = graph.special_pass_ids();
    assert!(comp.is_some());
    assert!(overlay.is_some());
    let res = graph.pass_resources();
    assert!(res[1].writes.contains(&ResourceSlot::Position));
}

#[test]
fn main_render_graph_rt_shadow_compute_between_mesh_and_rtao() {
    let graph = build_main_render_graph(true, true, false).expect("graph");
    let names = graph.pass_names();
    assert_eq!(names.len(), 7);
    let mesh_i = names.iter().position(|n| *n == "mesh").expect("mesh");
    let shadow_i = names
        .iter()
        .position(|n| *n == "rt_shadow_compute")
        .expect("rt_shadow_compute");
    let rtao_i = names
        .iter()
        .position(|n| *n == "rtao_compute")
        .expect("rtao_compute");
    assert!(mesh_i < shadow_i);
    assert!(shadow_i < rtao_i);
}

/// [`RtaoBlurPass`] samples the normal G-buffer; its [`RenderPass::resources`] must declare
/// [`ResourceSlot::Normal`] or per-pass [`crate::render::pass::graph::RenderTargetViews`] omit `mrt_normal_view` and blur
/// never runs (breaking composite AO).
#[test]
fn rtao_blur_pass_declares_normal_read_for_slot_map() {
    let pass = RtaoBlurPass::new();
    let r = pass.resources();
    assert!(
        r.reads.contains(&ResourceSlot::Normal),
        "blur execute() needs mrt_normal_view from slot map"
    );
}

#[test]
fn graph_builder_subgraph_wraps_main_view_flat_graph() {
    let inner = build_main_render_graph(false, false, false).expect("inner");
    let mut builder = GraphBuilder::new();
    let _main = builder.add_subgraph("main_view", inner);
    let graph = builder.build().expect("single subgraph");
    let names = graph.pass_names();
    assert!(names.iter().any(|n| n == "main_view/clustered_light"));
    assert!(names.iter().any(|n| n == "main_view/mesh"));
    assert!(names.iter().any(|n| n == "main_view/overlay"));
}

#[test]
fn graph_builder_edge_pass_to_subgraph() {
    let inner = build_main_render_graph(false, false, false).expect("inner");
    let mut builder = GraphBuilder::new();
    let pre = builder.add_pass(Box::new(TestPass {
        name: "pre".to_string(),
    }));
    let sg = builder.add_subgraph("main", inner);
    builder.add_edge(pre, sg);
    let graph = builder.build().expect("dag");
    let names = graph.pass_names();
    assert_eq!(names[0], "pre");
    assert!(names[1].starts_with("main/"));
}

/// A valid acyclic graph topologically sorts to the unique order `a → b → c`.
#[test]
fn graph_builder_valid_graph_produces_expected_pass_order() {
    let mut builder = GraphBuilder::new();
    let a = builder.add_pass(Box::new(TestPass {
        name: "a".to_string(),
    }));
    let b = builder.add_pass(Box::new(TestPass {
        name: "b".to_string(),
    }));
    let c = builder.add_pass(Box::new(TestPass {
        name: "c".to_string(),
    }));
    builder.add_edge(a, b);
    builder.add_edge(b, c);
    let graph = builder.build().expect("linear chain has no cycle");
    assert_eq!(graph.pass_names(), &["a", "b", "c"]);
}

/// An edge cycle makes topological sort impossible; build returns [`GraphBuildError::CycleDetected`].
#[test]
fn graph_builder_cycle_returns_cycle_detected() {
    let mut builder = GraphBuilder::new();
    let a = builder.add_pass(Box::new(TestPass {
        name: "a".to_string(),
    }));
    let b = builder.add_pass(Box::new(TestPass {
        name: "b".to_string(),
    }));
    builder.add_edge(a, b);
    builder.add_edge(b, a);
    let result = builder.build();
    assert!(matches!(result, Err(GraphBuildError::CycleDetected)));
}

#[test]
fn graph_builder_dag_branching() {
    let mut builder = GraphBuilder::new();
    let a = builder.add_pass(Box::new(TestPass {
        name: "a".to_string(),
    }));
    let b = builder.add_pass(Box::new(TestPass {
        name: "b".to_string(),
    }));
    let c = builder.add_pass(Box::new(TestPass {
        name: "c".to_string(),
    }));
    let d = builder.add_pass(Box::new(TestPass {
        name: "d".to_string(),
    }));
    builder.add_edge(a, b);
    builder.add_edge(a, c);
    builder.add_edge(b, d);
    builder.add_edge(c, d);
    let graph = builder.build().expect("DAG has no cycle");
    let names = graph.pass_names();
    assert_eq!(names.len(), 4);
    assert_eq!(names[0], "a");
    assert_eq!(names[3], "d");
    assert!(names.contains(&"b".to_string()));
    assert!(names.contains(&"c".to_string()));
}

#[test]
fn graph_builder_special_passes_recorded() {
    let mut builder = GraphBuilder::new();
    let _clustered = builder.add_pass(Box::new(ClusteredLightPass::new()));
    let _mesh = builder.add_pass(Box::new(MeshRenderPass::with_rtao_mrt_graph(true)));
    let _rtao = builder.add_pass(Box::new(RtaoComputePass::new()));
    let _rtao_blur = builder.add_pass(Box::new(RtaoBlurPass::new()));
    let composite = builder.add_pass(Box::new(CompositePass::new()));
    let overlay = builder.add_pass(Box::new(OverlayRenderPass::new()));
    builder.add_edge(_clustered, _mesh);
    builder.add_edge(_mesh, _rtao);
    builder.add_edge(_rtao, _rtao_blur);
    builder.add_edge(_rtao_blur, composite);
    builder.add_edge(composite, overlay);
    let graph = builder
        .build_with_special_passes(Some(composite), Some(overlay))
        .expect("graph has no cycle");
    let (comp_id, overlay_id) = graph.special_pass_ids();
    assert!(comp_id.is_some(), "composite PassId should be recorded");
    assert!(overlay_id.is_some(), "overlay PassId should be recorded");
    assert_eq!(comp_id, Some(composite));
    assert_eq!(overlay_id, Some(overlay));
}

#[test]
fn graph_builder_stores_pass_resources() {
    let mut builder = GraphBuilder::new();
    let _clustered = builder.add_pass(Box::new(ClusteredLightPass::new()));
    let _mesh = builder.add_pass(Box::new(MeshRenderPass::with_rtao_mrt_graph(true)));
    let _rtao = builder.add_pass(Box::new(RtaoComputePass::new()));
    let _rtao_blur = builder.add_pass(Box::new(RtaoBlurPass::new()));
    let composite = builder.add_pass(Box::new(CompositePass::new()));
    let overlay = builder.add_pass(Box::new(OverlayRenderPass::new()));
    builder.add_edge(_clustered, _mesh);
    builder.add_edge(_mesh, _rtao);
    builder.add_edge(_rtao, _rtao_blur);
    builder.add_edge(_rtao_blur, composite);
    builder.add_edge(composite, overlay);
    let graph = builder
        .build_with_special_passes(Some(composite), Some(overlay))
        .expect("graph has no cycle");
    let resources = graph.pass_resources();
    assert_eq!(resources.len(), 6);
    assert!(resources[0].writes.contains(&ResourceSlot::ClusterBuffers));
    assert!(resources[1].reads.contains(&ResourceSlot::ClusterBuffers));
    assert!(resources[2].writes.contains(&ResourceSlot::AoRaw));
    assert!(resources[3].writes.contains(&ResourceSlot::Ao));
    assert!(resources[4].writes.contains(&ResourceSlot::Surface));
    assert!(resources[5].writes.contains(&ResourceSlot::Surface));
}

/// A pass that reads a slot no predecessor writes fails with [`GraphBuildError::MissingDependency`].
#[test]
fn graph_builder_missing_read_returns_missing_dependency() {
    let mut builder = GraphBuilder::new();
    let a = builder.add_pass(Box::new(TestPass {
        name: "a".to_string(),
    }));
    let composite = builder.add_pass(Box::new(CompositePass::new()));
    builder.add_edge(a, composite);
    let result = builder.build();
    assert!(
        matches!(
            result,
            Err(GraphBuildError::MissingDependency {
                slot: ResourceSlot::Color,
                ..
            })
        ),
        "composite reads Color but no earlier pass produces it"
    );
}

#[test]
fn texture_read_target_uses_depth_overlay_vs_blur() {
    let overlay = PassResources {
        reads: vec![ResourceSlot::Depth],
        writes: vec![ResourceSlot::Surface],
    };
    assert_eq!(
        texture_read_target_uses(ResourceSlot::Depth, &overlay),
        Some(wgpu::TextureUses::DEPTH_STENCIL_WRITE)
    );
    let blur = PassResources {
        reads: vec![
            ResourceSlot::AoRaw,
            ResourceSlot::Depth,
            ResourceSlot::Normal,
        ],
        writes: vec![ResourceSlot::Ao],
    };
    assert_eq!(
        texture_read_target_uses(ResourceSlot::Depth, &blur),
        Some(wgpu::TextureUses::DEPTH_STENCIL_READ)
    );
}
