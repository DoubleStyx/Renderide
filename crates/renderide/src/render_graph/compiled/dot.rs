//! Graphviz DOT export of a [`super::CompiledRenderGraph`].
//!
//! Mirrors Filament's `FrameGraph::export_graphviz` so the same mental model applies: the dump is
//! a snapshot of the retained (post-cull, post-topo-sort) graph with resources and passes on
//! separate node shapes and edges colored by access type. Two clusters separate
//! [`super::super::pass::PassPhase::FrameGlobal`] and [`super::super::pass::PassPhase::PerView`]
//! passes.
//!
//! This is not a hot-path facility — it is emitted on demand from the diagnostics HUD or a
//! developer-triggered dump. `to_dot` allocates a single [`String`]; callers decide what to do
//! with it (write to stdout, file, or the imgui overlay).

use std::fmt::Write as _;

use super::super::pass::PassPhase;
use super::super::resources::{
    AccessKind, BufferAccess, BufferResourceHandle, ResourceAccess, ResourceHandle, TextureAccess,
    TextureResourceHandle,
};
use super::{CompiledPassInfo, CompiledRenderGraph};

/// Output format variants for [`CompiledRenderGraph::to_dot_with_format`].
///
/// The two variants differ only in how much detail is printed per node. A future consumer wanting
/// structured JSON can add a variant here without breaking existing callers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DotFormat {
    /// Pass names, resource labels, and access-colored edges. Good default for developer use.
    #[default]
    Standard,
    /// Standard output plus the full `{:?}` debug dump of each access appended as an edge label.
    /// Verbose, but useful when diagnosing a specific pass's declared accesses.
    Debug,
}

impl CompiledRenderGraph {
    /// Emits a Graphviz DOT representation of the retained graph.
    ///
    /// Pass through `dot -Tpng` (or the online Graphviz viewer) to render. Rebuilt each frame the
    /// caller asks for it; the string is deterministic for a given compiled graph instance.
    pub fn to_dot(&self) -> String {
        self.to_dot_with_format(DotFormat::Standard)
    }

    /// Same as [`Self::to_dot`] with an explicit [`DotFormat`].
    pub fn to_dot_with_format(&self, format: DotFormat) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "digraph RenderGraph {{");
        let _ = writeln!(out, "  rankdir=LR;");
        let _ = writeln!(
            out,
            "  node [shape=box style=rounded fontname=\"Helvetica\"];"
        );
        let _ = writeln!(out, "  edge [fontname=\"Helvetica\" fontsize=9];");

        self.emit_resource_nodes(&mut out);
        self.emit_pass_clusters(&mut out);
        self.emit_access_edges(&mut out, format);

        let _ = writeln!(out, "}}");
        out
    }

    fn emit_resource_nodes(&self, out: &mut String) {
        let _ = writeln!(out, "  // transient textures");
        for (idx, compiled) in self.transient_textures.iter().enumerate() {
            let label = escape_label(compiled.desc.label);
            let _ = writeln!(
                out,
                "  \"tex_t{idx}\" [label=\"{label}\\n(transient tex)\" shape=oval];"
            );
        }
        let _ = writeln!(out, "  // imported textures");
        for (idx, decl) in self.imported_textures.iter().enumerate() {
            let label = escape_label(decl.label);
            let _ = writeln!(
                out,
                "  \"tex_i{idx}\" [label=\"{label}\\n(imported tex)\" shape=oval style=\"filled\" fillcolor=\"#e8e8ff\"];"
            );
        }
        let _ = writeln!(out, "  // texture subresources");
        for (idx, desc) in self.subresources.iter().enumerate() {
            let label = escape_label(desc.label);
            let _ = writeln!(
                out,
                "  \"tex_sub{idx}\" [label=\"{label}\\n(subresource)\" shape=oval style=\"dashed\"];"
            );
        }
        let _ = writeln!(out, "  // transient buffers");
        for (idx, compiled) in self.transient_buffers.iter().enumerate() {
            let label = escape_label(compiled.desc.label);
            let _ = writeln!(
                out,
                "  \"buf_t{idx}\" [label=\"{label}\\n(transient buf)\" shape=cylinder];"
            );
        }
        let _ = writeln!(out, "  // imported buffers");
        for (idx, decl) in self.imported_buffers.iter().enumerate() {
            let label = escape_label(decl.label);
            let _ = writeln!(
                out,
                "  \"buf_i{idx}\" [label=\"{label}\\n(imported buf)\" shape=cylinder style=\"filled\" fillcolor=\"#e8e8ff\"];"
            );
        }
    }

    fn emit_pass_clusters(&self, out: &mut String) {
        self.emit_pass_cluster(
            out,
            "cluster_frame_global",
            "FrameGlobal",
            PassPhase::FrameGlobal,
        );
        self.emit_pass_cluster(out, "cluster_per_view", "PerView", PassPhase::PerView);
    }

    fn emit_pass_cluster(&self, out: &mut String, cluster_id: &str, title: &str, phase: PassPhase) {
        let _ = writeln!(out, "  subgraph {cluster_id} {{");
        let _ = writeln!(out, "    label=\"{title}\";");
        let _ = writeln!(out, "    style=dashed;");
        for (pass_idx, info) in self.pass_info.iter().enumerate() {
            if self.passes.get(pass_idx).map(|p| p.phase()) != Some(phase) {
                continue;
            }
            let name = escape_label(&info.name);
            let kind = format!("{:?}", info.kind);
            let _ = writeln!(
                out,
                "    \"pass_{pass_idx}\" [label=\"{name}\\n({kind})\" style=\"filled\" fillcolor=\"#ffffcc\"];"
            );
        }
        let _ = writeln!(out, "  }}");
    }

    fn emit_access_edges(&self, out: &mut String, format: DotFormat) {
        let _ = writeln!(out, "  // access edges");
        for (pass_idx, info) in self.pass_info.iter().enumerate() {
            for access in &info.accesses {
                emit_access_edge(out, pass_idx, info, access, format);
            }
        }
    }
}

fn emit_access_edge(
    out: &mut String,
    pass_idx: usize,
    _info: &CompiledPassInfo,
    access: &ResourceAccess,
    format: DotFormat,
) {
    let resource_id = match access.resource {
        ResourceHandle::Texture(TextureResourceHandle::Transient(h)) => {
            format!("tex_t{}", h.index())
        }
        ResourceHandle::Texture(TextureResourceHandle::Imported(h)) => {
            format!("tex_i{}", h.index())
        }
        ResourceHandle::TextureSubresource(h) => format!("tex_sub{}", h.index()),
        ResourceHandle::Buffer(BufferResourceHandle::Transient(h)) => format!("buf_t{}", h.index()),
        ResourceHandle::Buffer(BufferResourceHandle::Imported(h)) => format!("buf_i{}", h.index()),
    };
    let pass_node = format!("pass_{pass_idx}");
    let color = access_color(&access.access);
    let label = match format {
        DotFormat::Standard => access_short_label(&access.access).to_string(),
        DotFormat::Debug => format!("{:?}", access.access),
    };
    // Writes produce edges pass -> resource; reads produce edges resource -> pass. A read-write
    // access emits both.
    if access.writes() {
        let _ = writeln!(
            out,
            "  \"{pass_node}\" -> \"{resource_id}\" [color=\"{color}\" label=\"{label}\"];"
        );
    }
    if access.reads() {
        let _ = writeln!(
            out,
            "  \"{resource_id}\" -> \"{pass_node}\" [color=\"{color}\" label=\"{label}\"];"
        );
    }
}

fn access_short_label(access: &AccessKind) -> &'static str {
    match access {
        AccessKind::Texture(t) => match t {
            TextureAccess::ColorAttachment { .. } => "color",
            TextureAccess::DepthAttachment { .. } => "depth",
            TextureAccess::Sampled { .. } => "sampled",
            TextureAccess::Storage { .. } => "storage",
            TextureAccess::CopySrc => "copy-src",
            TextureAccess::CopyDst => "copy-dst",
            TextureAccess::Present => "present",
        },
        AccessKind::Buffer(b) => match b {
            BufferAccess::Uniform { .. } => "uniform",
            BufferAccess::Storage { .. } => "storage",
            BufferAccess::Index => "index",
            BufferAccess::Vertex => "vertex",
            BufferAccess::Indirect => "indirect",
            BufferAccess::CopySrc => "copy-src",
            BufferAccess::CopyDst => "copy-dst",
        },
    }
}

fn access_color(access: &AccessKind) -> &'static str {
    match access {
        AccessKind::Texture(t) => match t {
            TextureAccess::ColorAttachment { .. } => "#cc0000",
            TextureAccess::DepthAttachment { .. } => "#6600cc",
            TextureAccess::Sampled { .. } => "#008800",
            TextureAccess::Storage { .. } => "#cc6600",
            TextureAccess::CopySrc | TextureAccess::CopyDst => "#888888",
            TextureAccess::Present => "#000000",
        },
        AccessKind::Buffer(_) => "#2266aa",
    }
}

fn escape_label(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::builder::GraphBuilder;
    use crate::render_graph::context::{ComputePassCtx, RasterPassCtx};
    use crate::render_graph::error::{RenderPassError, SetupError};
    use crate::render_graph::pass::{ComputePass, PassBuilder, PassPhase, RasterPass};
    use crate::render_graph::resources::{
        BufferImportSource, FrameTargetRole, ImportSource, ImportedBufferDecl,
        ImportedBufferHandle, ImportedTextureDecl, ImportedTextureHandle, TextureAccess,
        TextureResourceHandle,
    };

    fn color_import() -> ImportedTextureDecl {
        ImportedTextureDecl {
            label: "color-bb",
            source: ImportSource::FrameTarget(FrameTargetRole::ColorAttachment),
            initial_access: TextureAccess::Present,
            final_access: TextureAccess::Present,
        }
    }

    fn uniform_buffer_import() -> ImportedBufferDecl {
        ImportedBufferDecl {
            label: "frame-uniforms",
            source: BufferImportSource::BackendFrameResource(
                crate::render_graph::resources::BackendFrameBufferKind::FrameUniforms,
            ),
            initial_access: crate::render_graph::resources::BufferAccess::Uniform {
                stages: wgpu::ShaderStages::FRAGMENT,
                dynamic_offset: false,
            },
            final_access: crate::render_graph::resources::BufferAccess::Uniform {
                stages: wgpu::ShaderStages::FRAGMENT,
                dynamic_offset: false,
            },
        }
    }

    struct WritePresent(&'static str, ImportedTextureHandle);
    impl RasterPass for WritePresent {
        fn name(&self) -> &str {
            self.0
        }
        fn phase(&self) -> PassPhase {
            PassPhase::PerView
        }
        fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
            {
                let mut r = b.raster();
                r.color(
                    TextureResourceHandle::Imported(self.1),
                    wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    None::<ImportedTextureHandle>,
                );
            }
            Ok(())
        }
        fn record(
            &self,
            _ctx: &mut RasterPassCtx<'_, '_>,
            _rpass: &mut wgpu::RenderPass<'_>,
        ) -> Result<(), RenderPassError> {
            Ok(())
        }
    }

    struct ReadUniform(&'static str, ImportedBufferHandle);
    impl ComputePass for ReadUniform {
        fn name(&self) -> &str {
            self.0
        }
        fn phase(&self) -> PassPhase {
            PassPhase::FrameGlobal
        }
        fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
            b.compute();
            b.cull_exempt();
            b.import_buffer(
                self.1,
                crate::render_graph::resources::BufferAccess::Uniform {
                    stages: wgpu::ShaderStages::COMPUTE,
                    dynamic_offset: false,
                },
            );
            Ok(())
        }
        fn record(&self, _ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
            Ok(())
        }
    }

    #[test]
    fn to_dot_emits_passes_resources_and_access_edges() {
        let mut b = GraphBuilder::new();
        let bb = b.import_texture(color_import());
        let ub = b.import_buffer(uniform_buffer_import());
        b.add_raster_pass(Box::new(WritePresent("present-pass", bb)));
        b.add_compute_pass(Box::new(ReadUniform("uniform-reader", ub)));
        let g = b.build().expect("graph builds");

        let dot = g.to_dot();
        assert!(dot.starts_with("digraph RenderGraph"));
        assert!(dot.contains("cluster_frame_global"));
        assert!(dot.contains("cluster_per_view"));
        assert!(dot.contains("present-pass"));
        assert!(dot.contains("uniform-reader"));
        assert!(dot.contains("color-bb"));
        assert!(dot.contains("frame-uniforms"));
        // Color attachment edge (present-pass writes color bb).
        assert!(
            dot.contains("\"pass_") && dot.contains("-> \"tex_i"),
            "expected at least one pass -> imported-texture edge, got: {dot}"
        );
        // Uniform read edge (buffer -> pass).
        assert!(
            dot.contains("\"buf_i") && dot.contains("-> \"pass_"),
            "expected at least one imported-buffer -> pass edge, got: {dot}"
        );
    }

    #[test]
    fn to_dot_debug_format_includes_access_debug_output() {
        let mut b = GraphBuilder::new();
        let bb = b.import_texture(color_import());
        b.add_raster_pass(Box::new(WritePresent("present-pass", bb)));
        let g = b.build().expect("graph builds");

        let standard = g.to_dot_with_format(DotFormat::Standard);
        let debug = g.to_dot_with_format(DotFormat::Debug);
        assert!(standard.contains("color"));
        assert!(debug.contains("ColorAttachment"));
    }
}
