//! Shared render-graph builder test fixtures.

pub(super) use super::super::GraphBuilder;
pub(super) use crate::render_graph::context::{ComputePassCtx, RasterPassCtx};
pub(super) use crate::render_graph::error::{GraphBuildError, RenderPassError, SetupError};
pub(super) use crate::render_graph::ids::PassId;
pub(super) use crate::render_graph::pass::{
    ComputePass, GroupScope, PassBuilder, PassMergeHint, PassPhase, RasterPass,
};
pub(super) use crate::render_graph::resources::{
    BufferAccess, BufferHandle, BufferImportSource, BufferSizePolicy, FrameTargetRole,
    HistorySlotId, ImportedBufferDecl, ImportedBufferHandle, ImportedTextureDecl,
    ImportedTextureHandle, StorageAccess, SubresourceHandle, TextureAccess,
    TextureAttachmentResolve, TextureAttachmentTarget, TextureHandle, TextureResourceHandle,
    TransientBufferDesc, TransientExtent, TransientSubresourceDesc, TransientTextureDesc,
    TransientTextureFormat,
};

/// Minimal compute test pass.
pub(super) struct TestComputePass {
    pub(super) name: &'static str,
    pub(super) phase: PassPhase,
    pub(super) texture_reads: Vec<TextureHandle>,
    pub(super) texture_writes: Vec<TextureHandle>,
    pub(super) subresource_reads: Vec<SubresourceHandle>,
    pub(super) subresource_writes: Vec<SubresourceHandle>,
    pub(super) buffer_reads: Vec<BufferHandle>,
    pub(super) buffer_writes: Vec<BufferHandle>,
    pub(super) imported_texture_writes: Vec<ImportedTextureHandle>,
    pub(super) imported_buffer_writes: Vec<ImportedBufferHandle>,
    pub(super) cull_exempt: bool,
}

impl TestComputePass {
    pub(super) fn new(name: &'static str) -> Self {
        Self {
            name,
            phase: PassPhase::PerView,
            texture_reads: Vec::new(),
            texture_writes: Vec::new(),
            subresource_reads: Vec::new(),
            subresource_writes: Vec::new(),
            buffer_reads: Vec::new(),
            buffer_writes: Vec::new(),
            imported_texture_writes: Vec::new(),
            imported_buffer_writes: Vec::new(),
            cull_exempt: false,
        }
    }

    pub(super) fn frame_global(mut self) -> Self {
        self.phase = PassPhase::FrameGlobal;
        self
    }

    pub(super) fn cull_exempt(mut self) -> Self {
        self.cull_exempt = true;
        self
    }
}

impl ComputePass for TestComputePass {
    fn name(&self) -> &str {
        self.name
    }

    fn phase(&self) -> PassPhase {
        self.phase
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        if self.cull_exempt {
            b.cull_exempt();
        }
        for &h in &self.texture_reads {
            b.read_texture(
                h,
                TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::COMPUTE,
                },
            );
        }
        for &h in &self.texture_writes {
            b.write_texture(h, TextureAccess::CopyDst);
        }
        for &h in &self.subresource_reads {
            b.read_texture_subresource(
                h,
                TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::COMPUTE,
                },
            );
        }
        for &h in &self.subresource_writes {
            b.write_texture_subresource(h, TextureAccess::CopyDst);
        }
        for &h in &self.buffer_reads {
            b.read_buffer(
                h,
                BufferAccess::Storage {
                    stages: wgpu::ShaderStages::COMPUTE,
                    access: StorageAccess::ReadOnly,
                },
            );
        }
        for &h in &self.buffer_writes {
            b.write_buffer(h, BufferAccess::CopyDst);
        }
        for &h in &self.imported_texture_writes {
            b.import_texture(h, TextureAccess::Present);
        }
        for &h in &self.imported_buffer_writes {
            b.import_buffer(h, BufferAccess::CopyDst);
        }
        Ok(())
    }

    fn record(&self, _ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        Ok(())
    }
}

/// Minimal raster test pass.
pub(super) struct TestRasterPass {
    pub(super) name: &'static str,
    pub(super) color: TextureResourceHandle,
    pub(super) texture_reads: Vec<TextureHandle>,
    pub(super) imported_texture_writes: Vec<ImportedTextureHandle>,
    pub(super) multiview_mask: Option<std::num::NonZeroU32>,
    pub(super) depth: Option<TextureResourceHandle>,
    pub(super) resolve: Option<TextureResourceHandle>,
    pub(super) frame_sampled_color: Option<(
        TextureResourceHandle,
        TextureResourceHandle,
        Option<TextureResourceHandle>,
    )>,
    pub(super) frame_sampled_depth: Option<(TextureResourceHandle, TextureResourceHandle)>,
}

impl TestRasterPass {
    pub(super) fn new(name: &'static str, color: impl Into<TextureResourceHandle>) -> Self {
        Self {
            name,
            color: color.into(),
            texture_reads: Vec::new(),
            imported_texture_writes: Vec::new(),
            multiview_mask: None,
            depth: None,
            resolve: None,
            frame_sampled_color: None,
            frame_sampled_depth: None,
        }
    }
}

impl RasterPass for TestRasterPass {
    fn name(&self) -> &str {
        self.name
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        {
            let mut r = b.raster();
            if let Some((single, msaa, res)) = self.frame_sampled_color {
                r.frame_sampled_color(
                    single,
                    msaa,
                    wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: wgpu::StoreOp::Store,
                    },
                    res,
                );
            } else {
                r.color(
                    self.color,
                    wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    self.resolve,
                );
            }
            if let Some((single, msaa)) = self.frame_sampled_depth {
                r.frame_sampled_depth(
                    single,
                    msaa,
                    wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.5),
                        store: wgpu::StoreOp::Store,
                    },
                    None,
                );
            } else if let Some(d) = self.depth {
                r.depth(
                    d,
                    wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.5),
                        store: wgpu::StoreOp::Store,
                    },
                    None,
                );
            }
            if let Some(mask) = self.multiview_mask {
                r.multiview(mask);
            }
        }
        for &h in &self.texture_reads {
            b.read_texture(
                h,
                TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::FRAGMENT,
                },
            );
        }
        for &h in &self.imported_texture_writes {
            b.import_texture(h, TextureAccess::Present);
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

// -----------------------------------------------------------------------------
// Helper descriptors
// -----------------------------------------------------------------------------

pub(super) fn tex_desc(label: &'static str) -> TransientTextureDesc {
    TransientTextureDesc::texture_2d(
        label,
        wgpu::TextureFormat::Rgba8Unorm,
        TransientExtent::Custom {
            width: 64,
            height: 64,
        },
        1,
        wgpu::TextureUsages::empty(),
    )
}

pub(super) fn frame_sampled_tex_desc(label: &'static str) -> TransientTextureDesc {
    TransientTextureDesc::frame_sampled_texture_2d(
        label,
        wgpu::TextureFormat::Rgba8Unorm,
        TransientExtent::Custom {
            width: 64,
            height: 64,
        },
        wgpu::TextureUsages::empty(),
    )
}

pub(super) fn mip_chain_tex_desc(label: &'static str, mip_levels: u32) -> TransientTextureDesc {
    TransientTextureDesc {
        label,
        format: TransientTextureFormat::Fixed(wgpu::TextureFormat::R32Float),
        extent: TransientExtent::Custom {
            width: 256,
            height: 256,
        },
        mip_levels,
        sample_count: crate::render_graph::resources::TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: crate::render_graph::resources::TransientArrayLayers::Fixed(1),
        base_usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        alias: false,
    }
}

pub(super) fn backbuffer_import() -> ImportedTextureDecl {
    ImportedTextureDecl {
        label: "backbuffer",
        source: crate::render_graph::resources::ImportSource::Frame(
            FrameTargetRole::ColorAttachment,
        ),
        initial_access: TextureAccess::ColorAttachment {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
            resolve_to: None,
        },
        final_access: TextureAccess::Present,
    }
}

pub(super) fn depth_import() -> ImportedTextureDecl {
    ImportedTextureDecl {
        label: "depth",
        source: crate::render_graph::resources::ImportSource::Frame(
            FrameTargetRole::DepthAttachment,
        ),
        initial_access: TextureAccess::DepthAttachment {
            depth: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            stencil: None,
        },
        final_access: TextureAccess::Sampled {
            stages: wgpu::ShaderStages::COMPUTE,
        },
    }
}

pub(super) fn buffer_import_readback() -> ImportedBufferDecl {
    ImportedBufferDecl {
        label: "readback",
        source: BufferImportSource::External,
        initial_access: BufferAccess::CopyDst,
        final_access: BufferAccess::CopyDst,
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------
