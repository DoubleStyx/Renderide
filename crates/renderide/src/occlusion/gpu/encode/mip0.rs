//! Hi-Z mip0 dispatch from the source depth attachment.
//!
//! Supports both the desktop (2D depth view) and stereo (2D-array depth view, one dispatch per
//! eye layer) layouts. Each path writes one bind group's worth of pyramid mip0 texels via the
//! compute pipeline cached in [`super::EncodeSession::pipes`].

use super::EncodeSession;

/// Whether the depth source is a 2D view (desktop) or a 2D array slice (one eye of a stereo target).
#[derive(Clone, Copy)]
pub(super) enum DepthBinding {
    /// Desktop (single-view) 2D depth view.
    D2,
    /// One layer of a multi-layer depth array view.
    D2Array {
        /// Array layer index sampled by the stereo mip0 shader.
        layer: u32,
    },
}

/// Fills Hi-Z mip0 from a depth texture (desktop 2D view or one layer of a stereo depth array).
pub(super) fn dispatch(
    session: &mut EncodeSession<'_>,
    pyramid_views: &[wgpu::TextureView],
    depth_bind: DepthBinding,
) {
    match depth_bind {
        DepthBinding::D2 => dispatch_desktop(session, pyramid_views),
        DepthBinding::D2Array { layer } => dispatch_stereo(session, pyramid_views, layer),
    }
}

/// Mip0 dispatch for the desktop (non-stereo) 2D depth view.
fn dispatch_desktop(session: &mut EncodeSession<'_>, pyramid_views: &[wgpu::TextureView]) {
    let device = session.device;
    let depth_view = session.depth_view;
    let layout = &session.pipes.bgl_mip0_desktop;
    let bg = session.scratch.bind_groups.mip0_desktop_or_build(|| {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hi_z_mip0_d_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&pyramid_views[0]),
                },
            ],
        });
        crate::profiling::note_resource_churn!(BindGroup, "occlusion::hi_z_mip0_desktop_bg");
        bind_group
    });
    let pass_query = session
        .profiler
        .map(|p| p.begin_pass_query("hi_z_mip0_desktop", session.encoder));
    let timestamp_writes = crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
    {
        let mut pass = session
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hi_z_mip0_desktop"),
                timestamp_writes,
            });
        pass.set_pipeline(&session.pipes.mip0_desktop);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(
            session.scratch.extent.0.div_ceil(8),
            session.scratch.extent.1.div_ceil(8),
            1,
        );
    };
    if let (Some(p), Some(q)) = (session.profiler, pass_query) {
        p.end_query(session.encoder, q);
    }
}

/// Mip0 dispatch for one array layer of a stereo depth target.
fn dispatch_stereo(
    session: &mut EncodeSession<'_>,
    pyramid_views: &[wgpu::TextureView],
    layer: u32,
) {
    let device = session.device;
    let depth_view = session.depth_view;
    let layout = &session.pipes.bgl_mip0_stereo;
    // Clone the uniform buffer handle so the bind-group build closure does not borrow
    // `session.scratch` for the duration of `mip0_stereo_or_build`'s `&mut bind_groups` borrow.
    let Some(layer_uniforms) = session.scratch.layer_uniforms.as_ref() else {
        return;
    };
    let Some(layer_uniform) = layer_uniforms.get(layer as usize).cloned() else {
        return;
    };
    let bg = session.scratch.bind_groups.mip0_stereo_or_build(layer, || {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hi_z_mip0_s_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: layer_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&pyramid_views[0]),
                },
            ],
        });
        crate::profiling::note_resource_churn!(BindGroup, "occlusion::hi_z_mip0_stereo_bg");
        bind_group
    });
    let pass_query = session
        .profiler
        .map(|p| p.begin_pass_query("hi_z_mip0_stereo", session.encoder));
    let timestamp_writes = crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
    {
        let mut pass = session
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hi_z_mip0_stereo"),
                timestamp_writes,
            });
        pass.set_pipeline(&session.pipes.mip0_stereo);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(
            session.scratch.extent.0.div_ceil(8),
            session.scratch.extent.1.div_ceil(8),
            1,
        );
    };
    if let (Some(p), Some(q)) = (session.profiler, pass_query) {
        p.end_query(session.encoder, q);
    }
}
