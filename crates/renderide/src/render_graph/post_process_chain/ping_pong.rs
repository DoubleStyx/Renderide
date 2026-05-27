//! Ping-pong HDR transient slot helper for post-processing chains.
//!
//! A one- or two-slot rotation lets each effect read the previous effect's output and write into a
//! sibling slot without forcing the chain to allocate `N+1` transient targets. The first effect
//! reads the chain input and writes into [`PingPongHdrSlots::ping`]; multi-effect chains also
//! allocate [`PingPongHdrSlots::pong`] and swap between the pair each step.

use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::resources::{
    TextureHandle, TransientArrayLayers, TransientExtent, TransientSampleCount,
    TransientTextureDesc, TransientTextureFormat,
};

/// The HDR ping-pong transient texture handles used by a post-processing chain.
#[derive(Clone, Copy, Debug)]
pub(super) struct PingPongHdrSlots {
    /// First-write target slot.
    pub ping: TextureHandle,
    /// Second-write target slot, allocated only for chains that can need a swap target.
    pub pong: Option<TextureHandle>,
}

impl PingPongHdrSlots {
    /// Creates the ping-pong transient HDR scene-color textures needed by `enabled_effect_count`.
    pub fn new(builder: &mut GraphBuilder, enabled_effect_count: usize) -> Self {
        Self {
            ping: builder.create_texture(post_process_color_transient_desc(
                "post_processed_color_hdr_a",
            )),
            pong: (enabled_effect_count > 1).then(|| {
                builder.create_texture(post_process_color_transient_desc(
                    "post_processed_color_hdr_b",
                ))
            }),
        }
    }
}

/// Walks a sequence of effects through a [`PingPongHdrSlots`] pair, exposing the current effect's
/// `(input, output)` handles and advancing the cursor each time an effect registers.
///
/// The first advance abandons the chain input and starts the rotation between allocated slots.
/// Subsequent advances are pure swaps when a second target exists.
pub(super) struct PingPongCursor {
    slots: PingPongHdrSlots,
    input: TextureHandle,
    output: TextureHandle,
    advanced_once: bool,
}

impl PingPongCursor {
    /// Starts the cursor at `(initial_input, slots.ping)`.
    pub fn start(slots: PingPongHdrSlots, initial_input: TextureHandle) -> Self {
        Self {
            input: initial_input,
            output: slots.ping,
            slots,
            advanced_once: false,
        }
    }

    /// Read source for the current effect.
    pub fn input(&self) -> TextureHandle {
        self.input
    }

    /// Write target for the current effect.
    pub fn output(&self) -> TextureHandle {
        self.output
    }

    /// Moves to the next effect: `input` becomes what the just-completed effect wrote, `output`
    /// becomes the sibling slot.
    pub fn advance(&mut self) {
        if self.advanced_once {
            if self.slots.pong.is_some() {
                std::mem::swap(&mut self.input, &mut self.output);
            }
        } else {
            self.input = self.slots.ping;
            self.output = match self.slots.pong {
                Some(pong) => pong,
                None => self.slots.ping,
            };
            self.advanced_once = true;
        }
    }

    /// Slot the most recent effect wrote into (i.e. the chain's final output).
    pub fn last_output(&self) -> TextureHandle {
        self.input
    }
}

/// Standard transient texture descriptor used by both ping-pong slots.
fn post_process_color_transient_desc(label: &'static str) -> TransientTextureDesc {
    TransientTextureDesc {
        label,
        format: TransientTextureFormat::SceneColorHdr,
        extent: TransientExtent::Backbuffer,
        mip_levels: 1,
        sample_count: TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        alias: true,
    }
}
