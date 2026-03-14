//! Platonic renderer types: engine-agnostic scene, view, and draw primitives.
//!
//! These types represent a generic AAA renderer model, independent of the host protocol.

use std::fmt;

use nalgebra::Vector2;

use crate::shared::RenderTransform;

/// Opaque handle for a mesh asset. Maps to host asset_id.
pub type MeshHandle = i32;

/// Opaque handle for a material asset. Maps to host asset_id.
pub type MaterialHandle = i32;

/// Opaque handle for a texture asset. Maps to host asset_id.
pub type TextureHandle = i32;

/// Scene identifier. Maps to host RenderSpace id.
pub type SceneId = i32;

/// View identifier.
pub type ViewId = i32;

/// Node identifier within a scene (transform index).
pub type NodeId = i32;

/// Transform: position, scale, rotation. Alias for host RenderTransform.
pub type Transform = RenderTransform;

/// Viewport dimensions.
#[derive(Clone, Copy, Debug, Default)]
pub struct Viewport {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Viewport {
    /// Creates a viewport from width and height.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Aspect ratio (width / height).
    pub fn aspect(&self) -> f32 {
        let h = self.height.max(1);
        self.width as f32 / h as f32
    }
}

/// Camera projection type.
#[derive(Clone, Copy, Debug)]
pub enum Projection {
    /// Perspective projection with vertical FOV.
    Perspective {
        /// Vertical field of view in degrees.
        fov_deg: f32,
        /// Aspect ratio (width / height).
        aspect: f32,
    },
    /// Orthographic projection.
    Orthographic {
        /// Half-size of the orthographic view.
        half_size: f32,
        /// Aspect ratio.
        aspect: f32,
    },
    /// Panoramic projection.
    Panoramic {
        /// Field of view in degrees.
        fov_deg: f32,
        /// Aspect ratio.
        aspect: f32,
    },
}

/// View: camera transform, projection, and viewport.
#[derive(Clone)]
pub struct View {
    /// Unique view identifier.
    pub id: ViewId,
    /// Camera transform (position, rotation, scale).
    pub transform: Transform,
    /// Projection parameters.
    pub projection: Projection,
    /// Viewport dimensions.
    pub viewport: Viewport,
    /// Near and far clip distances.
    pub near_far: (f32, f32),
}

impl fmt::Debug for View {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("View")
            .field("id", &self.id)
            .field("viewport", &self.viewport)
            .field("near_far", &self.near_far)
            .finish()
    }
}

/// Drawable: geometry reference with material and sort key.
#[derive(Clone, Debug)]
pub struct Drawable {
    /// Node (transform) this drawable is attached to.
    pub node_id: NodeId,
    /// Mesh asset handle.
    pub mesh_handle: MeshHandle,
    /// Optional material handle. -1 or None means no material.
    pub material_handle: Option<MaterialHandle>,
    /// Sort key for render order.
    pub sort_key: i32,
    /// Whether this is a skinned mesh (requires bone matrices).
    pub is_skinned: bool,
    /// For skinned meshes: transform IDs for each bone. Index i maps to bone i in the mesh.
    pub bone_transform_ids: Option<Vec<i32>>,
}

impl Default for Drawable {
    fn default() -> Self {
        Self {
            node_id: -1,
            mesh_handle: -1,
            material_handle: None,
            sort_key: 0,
            is_skinned: false,
            bone_transform_ids: None,
        }
    }
}

/// Scene: collection of nodes and drawables in a coordinate space.
#[derive(Clone)]
pub struct Scene {
    /// Scene identifier.
    pub id: SceneId,
    /// Whether this scene is the active (primary) render target.
    pub is_active: bool,
    /// Whether this is an overlay (e.g. UI) rendered on top.
    pub is_overlay: bool,
    /// View transform for this scene (camera/root).
    pub view_transform: Transform,
    /// Root transform of the scene hierarchy. All node world matrices are relative to this.
    pub root_transform: Transform,
    /// Transform hierarchy: index = node_id.
    pub nodes: Vec<Transform>,
    /// Parent index per node (-1 = root).
    pub node_parents: Vec<i32>,
    /// Mesh drawables.
    pub drawables: Vec<Drawable>,
    /// Skinned mesh drawables.
    pub skinned_drawables: Vec<Drawable>,
}

impl fmt::Debug for Scene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scene")
            .field("id", &self.id)
            .field("is_active", &self.is_active)
            .field("is_overlay", &self.is_overlay)
            .field("nodes_len", &self.nodes.len())
            .field("drawables_len", &self.drawables.len())
            .field("skinned_drawables_len", &self.skinned_drawables.len())
            .finish()
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            id: -1,
            is_active: false,
            is_overlay: false,
            view_transform: Transform::default(),
            root_transform: Transform::default(),
            nodes: Vec::new(),
            node_parents: Vec::new(),
            drawables: Vec::new(),
            skinned_drawables: Vec::new(),
        }
    }
}

/// Render target descriptor for offscreen renders.
#[derive(Clone, Debug)]
pub struct RenderTargetDesc {
    /// Buffer descriptor for shared memory result.
    pub buffer_id: i32,
    /// Resolution.
    pub resolution: Vector2<i32>,
}

/// Frame: one frame's worth of render state from the host.
#[derive(Clone, Debug)]
pub struct Frame {
    /// Frame index from host.
    pub frame_index: i32,
    /// Camera views (from CameraRenderTask).
    pub views: Vec<View>,
    /// Scenes (from RenderSpaceUpdate).
    pub scenes: Vec<Scene>,
    /// Offscreen render targets.
    pub render_targets: Vec<RenderTargetDesc>,
}

impl Default for Frame {
    fn default() -> Self {
        Self {
            frame_index: -1,
            views: Vec::new(),
            scenes: Vec::new(),
            render_targets: Vec::new(),
        }
    }
}
