//! Light updates from host: regular lights and lights buffer renderers.
//!
//! Applies incremental updates (states, removals, additions) to the light cache
//! for each render space. Regular lights (Light component) use LightRenderablesUpdate;
//! buffered lights (LocalLightsBufferRenderer) use LightsBufferRendererUpdate.

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::scene::LightCache;
use crate::shared::{
    LightRenderablesUpdate, LightState, LightsBufferRendererState, LightsBufferRendererUpdate,
};

use super::super::error::SceneError;

/// Applies lights buffer renderer updates from host.
///
/// Reads removals (i32 indices), additions (i32 transform indices), and
/// states (LightsBufferRendererState) from shared memory and merges with the
/// light cache. Uses space_id as the buffer key (1:1 mapping with lights_buffer_unique_id).
///
/// Per RenderablesUpdate base class, additions are int[] (transform indices), not state structs.
pub(crate) fn apply_lights_buffer_renderers_update(
    light_cache: &mut LightCache,
    shm: &mut SharedMemoryAccessor,
    update: &LightsBufferRendererUpdate,
    space_id: i32,
) -> Result<(), SceneError> {
    let i32_size = std::mem::size_of::<i32>() as i32;
    let state_size = std::mem::size_of::<LightsBufferRendererState>() as i32;

    let removals = if update.removals.length >= i32_size {
        let ctx = format!("lights removals space_id={}", space_id);
        shm.access_with_context::<i32>(&update.removals, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    let additions = if update.additions.length >= i32_size {
        let ctx = format!("lights additions space_id={}", space_id);
        shm.access_with_context::<i32>(&update.additions, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    let states = if update.states.length >= state_size {
        let ctx = format!("lights states space_id={}", space_id);
        shm.access_with_context::<LightsBufferRendererState>(&update.states, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    light_cache.apply_update(space_id, &removals, &additions, &states);

    Ok(())
}

/// Applies regular light updates (Light components) from host.
///
/// Reads removals (i32 indices), additions (i32 transform indices), and states
/// (LightState) from shared memory and merges into the light cache for the space.
/// Each LightState describes one scene light; position/direction come from the
/// transform at renderable_index.
pub(crate) fn apply_lights_update(
    light_cache: &mut LightCache,
    shm: &mut SharedMemoryAccessor,
    update: &LightRenderablesUpdate,
    space_id: i32,
) -> Result<(), SceneError> {
    let i32_size = std::mem::size_of::<i32>() as i32;
    let state_size = std::mem::size_of::<LightState>() as i32;

    let removals = if update.removals.length >= i32_size {
        let ctx = format!("regular lights removals space_id={}", space_id);
        shm.access_with_context::<i32>(&update.removals, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    let additions = if update.additions.length >= i32_size {
        let ctx = format!("regular lights additions space_id={}", space_id);
        shm.access_with_context::<i32>(&update.additions, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    let states = if update.states.length >= state_size {
        let ctx = format!("regular lights states space_id={}", space_id);
        shm.access_with_context::<LightState>(&update.states, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    light_cache.apply_regular_lights_update(space_id, &removals, &additions, &states);

    Ok(())
}
