//! Applies host light batches from shared memory into [`LightCache`](super::LightCache).

use std::mem::size_of;

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    LightRenderablesUpdate, LightState, LightsBufferRendererState, LightsBufferRendererUpdate,
    LIGHTS_BUFFER_RENDERER_STATE_HOST_ROW_BYTES, LIGHT_STATE_HOST_ROW_BYTES,
};

use crate::scene::error::SceneError;

use super::LightCache;

/// Applies [`LightRenderablesUpdate`] for one render space (`space_id` = host render space id).
pub fn apply_light_renderables_update(
    light_cache: &mut LightCache,
    shm: &mut SharedMemoryAccessor,
    update: &LightRenderablesUpdate,
    space_id: i32,
) -> Result<(), SceneError> {
    profiling::scope!("scene::apply_lights");
    let i32_size = size_of::<i32>() as i32;
    let state_size = LIGHT_STATE_HOST_ROW_BYTES as i32;

    let removals = if update.removals.length >= i32_size {
        let ctx = format!("light renderables removals space_id={space_id}");
        shm.access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    let additions = if update.additions.length >= i32_size {
        let ctx = format!("light renderables additions space_id={space_id}");
        shm.access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    let states = if update.states.length >= state_size {
        let ctx = format!("light renderables states space_id={space_id}");
        shm.access_copy_memory_packable_rows::<LightState>(
            &update.states,
            LIGHT_STATE_HOST_ROW_BYTES,
            Some(&ctx),
        )
        .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    light_cache.apply_regular_lights_update(space_id, &removals, &additions, &states);
    Ok(())
}

/// Applies [`LightsBufferRendererUpdate`] for one render space.
pub fn apply_lights_buffer_renderers_update(
    light_cache: &mut LightCache,
    shm: &mut SharedMemoryAccessor,
    update: &LightsBufferRendererUpdate,
    space_id: i32,
) -> Result<(), SceneError> {
    profiling::scope!("scene::apply_lights_buffer_renderers");
    let i32_size = size_of::<i32>() as i32;
    let state_size = LIGHTS_BUFFER_RENDERER_STATE_HOST_ROW_BYTES as i32;

    let removals = if update.removals.length >= i32_size {
        let ctx = format!("lights buffer renderers removals space_id={space_id}");
        shm.access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    let additions = if update.additions.length >= i32_size {
        let ctx = format!("lights buffer renderers additions space_id={space_id}");
        shm.access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    let states = if update.states.length >= state_size {
        let ctx = format!("lights buffer renderers states space_id={space_id}");
        shm.access_copy_memory_packable_rows::<LightsBufferRendererState>(
            &update.states,
            LIGHTS_BUFFER_RENDERER_STATE_HOST_ROW_BYTES,
            Some(&ctx),
        )
        .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    light_cache.apply_update(space_id, &removals, &additions, &states);
    Ok(())
}
