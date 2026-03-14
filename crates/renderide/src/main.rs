use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use nalgebra::{Matrix4, Orthographic3, Point3, UnitQuaternion, Vector2, Vector3};
use crate::shared::{
    CameraProjection, CameraRenderTask, IndexBufferFormat, InputState, Key, KeyboardState,
    MouseState, RenderTransform, VertexAttributeType, WindowState,
};
use crate::session::Session;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents, EventLoop};
use winit::window::{CursorGrabMode, Window, WindowAttributes};

mod assets;
mod backend;
mod command;
mod core;
mod gpu_mesh;
mod init;
mod log;
mod scene;
mod session;
mod shared;
mod view;

fn main() {
    log::init_log();
    log::log_write("[Renderide] main: starting");

    // Panic hook: write panics to Renderide.log before default stderr output.
    let log_path = log::log_path();
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            let _ = writeln!(f, "PANIC: {}", info);
            let _ = writeln!(f, "Backtrace:\n{:?}", std::backtrace::Backtrace::capture());
            let _ = f.flush();
        }
        default_hook(info);
    }));
    let event_loop = EventLoop::new().unwrap();
    let mut app = RenderideApp::new();

    if let Err(e) = app.session.init() {
        log::log_write(&format!("[Renderide] main: Failed to initialize Session: {}", e));
        std::process::exit(1);
    }
    log::log_write("[Renderide] main: init OK, entering event loop");

    let _ = event_loop.run_app(&mut app);

    if let Some(code) = app.exit_code {
        std::process::exit(code);
    }
}

/// Accumulated window input for gaze/mouse (sent to host via FrameStartData).
struct WindowInputState {
    mouse_delta: Vector2<f32>,
    scroll_delta: Vector2<f32>,
    window_position: Vector2<f32>,
    window_resolution: (u32, u32),
    left_held: bool,
    right_held: bool,
    middle_held: bool,
    button4_held: bool,
    button5_held: bool,
    mouse_active: bool,
    window_focused: bool,
    held_keys: Vec<Key>,
}

impl Default for WindowInputState {
    fn default() -> Self {
        Self {
            mouse_delta: Vector2::zeros(),
            scroll_delta: Vector2::zeros(),
            window_position: Vector2::zeros(),
            window_resolution: (0, 0),
            left_held: false,
            right_held: false,
            middle_held: false,
            button4_held: false,
            button5_held: false,
            mouse_active: false,
            window_focused: true,
            held_keys: Vec::new(),
        }
    }
}

impl WindowInputState {
    fn take_input_state(&mut self) -> InputState {
        let mouse = MouseState {
            is_active: self.mouse_active,
            left_button_state: self.left_held,
            right_button_state: self.right_held,
            middle_button_state: self.middle_held,
            button4_state: self.button4_held,
            button5_state: self.button5_held,
            desktop_position: self.window_position,
            window_position: self.window_position,
            direct_delta: std::mem::take(&mut self.mouse_delta),
            scroll_wheel_delta: std::mem::take(&mut self.scroll_delta),
        };
        let window = WindowState {
            is_window_focused: self.window_focused,
            is_fullscreen: false,
            window_resolution: Vector2::new(self.window_resolution.0 as i32, self.window_resolution.1 as i32),
            resolution_settings_applied: false,
            drag_and_drop_event: None,
        };
        let keyboard = Some(KeyboardState {
            type_delta: None,
            held_keys: self.held_keys.clone(),
        });
        InputState {
            mouse: Some(mouse),
            keyboard,
            window: Some(window),
            vr: None,
            gamepads: Vec::new(),
            touches: Vec::new(),
            displays: Vec::new(),
        }
    }
}

/// Global render frame counter (used by proof logs).
static mut RENDER_COUNT: u64 = 0;
/// Whether we have logged set_cursor_position failure (Wayland: cursor position only when locked).
static CURSOR_POSITION_ERROR_LOGGED: AtomicBool = AtomicBool::new(false);

struct RenderideApp {
    session: Session,
    window: Option<Window>,
    /// wgpu state - initialized when window is created
    gpu: Option<GpuState>,
    /// Exit code to use when event loop exits (Update requested quit).
    exit_code: Option<i32>,
    input: WindowInputState,
    /// When unfocused, throttle redraw requests to 60 Hz so we keep rendering (winit stops
    /// RedrawRequested when window loses focus).
    last_unfocused_redraw: Option<Instant>,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    mesh_pipeline: gpu_mesh::MeshPipeline,
    mesh_buffer_cache: std::collections::HashMap<i32, gpu_mesh::GpuMeshBuffers>,
    depth_texture: Option<wgpu::Texture>,
}

impl RenderideApp {
    fn new() -> Self {
        Self {
            session: Session::new(),
            window: None,
            gpu: None,
            exit_code: None,
            input: WindowInputState::default(),
            last_unfocused_redraw: None,
        }
    }
}

impl ApplicationHandler for RenderideApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.listen_device_events(DeviceEvents::Always);
        if self.window.is_none() {
            let attrs = WindowAttributes::default().with_title("Renderide");
            match event_loop.create_window(attrs) {
                Ok(w) => self.window = Some(w),
                Err(e) => eprintln!("Failed to create window: {}", e),
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            // Host (CameraLook YX swizzle) expects: yaw += direct_delta.x, pitch -= direct_delta.y.
            // Winit: right positive X, down positive Y. Match Gloobie: x unchanged, y negated.
            self.input.mouse_delta.x += delta.0 as f32;
            self.input.mouse_delta.y -= delta.1 as f32;
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(ref window) = self.window {
                    let size = window.inner_size();
                    self.input.window_resolution = (size.width, size.height);
                    let center = Vector2::new((size.width / 2) as f32, (size.height / 2) as f32);
                    let lock = self.session.cursor_lock_requested();

                    if lock {
                        let _ = window.set_cursor_grab(CursorGrabMode::Locked)
                            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                        let _ = window.set_cursor_visible(false);
                        // Use only DeviceEvent::MouseMotion for direct_delta when locked - avoid
                        // position-based delta which causes "absolute" head rotation.
                        let center_phys = PhysicalPosition::new(size.width / 2, size.height / 2);
                        if let Err(e) = window.set_cursor_position(center_phys) {
                            if !CURSOR_POSITION_ERROR_LOGGED.swap(true, Ordering::Relaxed) {
                                log::log_write(&format!(
                                    "[Renderide] set_cursor_position: {:?} (Wayland: cursor position only when locked; further errors suppressed)",
                                    e
                                ));
                            }
                        }
                        self.input.window_position = center;
                    } else {
                        let _ = window.set_cursor_grab(CursorGrabMode::None);
                        let _ = window.set_cursor_visible(true);
                        // When not focused, send center to avoid grey screen from wrong gaze.
                        if !self.input.window_focused {
                            self.input.window_position = center;
                        }
                    }
                }

                let mut input = self.input.take_input_state();
                if let Some(ref mut m) = input.mouse {
                    m.is_active = m.is_active || self.session.cursor_lock_requested();
                }
                self.session.set_pending_input(input);
                static mut FRAME: u64 = 0;
                unsafe { FRAME += 1; }
                if unsafe { FRAME } % log::DIAG_FRAME_INTERVAL == 0 {
                    log::log_write(&format!("[Renderide] redraw frame {}", unsafe { FRAME }));
                }
                if let Some(code) = self.session.update() {
                    self.exit_code = Some(code);
                    event_loop.exit();
                    return;
                }
                self.session.process_render_tasks();

                // wgpu: init on first redraw, then render
                if let (Some(window), None) = (&self.window, &self.gpu) {
                    match pollster::block_on(init_gpu(window)) {
                        Ok(gpu) => self.gpu = Some(gpu),
                        Err(e) => log::log_write(&format!("wgpu init error: {:?}", e)),
                    }
                }
                if let Some(ref mut gpu) = self.gpu {
                    for asset_id in self.session.drain_pending_mesh_unloads() {
                        gpu.mesh_buffer_cache.remove(&asset_id);
                    }
                    if let Err(e) = render_frame(gpu, &mut self.session) {
                        log::log_write(&format!("wgpu render error: {}", e));
                    }
                }
            }
            WindowEvent::Resized(size) => {
                self.input.window_resolution = (size.width, size.height);
                if let Some(ref mut gpu) = self.gpu {
                    gpu.config.width = size.width;
                    gpu.config.height = size.height;
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    gpu.depth_texture = Some(create_depth_texture(&gpu.device, &gpu.config));
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.window_position.x = position.x as f32;
                self.input.window_position.y = position.y as f32;
            }
            WindowEvent::CursorEntered { .. } => self.input.mouse_active = true,
            WindowEvent::CursorLeft { .. } => self.input.mouse_active = false,
            WindowEvent::Focused(focused) => self.input.window_focused = focused,
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => self.input.left_held = pressed,
                    MouseButton::Right => self.input.right_held = pressed,
                    MouseButton::Middle => self.input.middle_held = pressed,
                    MouseButton::Back => self.input.button4_held = pressed,
                    MouseButton::Forward => self.input.button5_held = pressed,
                    MouseButton::Other(_) => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                const SCROLL_SCALE: f32 = 120.0;
                match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        self.input.scroll_delta.x += x * SCROLL_SCALE;
                        self.input.scroll_delta.y += y * SCROLL_SCALE;
                    }
                    MouseScrollDelta::PixelDelta(p) => {
                        self.input.scroll_delta.x += p.x as f32;
                        self.input.scroll_delta.y += p.y as f32;
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.repeat {
                    return;
                }
                if let Some(key) = winit_key_to_renderite_key(event.physical_key) {
                    match event.state {
                        ElementState::Pressed => {
                            if !self.input.held_keys.contains(&key) {
                                self.input.held_keys.push(key);
                            }
                        }
                        ElementState::Released => {
                            self.input.held_keys.retain(|held| *held != key);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let unfocused_redraw_interval = Duration::from_secs_f32(1.0 / 60.0);

        if let Some(ref window) = self.window {
            if self.input.window_focused {
                self.last_unfocused_redraw = None;
                window.request_redraw();
                event_loop.set_control_flow(ControlFlow::Wait);
            } else {
                // When unfocused, use a timer to wake the event loop every 16ms so we keep
                // rendering. Without this, the loop blocks indefinitely and we stop drawing.
                event_loop.set_control_flow(ControlFlow::WaitUntil(
                    Instant::now() + unfocused_redraw_interval,
                ));

                let now = Instant::now();
                let should_redraw = self
                    .last_unfocused_redraw
                    .map(|t| now.duration_since(t) >= unfocused_redraw_interval)
                    .unwrap_or(true);
                if should_redraw {
                    self.last_unfocused_redraw = Some(now);
                    let mut input = self.input.take_input_state();
                    if let Some(ref mut m) = input.mouse {
                        m.is_active = m.is_active || self.session.cursor_lock_requested();
                    }
                    self.session.set_pending_input(input);
                    if let Some(code) = self.session.update() {
                        self.exit_code = Some(code);
                        event_loop.exit();
                        return;
                    }
                    self.session.process_render_tasks();
                    if let Some(ref mut gpu) = self.gpu {
                        for asset_id in self.session.drain_pending_mesh_unloads() {
                            gpu.mesh_buffer_cache.remove(&asset_id);
                        }
                        if let Err(e) = render_frame(gpu, &mut self.session) {
                            log::log_write(&format!("wgpu render error: {}", e));
                        }
                    }
                }
            }
        }
    }
}

async fn init_gpu(window: &Window) -> Result<GpuState, Box<dyn std::error::Error + Send + Sync>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let surface = instance
        .create_surface(window)
        .map_err(|e| format!("create_surface: {:?}", e))?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("request_adapter: {:?}", e))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(|e| format!("request_device: {:?}", e))?;
    let size = window.inner_size();
    let mut config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
    // Fifo = vsync: blocks at present() until next vblank. Without this, we render thousands of
    // frames per second (event loop never waits; about_to_wait keeps requesting redraws).
    config.present_mode = wgpu::PresentMode::Fifo;
    surface.configure(&device, &config);
    let mesh_pipeline = gpu_mesh::MeshPipeline::new(&device, &config);
    let depth_texture = create_depth_texture(&device, &config);

    Ok(GpuState {
        surface: unsafe { std::mem::transmute(surface) },
        device,
        queue,
        config,
        mesh_pipeline,
        mesh_buffer_cache: std::collections::HashMap::new(),
        depth_texture: Some(depth_texture),
    })
}

fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

/// Clamp near/far to valid range for perspective/orthographic projection.
fn clamp_near_far(near: f32, far: f32) -> (f32, f32) {
    let near = near.max(0.001);
    let far = if far > near { far } else { near + 1.0 };
    (near, far)
}

/// Build reverse-Z perspective projection (row-major, standard formula).

/// Computes bone matrices for skinned mesh rendering.
///
/// Formula: `bone_matrix[i] = world_matrix[bone_id] * bind_pose[i]` per bone.
///
/// # Bind pose convention
///
/// The host sends bind poses via `MeshBufferGenerator` (Elements) as `bone.BindPose.Transposed`.
/// In Unity terminology, `BindPose` is the inverse of the bone's world matrix at bind time,
/// i.e. the inverse bind pose (mesh→bone). Vertices are in mesh/bind space; skinning:
/// `v_world = world_bone * inverse_bind_pose * v_mesh`. So we use `world * bind_pose` where
/// `bind_pose` is the inverse bind pose. Gloobie names this `inverse_bind_poses`; the formula
/// is equivalent.
fn compute_bone_matrices(
    scene_graph: &crate::scene::SceneGraph,
    space_id: i32,
    bone_transform_ids: &[i32],
    bind_poses: &[[[f32; 4]; 4]],
) -> Vec<[[f32; 4]; 4]> {
    let mut out = Vec::with_capacity(bone_transform_ids.len().min(bind_poses.len()));
    for (i, &tid) in bone_transform_ids.iter().enumerate() {
        let bind = bind_poses.get(i).copied().unwrap_or(nalgebra::Matrix4::identity().into());
        let bind_mat = Matrix4::from_fn(|r, c| bind[r][c]);
        let world = if tid >= 0 {
            scene_graph
                .get_world_matrix(space_id, tid as usize)
                .unwrap_or_else(Matrix4::identity)
        } else {
            Matrix4::identity()
        };
        let combined: [[f32; 4]; 4] = (world * bind_mat).into();
        out.push(combined);
    }
    out
}

/// far maps to NDC z=0, near maps to NDC z=1. Depth compare GreaterEqual, clear 0.0.
/// Row 2: z_clip = near/(far-near)*z + (far*near)/(far-near); Row 3: w_clip = -z.
/// Derives horizontal FOV from vertical FOV and aspect (Gloobie-style), clamped to avoid
/// degenerate projections at extreme aspect ratios.
fn reverse_z_projection(aspect: f32, vertical_fov: f32, near: f32, far: f32) -> Matrix4<f32> {
    let vertical_half = vertical_fov / 2.0;
    let tan_vertical_half = vertical_half.tan();
    let horizontal_fov = (tan_vertical_half * aspect)
        .atan()
        .clamp(0.1, std::f32::consts::FRAC_PI_2 - 0.1)
        * 2.0;
    let tan_horizontal_half = (horizontal_fov / 2.0).tan();
    let f_x = 1.0 / tan_horizontal_half;
    let f_y = 1.0 / tan_vertical_half;
    let proj = Matrix4::new(
        f_x, 0.0, 0.0, 0.0,
        0.0, f_y, 0.0, 0.0,
        0.0, 0.0, near / (far - near), (far * near) / (far - near),
        0.0, 0.0, -1.0, 0.0,
    );
    proj
}

/// Prevents degenerate view matrices from near-zero scale components.
fn filter_scale(scale: Vector3<f32>) -> Vector3<f32> {
    const MIN_SCALE: f32 = 1e-8;
    if scale.x.abs() < MIN_SCALE || scale.y.abs() < MIN_SCALE || scale.z.abs() < MIN_SCALE {
        Vector3::new(1.0, 1.0, 1.0)
    } else {
        scale
    }
}

/// Applies Z-axis flip so view space has -Z forward, matching projection (w_clip = -z_view).
/// Host uses +Z forward (Unity left-handed); graphics APIs typically use -Z forward.
fn apply_view_handedness_fix(view: Matrix4<f32>) -> Matrix4<f32> {
    let z_flip = Matrix4::new_nonuniform_scaling(&Vector3::new(1.0, 1.0, -1.0));
    z_flip * view
}

fn winit_key_to_renderite_key(physical_key: PhysicalKey) -> Option<Key> {
    let code = match physical_key {
        PhysicalKey::Code(c) => c,
        PhysicalKey::Unidentified(_) => return None,
    };
    Some(match code {
        KeyCode::Backspace => Key::backspace,
        KeyCode::Tab => Key::tab,
        KeyCode::Enter => Key::r#return,
        KeyCode::Escape => Key::escape,
        KeyCode::Space => Key::space,
        KeyCode::Digit0 => Key::alpha0,
        KeyCode::Digit1 => Key::alpha1,
        KeyCode::Digit2 => Key::alpha2,
        KeyCode::Digit3 => Key::alpha3,
        KeyCode::Digit4 => Key::alpha4,
        KeyCode::Digit5 => Key::alpha5,
        KeyCode::Digit6 => Key::alpha6,
        KeyCode::Digit7 => Key::alpha7,
        KeyCode::Digit8 => Key::alpha8,
        KeyCode::Digit9 => Key::alpha9,
        KeyCode::KeyA => Key::a,
        KeyCode::KeyB => Key::b,
        KeyCode::KeyC => Key::c,
        KeyCode::KeyD => Key::d,
        KeyCode::KeyE => Key::e,
        KeyCode::KeyF => Key::f,
        KeyCode::KeyG => Key::g,
        KeyCode::KeyH => Key::h,
        KeyCode::KeyI => Key::i,
        KeyCode::KeyJ => Key::j,
        KeyCode::KeyK => Key::k,
        KeyCode::KeyL => Key::l,
        KeyCode::KeyM => Key::m,
        KeyCode::KeyN => Key::n,
        KeyCode::KeyO => Key::o,
        KeyCode::KeyP => Key::p,
        KeyCode::KeyQ => Key::q,
        KeyCode::KeyR => Key::r,
        KeyCode::KeyS => Key::s,
        KeyCode::KeyT => Key::t,
        KeyCode::KeyU => Key::u,
        KeyCode::KeyV => Key::v,
        KeyCode::KeyW => Key::w,
        KeyCode::KeyX => Key::x,
        KeyCode::KeyY => Key::y,
        KeyCode::KeyZ => Key::z,
        KeyCode::BracketLeft => Key::left_bracket,
        KeyCode::Backslash => Key::backslash,
        KeyCode::BracketRight => Key::right_bracket,
        KeyCode::Minus => Key::minus,
        KeyCode::Equal => Key::equals,
        KeyCode::Backquote => Key::back_quote,
        KeyCode::Semicolon => Key::semicolon,
        KeyCode::Quote => Key::quote,
        KeyCode::Comma => Key::comma,
        KeyCode::Period => Key::period,
        KeyCode::Slash => Key::slash,
        KeyCode::Numpad0 => Key::keypad0,
        KeyCode::Numpad1 => Key::keypad1,
        KeyCode::Numpad2 => Key::keypad2,
        KeyCode::Numpad3 => Key::keypad3,
        KeyCode::Numpad4 => Key::keypad4,
        KeyCode::Numpad5 => Key::keypad5,
        KeyCode::Numpad6 => Key::keypad6,
        KeyCode::Numpad7 => Key::keypad7,
        KeyCode::Numpad8 => Key::keypad8,
        KeyCode::Numpad9 => Key::keypad9,
        KeyCode::NumpadDecimal => Key::keypad_period,
        KeyCode::NumpadDivide => Key::keypad_divide,
        KeyCode::NumpadMultiply => Key::keypad_multiply,
        KeyCode::NumpadSubtract => Key::keypad_minus,
        KeyCode::NumpadAdd => Key::keypad_plus,
        KeyCode::NumpadEnter => Key::keypad_enter,
        KeyCode::NumpadEqual => Key::keypad_equals,
        KeyCode::ArrowUp => Key::up_arrow,
        KeyCode::ArrowDown => Key::down_arrow,
        KeyCode::ArrowLeft => Key::left_arrow,
        KeyCode::ArrowRight => Key::right_arrow,
        KeyCode::Insert => Key::insert,
        KeyCode::Home => Key::home,
        KeyCode::End => Key::end,
        KeyCode::PageUp => Key::page_up,
        KeyCode::PageDown => Key::page_down,
        KeyCode::F1 => Key::f1,
        KeyCode::F2 => Key::f2,
        KeyCode::F3 => Key::f3,
        KeyCode::F4 => Key::f4,
        KeyCode::F5 => Key::f5,
        KeyCode::F6 => Key::f6,
        KeyCode::F7 => Key::f7,
        KeyCode::F8 => Key::f8,
        KeyCode::F9 => Key::f9,
        KeyCode::F10 => Key::f10,
        KeyCode::F11 => Key::f11,
        KeyCode::F12 => Key::f12,
        KeyCode::F13 => Key::f13,
        KeyCode::F14 => Key::f14,
        KeyCode::F15 => Key::f15,
        KeyCode::NumLock => Key::numlock,
        KeyCode::CapsLock => Key::caps_lock,
        KeyCode::ScrollLock => Key::scroll_lock,
        KeyCode::ShiftLeft => Key::left_shift,
        KeyCode::ShiftRight => Key::right_shift,
        KeyCode::ControlLeft => Key::left_control,
        KeyCode::ControlRight => Key::right_control,
        KeyCode::AltLeft => Key::left_alt,
        KeyCode::AltRight => Key::right_alt,
        KeyCode::SuperLeft => Key::left_windows,
        KeyCode::SuperRight => Key::right_windows,
        KeyCode::Delete => Key::delete,
        KeyCode::PrintScreen => Key::print,
        KeyCode::Pause => Key::pause,
        KeyCode::ContextMenu => Key::menu,
        _ => return None,
    })
}

/// Winding diagnostics: for first few meshes (floor-like and curved), log first triangle indices,
/// vertex positions, computed face normal, and winding hint (Y+ = top face when viewed from above).
fn log_winding_diagnostics(
    draw_batches: &[crate::session::SpaceDrawBatch],
    mesh_assets: &crate::assets::AssetRegistry,
    render_count: u64,
) {
    if render_count % log::DIAG_FRAME_INTERVAL != 0 {
        return;
    }
    use crate::shared::VertexAttributeType;
    let mut logged_meshes: std::collections::HashSet<i32> = std::collections::HashSet::new();
    let mut count = 0;
    for batch in draw_batches {
        for (model, mesh_asset_id, _, _, _) in &batch.draws {
            if *mesh_asset_id < 0 || count >= 5 {
                continue;
            }
            let Some(mesh) = mesh_assets.get_mesh(*mesh_asset_id) else { continue };
            if mesh.vertex_count < 3 || mesh.index_count < 3 || logged_meshes.contains(mesh_asset_id) {
                continue;
            }
            logged_meshes.insert(*mesh_asset_id);

            let (pos_off, _) = crate::assets::attribute_offset_and_size(&mesh.vertex_attributes, VertexAttributeType::position)
                .unwrap_or((0, 12));
            let stride = mesh.vertex_data.len() / mesh.vertex_count.max(1) as usize;
            let idx_bytes = match mesh.index_format {
                IndexBufferFormat::u_int16 => 2,
                IndexBufferFormat::u_int32 => 4,
            };

            let read_idx = |i: usize| -> Option<u32> {
                let base = i * idx_bytes;
                if base + idx_bytes > mesh.index_data.len() {
                    return None;
                }
                match mesh.index_format {
                    IndexBufferFormat::u_int16 => {
                        Some(u16::from_le_bytes(mesh.index_data[base..base + 2].try_into().ok()?) as u32)
                    }
                    IndexBufferFormat::u_int32 => {
                        Some(u32::from_le_bytes(mesh.index_data[base..base + 4].try_into().ok()?))
                    }
                }
            };
            let read_pos = |vi: u32| -> Option<(f32, f32, f32)> {
                let base = vi as usize * stride + pos_off;
                if base + 12 > mesh.vertex_data.len() {
                    return None;
                }
                let x = f32::from_le_bytes(mesh.vertex_data[base..base + 4].try_into().ok()?);
                let y = f32::from_le_bytes(mesh.vertex_data[base + 4..base + 8].try_into().ok()?);
                let z = f32::from_le_bytes(mesh.vertex_data[base + 8..base + 12].try_into().ok()?);
                Some((x, y, z))
            };

            let i0 = match read_idx(0) {
                Some(x) => x,
                None => continue,
            };
            let i1 = match read_idx(1) {
                Some(x) => x,
                None => continue,
            };
            let i2 = match read_idx(2) {
                Some(x) => x,
                None => continue,
            };
            let v0 = match read_pos(i0) {
                Some(x) => x,
                None => continue,
            };
            let v1 = match read_pos(i1) {
                Some(x) => x,
                None => continue,
            };
            let v2 = match read_pos(i2) {
                Some(x) => x,
                None => continue,
            };

            let e1 = (v1.0 - v0.0, v1.1 - v0.1, v1.2 - v0.2);
            let e2 = (v2.0 - v0.0, v2.1 - v0.1, v2.2 - v0.2);
            let nx = e1.1 * e2.2 - e1.2 * e2.1;
            let ny = e1.2 * e2.0 - e1.0 * e2.2;
            let nz = e1.0 * e2.1 - e1.1 * e2.0;
            let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-6);
            let (nx, ny, nz) = (nx / len, ny / len, nz / len);

            let ext = &mesh.bounds.extents;
            let is_floor_like = ext.y < 0.1 && ext.x > 1.0 && ext.z > 1.0;
            let winding_hint = if ny > 0.5 { "normal_up_top_face" } else if ny < -0.5 { "normal_down_bottom_face" } else { "normal_lateral" };

            log::log_write(&format!(
                "[Renderide] winding[{}]: mesh_id={} floor_like={} tri0=({},{},{}) v0=({:.2},{:.2},{:.2}) v1=({:.2},{:.2},{:.2}) v2=({:.2},{:.2},{:.2}) face_normal=({:.3},{:.3},{:.3}) {} front_face=Cw_culls_CCW",
                render_count, mesh_asset_id, is_floor_like, i0, i1, i2,
                v0.0, v0.1, v0.2, v1.0, v1.1, v1.2, v2.0, v2.1, v2.2,
                nx, ny, nz, winding_hint
            ));
            count += 1;
        }
    }
}

/// Mesh diagnostics: log draw index, mesh_id, position, vertex/index counts, bounds, is_skinned (every DIAG_FRAME_INTERVAL frames).
fn log_mesh_diagnostics(
    draw_batches: &[crate::session::SpaceDrawBatch],
    mesh_assets: &crate::assets::AssetRegistry,
    render_count: u64,
) {
    if render_count % log::DIAG_FRAME_INTERVAL != 0 {
        return;
    }
    let diag_full = std::env::var("RENDERIDE_DIAG_FULL").is_ok();
    let total_draws: usize = draw_batches.iter().map(|b| b.draws.len()).sum();
    if total_draws == 0 {
        return;
    }
    let (first_n, last_n) = if diag_full {
        (total_draws, 0)
    } else if total_draws <= 35 {
        (total_draws, 0)
    } else {
        (25, 10)
    };
    let mut global_idx: usize = 0;
    for batch in draw_batches {
        for draw in &batch.draws {
            let (model, mesh_asset_id, is_skinned, _material_id, _) = draw;
            let tx = model[(0, 3)];
            let ty = model[(1, 3)];
            let tz = model[(2, 3)];
            let (verts, indices, bounds_c, bounds_ext) = if *mesh_asset_id >= 0 {
                if let Some(mesh) = mesh_assets.get_mesh(*mesh_asset_id) {
                    (
                        mesh.vertex_count,
                        mesh.index_count,
                        format!("({:.2},{:.2},{:.2})", mesh.bounds.center.x, mesh.bounds.center.y, mesh.bounds.center.z),
                        format!("({:.2},{:.2},{:.2})", mesh.bounds.extents.x, mesh.bounds.extents.y, mesh.bounds.extents.z),
                    )
                } else {
                    (0, 0, "(?,?,?)".to_string(), "(?,?,?)".to_string())
                }
            } else {
                (0, 0, "(?,?,?)".to_string(), "(?,?,?)".to_string())
            };
            let should_log = diag_full
                || global_idx < first_n
                || (last_n > 0 && global_idx >= total_draws.saturating_sub(last_n));
            if should_log {
                log::log_write(&format!(
                    "[Renderide] mesh_diag[{}]: draw {} mesh_id={} pos=({:.2},{:.2},{:.2}) verts={} indices={} bounds_c={} ext={} skinned={}",
                    render_count,
                    global_idx,
                    mesh_asset_id,
                    tx, ty, tz,
                    verts,
                    indices,
                    bounds_c,
                    bounds_ext,
                    is_skinned
                ));
            }
            global_idx += 1;
        }
    }
}

/// Mesh assets summary: for each mesh_asset_id in draw batches, log (mesh_id, vertex_count, index_count, bounds). Include meshes with verts=0.
fn log_mesh_assets_summary(
    draw_batches: &[crate::session::SpaceDrawBatch],
    mesh_assets: &crate::assets::AssetRegistry,
    render_count: u64,
) {
    let diag_full = std::env::var("RENDERIDE_DIAG_FULL").is_ok();
    let proof_enabled = std::env::var("RENDERIDE_RENDER_PROOF").is_ok();
    if !diag_full && !proof_enabled && render_count % log::DIAG_FRAME_INTERVAL != 0 {
        return;
    }
    let mut mesh_ids: std::collections::HashSet<i32> = std::collections::HashSet::new();
    for batch in draw_batches {
        for (_, mesh_asset_id, _, _, _) in &batch.draws {
            mesh_ids.insert(*mesh_asset_id);
        }
    }
    let limit = if diag_full { usize::MAX } else { 60 };
    for (i, &mesh_id) in mesh_ids.iter().take(limit).enumerate() {
        let (verts, indices, bounds_c, bounds_ext) = if mesh_id >= 0 {
            if let Some(mesh) = mesh_assets.get_mesh(mesh_id) {
                (
                    mesh.vertex_count,
                    mesh.index_count,
                    format!("({:.2},{:.2},{:.2})", mesh.bounds.center.x, mesh.bounds.center.y, mesh.bounds.center.z),
                    format!("({:.2},{:.2},{:.2})", mesh.bounds.extents.x, mesh.bounds.extents.y, mesh.bounds.extents.z),
                )
            } else {
                (-1, -1, "(missing)".to_string(), "(missing)".to_string())
            }
        } else {
            (-1, -1, "(negative_id)".to_string(), "(negative_id)".to_string())
        };
        log::log_write(&format!(
            "[Renderide] mesh_assets_summary[{}]: mesh_id={} verts={} indices={} bounds_c={} ext={}",
            i, mesh_id, verts, indices, bounds_c, bounds_ext
        ));
    }
    if mesh_ids.len() > limit {
        log::log_write(&format!(
            "[Renderide] mesh_assets_summary: ... and {} more (total {} unique mesh_ids)",
            mesh_ids.len() - limit, mesh_ids.len()
        ));
    }
}

/// Mesh render summary: vertex/index totals, positions, bounds for first N draws.
/// Throttled every 30 frames. Helps debug empty buffers, extreme positions, wrong bounds.
fn log_mesh_render_summary(
    draw_batches: &[crate::session::SpaceDrawBatch],
    mesh_assets: &crate::assets::AssetRegistry,
    frame: u64,
) {
    if frame % 30 != 0 {
        return;
    }
    let total_draws: usize = draw_batches.iter().map(|b| b.draws.len()).sum();
    let mut total_verts: i64 = 0;
    let mut total_indices: i64 = 0;
    for batch in draw_batches {
        for (_, mesh_asset_id, _, _, _) in &batch.draws {
            if *mesh_asset_id >= 0 {
                if let Some(mesh) = mesh_assets.get_mesh(*mesh_asset_id) {
                    total_verts += mesh.vertex_count as i64;
                    total_indices += mesh.index_count as i64;
                }
            }
        }
    }
    log::log_write(&format!(
        "[MESH RENDER] frame {} batches={} total_draws={} mesh_assets={} total_verts={} total_indices={}",
        frame,
        draw_batches.len(),
        total_draws,
        mesh_assets.mesh_count(),
        total_verts,
        total_indices,
    ));
    let mut draw_idx: usize = 0;
    let sample_limit = 15;
    for batch in draw_batches {
        for (model, mesh_asset_id, _, _, _) in &batch.draws {
            if draw_idx >= sample_limit {
                break;
            }
            let tx = model[(0, 3)];
            let ty = model[(1, 3)];
            let tz = model[(2, 3)];
            let (verts, indices, bounds_c, bounds_ext) = if *mesh_asset_id >= 0 {
                if let Some(mesh) = mesh_assets.get_mesh(*mesh_asset_id) {
                    (
                        mesh.vertex_count,
                        mesh.index_count,
                        format!("({:.2},{:.2},{:.2})", mesh.bounds.center.x, mesh.bounds.center.y, mesh.bounds.center.z),
                        format!("({:.2},{:.2},{:.2})", mesh.bounds.extents.x, mesh.bounds.extents.y, mesh.bounds.extents.z),
                    )
                } else {
                    (0, 0, "(missing)".to_string(), "(missing)".to_string())
                }
            } else {
                (0, 0, "(neg_id)".to_string(), "(neg_id)".to_string())
            };
            log::log_write(&format!(
                "[MESH RENDER] draw {} mesh_id={} pos=({:.2},{:.2},{:.2}) bounds_c={} ext={} verts={} indices={}",
                draw_idx, mesh_asset_id, tx, ty, tz, bounds_c, bounds_ext, verts, indices
            ));
            draw_idx += 1;
        }
        if draw_idx >= sample_limit {
            break;
        }
    }
}

/// Render proof: mathematically verify why geometry may not show. Logs view_proj, MVP, clip-space,
/// NDC, frustum checks, pipeline config. Run when RENDERIDE_RENDER_PROOF=1 or every DIAG_FRAME_INTERVAL.
fn log_render_proof(
    draw_batches: &[crate::session::SpaceDrawBatch],
    mesh_assets: &crate::assets::AssetRegistry,
    primary_camera: Option<&CameraRenderTask>,
    viewport_width: u32,
    viewport_height: u32,
    near: f32,
    far: f32,
    fov: f32,
    render_count: u64,
) {
    let proof_enabled = std::env::var("RENDERIDE_RENDER_PROOF").is_ok();
    let diag_full = std::env::var("RENDERIDE_DIAG_FULL").is_ok();
    if !proof_enabled && !diag_full {
        return;
    }
    let total_draws: usize = draw_batches.iter().map(|b| b.draws.len()).sum();
    if total_draws == 0 {
        log::log_write("[Renderide] render_proof: no draws, nothing to prove");
        return;
    }

    log::log_write(&format!(
        "[Renderide] render_proof[{}]: viewport={}x{} near={} far={} fov={}",
        render_count, viewport_width, viewport_height, near, far, fov
    ));
    log::log_write("[Renderide] render_proof: depth_clear=0.0 depth_compare=GreaterEqual depth_format=Depth24Plus front_face=Cw cull=Back");

    let mut draw_idx: usize = 0;
    let mut in_ndc_count: usize = 0;
    let mut behind_cam_count: usize = 0;
    let mut head_height_sample_logged: bool = false;
    for batch in draw_batches {
        let view_proj = if let Some(cam) = primary_camera {
            camera_view_proj(cam, viewport_width, viewport_height)
        } else {
            let mut vt = batch.view_transform;
            if vt.position.y.abs() < 0.2 || vt.position.y > 1.5 {
                vt.position.y = 1.0; // eye height for desktop mode
            }
            view_transform_to_view_proj(&vt, viewport_width, viewport_height, near, far, fov)
        };
        let view_pos = if let Some(cam) = primary_camera {
            &cam.position
        } else {
            &batch.view_transform.position
        };
        if draw_idx == 0 {
            log::log_write(&format!(
                "[Renderide] render_proof: batch space_id={} view_pos=({:.2},{:.2},{:.2}) view_proj row0={:.4},{:.4},{:.4},{:.4}",
                batch.space_id,
                view_pos.x, view_pos.y, view_pos.z,
                view_proj[(0,0)], view_proj[(0,1)], view_proj[(0,2)], view_proj[(0,3)]
            ));
            log::log_write(&format!(
                "[Renderide] render_proof: view_proj full matrix:\n  row0=({:.4},{:.4},{:.4},{:.4})\n  row1=({:.4},{:.4},{:.4},{:.4})\n  row2=({:.4},{:.4},{:.4},{:.4})\n  row3=({:.4},{:.4},{:.4},{:.4})",
                view_proj[(0,0)], view_proj[(0,1)], view_proj[(0,2)], view_proj[(0,3)],
                view_proj[(1,0)], view_proj[(1,1)], view_proj[(1,2)], view_proj[(1,3)],
                view_proj[(2,0)], view_proj[(2,1)], view_proj[(2,2)], view_proj[(2,3)],
                view_proj[(3,0)], view_proj[(3,1)], view_proj[(3,2)], view_proj[(3,3)]
            ));
        }

        for (model, mesh_asset_id, _, _, _) in &batch.draws {
            let world_pos = Vector3::new(model[(0, 3)], model[(1, 3)], model[(2, 3)]);
            let mvp = view_proj * model;

            let clip_pos = mvp * nalgebra::Vector4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
            let ndc_w = clip_pos.w;
            let ndc_x = if ndc_w.abs() > 1e-6 { clip_pos.x / ndc_w } else { f32::NAN };
            let ndc_y = if ndc_w.abs() > 1e-6 { clip_pos.y / ndc_w } else { f32::NAN };
            let ndc_z = if ndc_w.abs() > 1e-6 { clip_pos.z / ndc_w } else { f32::NAN };

            let in_ndc_x = ndc_x >= -1.0 && ndc_x <= 1.0 && ndc_x.is_finite();
            let in_ndc_y = ndc_y >= -1.0 && ndc_y <= 1.0 && ndc_y.is_finite();
            // Standard projection (nalgebra Perspective3) produces NDC z in [-1, 1], not [0, 1].
            let in_ndc_z = ndc_z >= -1.0 && ndc_z <= 1.0 && ndc_z.is_finite();
            let in_ndc = in_ndc_x && in_ndc_y && in_ndc_z;

            let behind_camera = ndc_w <= 0.0;
            if in_ndc { in_ndc_count += 1; }
            if behind_camera { behind_cam_count += 1; }

            let vert_count = mesh_assets.get_mesh(*mesh_asset_id).map(|m| m.vertex_count).unwrap_or(-1);

            let near_head_height = world_pos.y >= 1.4 && world_pos.y <= 2.2;
            if near_head_height && !head_height_sample_logged {
                log::log_write(&format!(
                    "[Renderide] render_proof: head_height_sample draw[{}] mesh_id={} world=({:.2},{:.2},{:.2}) clip=({:.2},{:.2},{:.2},{:.2}) ndc=({:.3},{:.3},{:.3}) in_ndc={} behind_cam={} verts={}",
                    draw_idx, mesh_asset_id,
                    world_pos.x, world_pos.y, world_pos.z,
                    clip_pos.x, clip_pos.y, clip_pos.z, clip_pos.w,
                    ndc_x, ndc_y, ndc_z,
                    in_ndc, behind_camera, vert_count
                ));
                head_height_sample_logged = true;
            }

            let should_log_detail = proof_enabled || (diag_full && draw_idx < 5);
            if should_log_detail {
                log::log_write(&format!(
                    "[Renderide] render_proof: draw[{}] mesh_id={} world=({:.2},{:.2},{:.2}) clip=({:.2},{:.2},{:.2},{:.2}) ndc=({:.3},{:.3},{:.3}) in_ndc={} behind_cam={} verts={}",
                    draw_idx, mesh_asset_id,
                    world_pos.x, world_pos.y, world_pos.z,
                    clip_pos.x, clip_pos.y, clip_pos.z, clip_pos.w,
                    ndc_x, ndc_y, ndc_z,
                    in_ndc, behind_camera, vert_count
                ));

                if draw_idx == 0 && vert_count > 0 {
                    if let Some(mesh) = mesh_assets.get_mesh(*mesh_asset_id) {
                        let first_vert = crate::assets::attribute_offset_and_size(&mesh.vertex_attributes, VertexAttributeType::position)
                            .and_then(|(pos_off, pos_size)| {
                                if pos_size >= 12 && mesh.vertex_data.len() >= pos_off + 12 {
                                    let px = f32::from_le_bytes(mesh.vertex_data[pos_off..pos_off+4].try_into().unwrap_or([0;4]));
                                    let py = f32::from_le_bytes(mesh.vertex_data[pos_off+4..pos_off+8].try_into().unwrap_or([0;4]));
                                    let pz = f32::from_le_bytes(mesh.vertex_data[pos_off+8..pos_off+12].try_into().unwrap_or([0;4]));
                                    Some(format!("local_first_vert=({:.2},{:.2},{:.2})", px, py, pz))
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_else(|| "local_first_vert=(no_pos_attr)".to_string());
                        log::log_write(&format!("[Renderide] render_proof: draw[0] {}", first_vert));
                        // Per-vertex NDC for first draw: verify all vertices are in [-1,1]^3
                        if let Some((pos_off, pos_size)) = crate::assets::attribute_offset_and_size(&mesh.vertex_attributes, VertexAttributeType::position) {
                            if pos_size >= 12 {
                                let stride = mesh.vertex_data.len() / mesh.vertex_count.max(1) as usize;
                                let vert_limit = proof_enabled.then_some(mesh.vertex_count as usize).unwrap_or(8.min(mesh.vertex_count as usize));
                                for vi in 0..vert_limit {
                                    let base = vi * stride + pos_off;
                                    if base + 12 <= mesh.vertex_data.len() {
                                        let lx = f32::from_le_bytes(mesh.vertex_data[base..base+4].try_into().unwrap_or([0;4]));
                                        let ly = f32::from_le_bytes(mesh.vertex_data[base+4..base+8].try_into().unwrap_or([0;4]));
                                        let lz = f32::from_le_bytes(mesh.vertex_data[base+8..base+12].try_into().unwrap_or([0;4]));
                                        let world = model * nalgebra::Vector4::new(lx, ly, lz, 1.0);
                                        let clip = mvp * nalgebra::Vector4::new(lx, ly, lz, 1.0);
                                        let (ndc_x, ndc_y, ndc_z) = if clip.w.abs() > 1e-6 {
                                            (clip.x / clip.w, clip.y / clip.w, clip.z / clip.w)
                                        } else {
                                            (f32::NAN, f32::NAN, f32::NAN)
                                        };
                                        let in_cube = ndc_x >= -1.0 && ndc_x <= 1.0 && ndc_y >= -1.0 && ndc_y <= 1.0 && ndc_z >= -1.0 && ndc_z <= 1.0 && ndc_x.is_finite();
                                        log::log_write(&format!(
                                            "[Renderide] render_proof: draw[0] vert[{}] local=({:.2},{:.2},{:.2}) world=({:.2},{:.2},{:.2}) clip_w={:.4} ndc=({:.3},{:.3},{:.3}) in_ndc_cube={}",
                                            vi, lx, ly, lz, world.x, world.y, world.z, clip.w, ndc_x, ndc_y, ndc_z, in_cube
                                        ));
                                    }
                                }
                                if mesh.vertex_count as usize > vert_limit {
                                    log::log_write(&format!("[Renderide] render_proof: draw[0] ... and {} more vertices (total {})", mesh.vertex_count as usize - vert_limit, mesh.vertex_count));
                                }
                            }
                        }
                    }
                }
            }

            draw_idx += 1;
        }
    }

    log::log_write(&format!(
        "[Renderide] render_proof: summary: total_draws={} in_ndc_count={} (should_be_visible) behind_cam_count={} (culled) in_ndc means visible in viewport; behind_cam=true means culled",
        total_draws, in_ndc_count, behind_cam_count
    ));
}

/// Build view-projection matrix from CameraRenderTask.
/// Host uses +Z forward (Unity left-handed); we use look_at_lh then apply handedness fix.
fn camera_view_proj(
    task: &CameraRenderTask,
    viewport_width: u32,
    viewport_height: u32,
) -> Matrix4<f32> {
    let unit_rot = UnitQuaternion::try_new(task.rotation, 1e-8)
        .unwrap_or_else(UnitQuaternion::identity);
    let forward = unit_rot.transform_vector(&Vector3::new(0.0, 0.0, 1.0));
    let up = unit_rot.transform_vector(&Vector3::new(0.0, 1.0, 0.0));
    let target = Point3::from(task.position) + forward;
    let view_mat = Matrix4::look_at_lh(
        &Point3::from(task.position),
        &target,
        &up,
    );
    let view_mat = apply_view_handedness_fix(view_mat);

    let (width, height) = (viewport_width as f32, viewport_height as f32);
    let aspect = width / height.max(1.0);

    let proj_mat = match task.parameters.as_ref() {
        Some(params) => {
            let (near, far) = clamp_near_far(params.near_clip, params.far_clip);
            match params.projection {
                CameraProjection::perspective => {
                    let fov = params.fov.to_radians();
                    reverse_z_projection(aspect, fov, near, far)
                }
                CameraProjection::orthographic => {
                    let half_h = params.orthographic_size * 0.5;
                    let half_w = half_h * aspect;
                    Orthographic3::new(-half_w, half_w, -half_h, half_h, near, far).to_homogeneous()
                }
                CameraProjection::panoramic => {
                    reverse_z_projection(aspect, params.fov.to_radians(), near, far)
                }
            }
        }
        None => reverse_z_projection(aspect, 75.0f32.to_radians(), 0.01, 1024.0),
    };

    proj_mat * view_mat
}

/// Build view-projection matrix from RenderTransform (main display view from active render space).
fn view_transform_to_view_proj(
    transform: &RenderTransform,
    viewport_width: u32,
    viewport_height: u32,
    near: f32,
    far: f32,
    fov_deg: f32,
) -> Matrix4<f32> {
    let (near, far) = clamp_near_far(near, far);

    let view_mat = crate::core::render_transform_to_matrix(transform)
        .try_inverse()
        .unwrap_or_else(Matrix4::identity);
    let view_mat = apply_view_handedness_fix(view_mat);

    let (width, height) = (viewport_width as f32, viewport_height as f32);
    let aspect = width / height.max(1.0);
    let vertical_fov = fov_deg.to_radians();

    let proj_mat = reverse_z_projection(aspect, vertical_fov, near, far);

    proj_mat * view_mat
}

fn render_frame(
    gpu: &mut GpuState,
    session: &mut Session,
) -> Result<(), wgpu::SurfaceError> {
    let draw_batches = session.collect_draw_batches();
    let draw_data_len: usize = draw_batches.iter().map(|b| b.draws.len()).sum();
    unsafe { RENDER_COUNT += 1; }
    let frame = unsafe { RENDER_COUNT };

    // Gated: drowning signal — re-enable when debugging draw list
    // if frame % log::DIAG_FRAME_INTERVAL == 0 {
    //     log::log_write(&format!(
    //         "[FINAL DRAW LIST] frame {} | batches_created={} | total_draws_across_all_spaces={}",
    //         frame, draw_batches.len(), draw_data_len
    //     ));
    // }

    // Gated: keep only PERSISTENT NDC TEST, FORCED CUBE NDC PROOF, VIEW PROOF, VIEW MATRIX
    // let using_camera = rendering_manager.primary_camera_task().is_some();
    // let view_src = if using_camera { "CameraRenderTask" } else { "FallbackViewTransform" };
    // let should_log_math = draw_data_len > 0 && frame % log::DIAG_FRAME_INTERVAL == 0;
    // if should_log_math {
    //     if let Some(batch) = draw_batches.first() {
    //         let vt = &batch.view_transform;
    //         let forward = UnitQuaternion::new_normalize(vt.rotation)
    //             .transform_vector(&Vector3::new(0.0, 0.0, -1.0));
    //         log::log_write(&format!(
    //             "[MATH] frame {} | host_frame={} | src={} | pos=({:.3},{:.3},{:.3}) | rot=({:.4},{:.4},{:.4},{:.4}) | fwd=({:.3},{:.3},{:.3})",
    //             frame, rendering_manager.last_frame_index, view_src,
    //             vt.position.x, vt.position.y, vt.position.z,
    //             vt.rotation.i, vt.rotation.j, vt.rotation.k, vt.rotation.w,
    //             forward.x, forward.y, forward.z
    //         ));
    //     }
    // }
    // if frame % log::DIAG_FRAME_INTERVAL == 0 && draw_data_len > 0 {
    //     for batch in &draw_batches {
    //         log::log_write(&format!(
    //             "[Renderide] render_frame: batch space_id={} draws={}",
    //             batch.space_id,
    //             batch.draws.len()
    //         ));
    //     }
    //     log::log_write(&format!(
    //         "[Renderide] render_frame: draw_count={} batches={} has_camera={} mesh_assets={}",
    //         draw_data_len,
    //         draw_batches.len(),
    //         rendering_manager.primary_camera_task().is_some(),
    //         rendering_manager.mesh_assets().len()
    //     ));
    // }

    let output = gpu.surface.get_current_texture()?;
    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let mesh_assets = session.asset_registry();
    log_mesh_render_summary(&draw_batches, mesh_assets, frame);
    let diag_full = std::env::var("RENDERIDE_DIAG_FULL").is_ok();
    if diag_full {
        log_mesh_diagnostics(&draw_batches, mesh_assets, unsafe { RENDER_COUNT });
        log_mesh_assets_summary(&draw_batches, mesh_assets, unsafe { RENDER_COUNT });
        log_winding_diagnostics(&draw_batches, mesh_assets, unsafe { RENDER_COUNT });
    }
    // Gated: keep only forced-cube NDC proof for now
    // log_render_proof(
    //     &draw_batches,
    //     mesh_assets,
    //     rendering_manager.primary_camera_task(),
    //     gpu.config.width,
    //     gpu.config.height,
    //     rendering_manager.near_clip,
    //     rendering_manager.far_clip,
    //     rendering_manager.desktop_fov,
    //     unsafe { RENDER_COUNT },
    // );

    let depth_view = gpu
        .depth_texture
        .as_ref()
        .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));

    // Gated: keep only key proofs
    // if unsafe { RENDER_COUNT } % log::DIAG_FRAME_INTERVAL == 0 && draw_data_len > 0 {
    //     log::log_write(&format!(
    //         "[Renderide] render_frame: surface={}x{} depth_attachment={}",
    //         gpu.config.width, gpu.config.height,
    //         if depth_view.is_some() { "present" } else { "MISSING" }
    //     ));
    // }

    // Use primary view for logging; actual rendering uses per-batch view (see below).
    let mut view_transform = session
        .primary_view_transform()
        .cloned()
        .unwrap_or_default();
    view_transform.scale = filter_scale(view_transform.scale);

    let aspect = gpu.config.width as f32 / gpu.config.height.max(1) as f32;
    let proj = reverse_z_projection(
        aspect,
        session.desktop_fov().to_radians(),
        session.near_clip().max(0.01),
        session.far_clip(),
    );

    if frame % 30 == 0 {
        let q = UnitQuaternion::try_new(view_transform.rotation, 1e-8)
            .unwrap_or_else(UnitQuaternion::identity);
        let fwd = q.transform_vector(&Vector3::new(0.0, 0.0, -1.0));
        let space_id = session.primary_view_space_id().unwrap_or(-1);
        let override_pos = session.primary_view_override().unwrap_or(false);
        let view_ext = session.primary_view_position_is_external().unwrap_or(false);
        log::log_write(&format!(
            "[HOST CAMERA] space_id={} override_view={} view_position_is_external={} pos=({:.2},{:.2},{:.2}) fwd=({:.3},{:.3},{:.3})",
            space_id,
            override_pos,
            view_ext,
            view_transform.position.x,
            view_transform.position.y,
            view_transform.position.z,
            fwd.x,
            fwd.y,
            fwd.z
        ));
        if let Some(root) = session.primary_root_transform() {
            let root_match = (view_transform.position.x - root.position.x).abs() < 1e-4
                && (view_transform.position.y - root.position.y).abs() < 1e-4
                && (view_transform.position.z - root.position.z).abs() < 1e-4;
            if !root_match {
                log::log_write(&format!(
                    "[HOST CAMERA] view vs root differ: view_pos=({:.2},{:.2},{:.2}) root_pos=({:.2},{:.2},{:.2})",
                    view_transform.position.x,
                    view_transform.position.y,
                    view_transform.position.z,
                    root.position.x,
                    root.position.y,
                    root.position.z
                ));
            }
        }
        // Phase 4: When view_position_is_external is true and we use root (override=false, e.g. VR),
        // the view may need to come from input/head state. We use root for now; future VR support
        // may need head pose from output_state.vr or similar.
        if view_ext && !override_pos {
            log::log_write(
                "[HOST CAMERA] view_position_is_external=true with override=false: using root for view; future VR may need head pose from output_state",
            );
        }
    }

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.8,
                        b: 0.0,
                        a: 1.0,
                    }), // bright green background (confirmation)
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth_view.as_ref().map(|dv| wgpu::RenderPassDepthStencilAttachment {
                view: dv,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0), // reverse-Z: 0 = far
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let use_debug_uv = std::env::var("RENDERIDE_DEBUG_UV").is_ok();
        let diag_full = std::env::var("RENDERIDE_DIAG_FULL").is_ok();

        // Debug: track vertex buffer sizes and skip reasons
        let mut total_vertex_bytes: u64 = 0;
        let mut total_index_count: u32 = 0;
        let mut draws_submitted: usize = 0;
        let mut skipped_no_mesh_asset: usize = 0;
        let mut skipped_negative_asset: usize = 0;
        let mut skipped_bad_vertex_count: usize = 0;
        let mut skipped_bad_index_count: usize = 0;
        let mut skipped_buffer_cache_miss: usize = 0;
        let mut skip_reasons: std::collections::HashMap<i32, &'static str> =
            std::collections::HashMap::new();
        let mut floor_in_frustum = false;
        let mut skinned_path_taken: usize = 0;
        let mut skinned_fallback_missing_data: usize = 0;

        let render_count = unsafe { RENDER_COUNT };

        // Pass 1: ensure all mesh buffers are cached (insert if missing).
        for batch in &draw_batches {
            for (model, mesh_asset_id, _is_skinned, _material_id, _) in &batch.draws {
                if *mesh_asset_id < 0 {
                    continue;
                }
                let Some(mesh) = mesh_assets.get_mesh(*mesh_asset_id) else {
                    continue;
                };
                if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                    continue;
                }
                if !gpu.mesh_buffer_cache.contains_key(mesh_asset_id) {
                    let stride = crate::assets::compute_vertex_stride(&mesh.vertex_attributes) as usize;
                    let stride = if stride > 0 {
                        stride
                    } else {
                        gpu_mesh::compute_vertex_stride_from_mesh(mesh)
                    };
                    if let Some(b) = gpu_mesh::create_mesh_buffers(&gpu.device, mesh, stride) {
                        gpu.mesh_buffer_cache.insert(*mesh_asset_id, b);
                    }
                }
            }
        }

        // Skinned draw proof: log counts every 30 frames.
        if draw_data_len > 0 && frame % 30 == 0 {
            let skinned_total: usize = draw_batches
                .iter()
                .flat_map(|b| &b.draws)
                .filter(|(_, _, is_skinned, _, _)| *is_skinned)
                .count();
            let skinned_with_bones: usize = draw_batches
                .iter()
                .flat_map(|b| &b.draws)
                .filter(|(_, _, is_skinned, _, bone_ids)| *is_skinned && bone_ids.is_some())
                .count();
            log::log_write(&format!(
                "[SKINNED PROOF] frame {} skinned_draws_in_batch={} skinned_with_bone_ids={} (need both for skinned path)",
                frame, skinned_total, skinned_with_bones
            ));
        }

        // Pass 2: collect draws and uniforms. Skinned draws use a separate path with bone matrices.
        struct BatchedDraw<'a> {
            vertex_buffer: &'a wgpu::Buffer,
            index_buffer: &'a wgpu::Buffer,
            submeshes: &'a [(u32, u32)],
            index_format: wgpu::IndexFormat,
            use_debug_uv: bool,
            has_uvs: bool,
            is_skinned: bool,
            material_id: i32,
        }
        let mut batched_draws: Vec<BatchedDraw<'_>> = Vec::new();
        let mut mvp_models: Vec<(Matrix4<f32>, Matrix4<f32>)> = Vec::new();
        let scene_graph = session.scene_graph();
        let viewport_w = gpu.config.width;
        let viewport_h = gpu.config.height.max(1);
        let near = session.near_clip().max(0.01);
        let far = session.far_clip();
        let fov = session.desktop_fov();

        for batch in &draw_batches {
            // Use this batch's view transform so each space uses its own camera (fixes movement
            // when primary view came from a different space).
            let mut batch_vt = batch.view_transform;
            batch_vt.scale = filter_scale(batch_vt.scale);
            let view_mat = crate::core::render_transform_to_matrix(&batch_vt)
                .try_inverse()
                .unwrap_or_else(Matrix4::identity);
            let view_mat = apply_view_handedness_fix(view_mat);
            let view_proj = proj * view_mat;

            for (model, mesh_asset_id, is_skinned, material_id, bone_transform_ids) in &batch.draws {
                let buffers_ref = if *mesh_asset_id >= 0 {
                    let Some(mesh) = mesh_assets.get_mesh(*mesh_asset_id) else {
                        skipped_no_mesh_asset += 1;
                        skip_reasons
                            .entry(*mesh_asset_id)
                            .or_insert("no_mesh_asset");
                        if skipped_no_mesh_asset == 1 {
                            log::log_write(&format!(
                                "[MESH RENDER] missing mesh asset_id={} (drawable references mesh not yet uploaded)",
                                mesh_asset_id
                            ));
                        }
                        continue;
                    };
                    if mesh.vertex_count <= 0 {
                        skipped_bad_vertex_count += 1;
                        skip_reasons
                            .entry(*mesh_asset_id)
                            .or_insert("bad_vertex_count");
                        continue;
                    }
                    if mesh.index_count <= 0 {
                        skipped_bad_index_count += 1;
                        skip_reasons
                            .entry(*mesh_asset_id)
                            .or_insert("bad_index_count");
                        continue;
                    }
                    let Some(b) = gpu.mesh_buffer_cache.get(mesh_asset_id) else {
                        skipped_buffer_cache_miss += 1;
                        skip_reasons
                            .entry(*mesh_asset_id)
                            .or_insert("buffer_cache_miss");
                        if skipped_buffer_cache_miss == 1 {
                            log::log_write(&format!(
                                "[MESH RENDER] buffer_cache_miss asset_id={} (mesh exists but GPU buffers not created)",
                                mesh_asset_id
                            ));
                        }
                        continue;
                    };
                    (b, mesh)
                } else {
                    skipped_negative_asset += 1;
                    skip_reasons
                        .entry(*mesh_asset_id)
                        .or_insert("negative_asset_id");
                    continue;
                };

                let (buffers_ref, mesh) = buffers_ref;

                total_vertex_bytes += buffers_ref.vertex_buffer.size();
                let draw_count: u32 = buffers_ref.submeshes.iter().map(|(_, c)| *c).sum();
                total_index_count += draw_count;
                draws_submitted += 1;

                let model_mvp = view_proj * model;
                let skinned_mvp = view_proj;

                // Skinned path: use vertex_buffer_skinned and compute bone matrices.
                // Bone matrices already produce world-space vertices; use view_proj only (no model).
                if *is_skinned {
                    let has_vb = buffers_ref.vertex_buffer_skinned.is_some();
                    let has_bind_poses = mesh.bind_poses.is_some();
                    let has_bone_ids = bone_transform_ids.is_some();
                    if has_vb && has_bind_poses && has_bone_ids {
                        let vb_skinned = buffers_ref.vertex_buffer_skinned.as_ref().unwrap();
                        let bind_poses = mesh.bind_poses.as_ref().unwrap();
                        let ids = bone_transform_ids.as_ref().unwrap();
                        let bone_matrices = compute_bone_matrices(
                            scene_graph,
                            batch.space_id,
                            ids,
                            bind_poses,
                        );
                        gpu.mesh_pipeline.upload_skinned_uniforms(&gpu.queue, skinned_mvp, &bone_matrices);
                        gpu.mesh_pipeline.draw_mesh_skinned(
                            &mut pass,
                            vb_skinned.as_ref(),
                            buffers_ref.index_buffer.as_ref(),
                            &buffers_ref.submeshes,
                            buffers_ref.index_format,
                        );
                        skinned_path_taken += 1;
                        if frame % 30 == 0 && skinned_path_taken <= 3 {
                            log::log_write(&format!(
                                "[SKINNED PATH] mesh_id={} bone_count={} (skinned draw taken)",
                                mesh_asset_id,
                                ids.len()
                            ));
                        }
                        if std::env::var("RENDERIDE_DIAG_FULL").is_ok()
                            && frame % 30 == 0
                            && skinned_path_taken == 1
                            && !bone_matrices.is_empty()
                        {
                            let b0 = &bone_matrices[0];
                            log::log_write(&format!(
                                "[SKINNED BONE DIAG] bone[0] matrix translation=({:.3},{:.3},{:.3}) (verify bone_matrix=world*bind_pose)",
                                b0[0][3], b0[1][3], b0[2][3]
                            ));
                        }
                        continue;
                    } else {
                        skinned_fallback_missing_data += 1;
                        let reasons: Vec<&str> = [
                            (!has_vb).then_some("no_vb_skinned"),
                            (!has_bind_poses).then_some("no_bind_poses"),
                            (!has_bone_ids).then_some("no_bone_ids"),
                        ]
                        .into_iter()
                        .flatten()
                        .collect();
                        if frame % 30 == 0 && skinned_fallback_missing_data <= 10 {
                            log::log_write(&format!(
                                "[SKINNED FALLBACK] mesh_id={} mesh.bone_count={} reasons=[{}] (rendered as rigid)",
                                mesh_asset_id,
                                mesh.bone_count,
                                reasons.join(", ")
                            ));
                        }
                    }
                }

                // Non-skinned (or skinned fallback when bone data missing): batched path.
                let vb = if use_debug_uv && buffers_ref.has_uvs && !*is_skinned {
                    buffers_ref.vertex_buffer_uv.as_ref().map(|b| b.as_ref()).unwrap_or(buffers_ref.vertex_buffer.as_ref())
                } else {
                    buffers_ref.vertex_buffer.as_ref()
                };

                mvp_models.push((model_mvp, *model));
                batched_draws.push(BatchedDraw {
                    vertex_buffer: vb,
                    index_buffer: buffers_ref.index_buffer.as_ref(),
                    submeshes: &buffers_ref.submeshes,
                    index_format: buffers_ref.index_format,
                    use_debug_uv,
                    has_uvs: buffers_ref.has_uvs,
                    is_skinned: false, // batched path always uses normal pipeline
                    material_id: *material_id,
                });
            }
        }

        // Single batched upload for non-skinned, then draw with dynamic offsets.
        gpu.mesh_pipeline.upload_uniforms_batch(&gpu.queue, &mvp_models);
        for (i, d) in batched_draws.iter().enumerate() {
            gpu.mesh_pipeline.draw_mesh_with_offset(
                &mut pass,
                d.vertex_buffer,
                d.index_buffer,
                d.submeshes,
                d.index_format,
                i as u32,
                d.use_debug_uv,
                d.has_uvs,
                d.is_skinned,
                d.material_id,
            );
        }

        // Mesh render diagnostics: skip counts and draw summary (throttled every 30 frames).
        if draw_data_len > 0 && frame % 30 == 0 {
            let total_skipped = skipped_no_mesh_asset
                + skipped_negative_asset
                + skipped_bad_vertex_count
                + skipped_bad_index_count
                + skipped_buffer_cache_miss;
            log::log_write(&format!(
                "[MESH RENDER] vertex_buffers total_bytes={} total_index_count={} draws_submitted={} draw_data_len={} total_skipped={} (no_mesh_asset={} negative_asset={} bad_vertex_count={} bad_index_count={} buffer_cache_miss={}) skinned_path={} skinned_fallback_missing_data={}",
                total_vertex_bytes,
                total_index_count,
                draws_submitted,
                draw_data_len,
                total_skipped,
                skipped_no_mesh_asset,
                skipped_negative_asset,
                skipped_bad_vertex_count,
                skipped_bad_index_count,
                skipped_buffer_cache_miss,
                skinned_path_taken,
                skinned_fallback_missing_data
            ));
            if !skip_reasons.is_empty() {
                let sample: Vec<String> = skip_reasons
                    .iter()
                    .take(15)
                    .map(|(id, reason)| format!("{}:{}", id, reason))
                    .collect();
                log::log_write(&format!(
                    "[MESH RENDER] skip_reasons (mesh_id:reason) sample={:?}",
                    sample
                ));
            }
        }

        // Gated: drowning signal
        // if (diag_full || frame % log::DIAG_FRAME_INTERVAL_THROTTLE == 0) && (draws_submitted == 0 || draw_data_len > 0) {
        //     log::log_write(&format!(
        //         "[MATH SUMMARY] frame {} | draws_submitted={} | total_draws={} | using_camera={}",
        //         frame, draws_submitted, draw_data_len, using_camera
        //     ));
        // }
        // if frame % log::DIAG_FRAME_INTERVAL_THROTTLE == 0 {
        //     let first_batch = draw_batches.first();
        //     let (cam_y, fwd_y) = first_batch
        //         .map(|b| (b.view_transform.position.y, batch_forward.y))
        //         .unwrap_or((0.0, 0.0));
        //     log::log_write(&format!(
        //         "[VISIBILITY SUMMARY] frame={} draws_submitted={} floor_in_frustum={} camera_y={:.2} forward_y={:.3}",
        //         frame, draws_submitted, floor_in_frustum, cam_y, fwd_y
        //     ));
        // }
    }

    gpu.queue.submit(std::iter::once(encoder.finish()));
    output.present();
    Ok(())
}
