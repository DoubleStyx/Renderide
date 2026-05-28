//! Renderer launch command handling.

use std::process::{Child, Command};

use interprocess::Publisher;

use crate::child_lifetime::ChildLifetimeGroup;
use crate::config::ResoBootConfig;
use crate::process_state::SharedChildSlot;
use crate::protocol::LoopAction;
use crate::renderer_link;

/// Spawns the renderer and registers it for lifetime and exit-watchdog management.
///
/// If a renderer was already registered, the previous process is killed and reaped first.
pub(super) fn handle_start_renderer(
    renderer_args: &[String],
    outgoing: &mut Publisher,
    config: &ResoBootConfig,
    lifetime: &ChildLifetimeGroup,
    renderer_child: &SharedChildSlot,
) -> LoopAction {
    let mut args: Vec<String> = renderer_args.to_vec();
    if let Some(ref level) = config.renderide_log_level {
        args.push("-LogLevel".to_string());
        args.push(level.as_arg().to_string());
    }
    let args_refs: Vec<&str> = args.iter().map(String::as_str).collect();

    renderer_link::ensure_link(config);

    logger::info!(
        "Spawning renderer: exe={} cwd={} args={:?}",
        config.renderite_executable.display(),
        config.renderite_directory.display(),
        args
    );
    let mut renderer_cmd = Command::new(&config.renderite_executable);
    renderer_cmd
        .args(&args_refs)
        .current_dir(&config.renderite_directory);
    lifetime.prepare_command(&mut renderer_cmd);
    match renderer_cmd.spawn() {
        Ok(process) => {
            handle_spawned_renderer(&args, outgoing, config, lifetime, renderer_child, process);
        }
        Err(e) => {
            logger::error!("Failed to start renderer: {}", e);
        }
    }
    LoopAction::Continue
}

/// Registers a spawned renderer and notifies the Host on success.
fn handle_spawned_renderer(
    args: &[String],
    outgoing: &mut Publisher,
    config: &ResoBootConfig,
    lifetime: &ChildLifetimeGroup,
    renderer_child: &SharedChildSlot,
    mut process: Child,
) {
    if let Err(e) = lifetime.register_spawned(&process) {
        logger::error!("Renderer started but could not join lifetime group: {}", e);
        let _ = process.kill();
        let _ = process.wait();
        return;
    }

    let pid = process.id();
    match renderer_child.replace(process) {
        Ok(Some(mut old)) => {
            logger::info!(
                "Replacing previous renderer PID {} with new process",
                old.id()
            );
            let _ = old.kill();
            let _ = old.wait();
        }
        Ok(None) => {}
        Err(_) => {
            logger::error!(
                "Renderer started but renderer_child mutex was poisoned; terminating spawned renderer"
            );
            return;
        }
    }
    logger::info!(
        "Renderer started PID {} cwd={} args={}",
        pid,
        config.renderite_directory.display(),
        args.join(" ")
    );
    let response = format!("RENDERITE_STARTED:{pid}");
    if !outgoing.try_enqueue(response.as_bytes()) {
        logger::warn!("Failed to enqueue RENDERITE_STARTED:{pid} on bootstrapper_out");
    }
}
