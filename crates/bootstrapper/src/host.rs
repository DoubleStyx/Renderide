//! Spawning Renderite Host (Wine + `LinuxBootstrap.sh` vs `dotnet Renderite.Host.dll`).

mod output;
mod priority;
mod runtime_config;
mod spawn;

pub use output::spawn_output_drainer;
pub use priority::set_host_above_normal_priority;
pub use spawn::spawn_host;
