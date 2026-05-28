//! Public logging facade backed by a single global file sink.

mod line;
mod sink;

pub use sink::{
    enabled, flush, init, init_with_mirror, is_initialized, log, log_with_target, set_max_level,
    set_mirror_writer, try_log,
};
