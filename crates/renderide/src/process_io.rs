//! Process bootstrap plumbing: redirect native stdio into the file logger and register a fatal crash
//! hook that appends to the same log path without using the main logger mutex.

pub(crate) mod fatal_crash_log;
pub(crate) mod native_stdio;
