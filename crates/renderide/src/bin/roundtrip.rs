//! Roundtrip binary for C#–Rust serialization tests.
//!
//! Unpacks a generated shared type from `input`, repacks it, and writes bytes to `output`.
//! The C# test harness compares the original packed buffer with the Rust output.
//!
//! # Usage
//!
//! ```text
//! roundtrip <type_name> <input.bin> <output.bin>
//! ```

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

use renderide::shared;
use thiserror::Error;

/// Parsed command-line arguments for a single roundtrip invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    /// Fully qualified or short type name accepted by [`shared::roundtrip_dispatch`].
    type_name: String,
    /// Path to the packed input bytes.
    input_path: PathBuf,
    /// Path for the repacked output bytes.
    output_path: PathBuf,
}

/// Fixed usage string for stderr when argument count is wrong.
const USAGE: &str = "Usage: roundtrip <type_name> <input.bin> <output.bin>";

/// Failure modes for [`run`], excluding usage / parse errors (handled in [`main`]).
#[derive(Debug, Error)]
enum RoundtripError {
    /// Failed to read the input file.
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// [`shared::roundtrip_dispatch`] returned an error.
    #[error("roundtrip failed for {type_name}: {source}")]
    Dispatch {
        type_name: String,
        #[source]
        source: std::io::Error,
    },
    /// Failed to write the output file.
    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Parses `argv` after the program name into [`Args`].
///
/// Expects exactly three arguments: type name, input path, output path. Paths may be any
/// OS-native representation; the type name must be valid UTF-8 (matches generated C# names).
fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Args, String> {
    let mut args: Vec<OsString> = args.into_iter().collect();
    if args.len() != 3 {
        return Err(format!("{USAGE}\nexpected 3 arguments, got {}", args.len()));
    }

    #[expect(
        clippy::map_err_ignore,
        reason = "`OsString::into_string` returns the original `OsString` on error, not useful here"
    )]
    let type_name = args
        .remove(0)
        .into_string()
        .map_err(|_| "type_name must be valid UTF-8".to_string())?;
    let input_path = PathBuf::from(args.remove(0));
    let output_path = PathBuf::from(args.remove(0));

    Ok(Args {
        type_name,
        input_path,
        output_path,
    })
}

/// Reads `input_path`, runs [`shared::roundtrip_dispatch`], writes `output_path`.
fn run(args: &Args) -> Result<(), RoundtripError> {
    let input = std::fs::read(&args.input_path).map_err(|source| RoundtripError::Read {
        path: args.input_path.clone(),
        source,
    })?;

    let output = shared::roundtrip_dispatch(&args.type_name, &input).map_err(|source| {
        RoundtripError::Dispatch {
            type_name: args.type_name.clone(),
            source,
        }
    })?;

    std::fs::write(&args.output_path, &output).map_err(|source| RoundtripError::Write {
        path: args.output_path.clone(),
        source,
    })?;

    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    let parsed = match parse_args(args) {
        Ok(a) => a,
        Err(e) => {
            logger::error!("{e}");
            return ExitCode::from(1);
        }
    };

    match run(&parsed) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            logger::error!("{e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn os_vec(items: &[&str]) -> Vec<OsString> {
        items.iter().map(|s| OsString::from(*s)).collect()
    }

    #[test]
    fn parse_args_happy_path() {
        let v = os_vec(&["RendererInitData", "/tmp/in.bin", "/tmp/out.bin"]);
        let got = parse_args(v).expect("parse");
        assert_eq!(got.type_name, "RendererInitData");
        assert_eq!(got.input_path, Path::new("/tmp/in.bin"));
        assert_eq!(got.output_path, Path::new("/tmp/out.bin"));
    }

    #[test]
    fn parse_args_too_few() {
        let err = parse_args(os_vec(&["a", "b"])).unwrap_err();
        assert!(err.contains("expected 3 arguments"));
    }

    #[test]
    fn parse_args_too_many() {
        let err = parse_args(os_vec(&["a", "b", "c", "d"])).unwrap_err();
        assert!(err.contains("expected 3 arguments"));
    }

    /// Non–UTF-8 `type_name` is only constructible portably on Unix (`OsString` bytes).
    #[cfg(unix)]
    #[test]
    fn parse_args_type_name_requires_utf8() {
        use std::os::unix::ffi::OsStringExt;
        let bad = OsString::from_vec(vec![0xff, 0xfe, 0xfd]);
        let err = parse_args(vec![bad, OsString::from("in"), OsString::from("out")]).unwrap_err();
        assert!(err.contains("UTF-8"));
    }
}
