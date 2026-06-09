//! Release manifest signing helper.

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use base64::Engine;
use ed25519_dalek::{Signer, SigningKey};
use thiserror::Error;

const RELEASE_PRIVATE_KEY_ENV: &str = "RENDERIDE_RELEASE_PRIVATE_KEY_HEX";

#[derive(Debug, Error)]
enum SignManifestError {
    #[error("usage: renderide-sign-release-manifest <manifest-path>")]
    Usage,
    #[error("{RELEASE_PRIVATE_KEY_ENV} must contain a 32-byte Ed25519 private key seed as hex")]
    MissingPrivateKey,
    #[error("expected 64 hex chars, got {0}")]
    InvalidHexLength(usize),
    #[error("invalid hex at byte {index}: {source}")]
    InvalidHexByte {
        index: usize,
        source: std::num::ParseIntError,
    },
    #[error("read manifest {path}: {source}")]
    ReadManifest { path: PathBuf, source: io::Error },
    #[error("write signature: {0}")]
    WriteSignature(io::Error),
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ = writeln!(io::stderr(), "{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), SignManifestError> {
    let mut args = env::args_os().skip(1);
    let manifest_path = PathBuf::from(args.next().ok_or(SignManifestError::Usage)?);
    if args.next().is_some() {
        return Err(SignManifestError::Usage);
    }

    let private_key_hex = env::var(RELEASE_PRIVATE_KEY_ENV)
        .map_err(|_source| SignManifestError::MissingPrivateKey)?;
    let signing_key = SigningKey::from_bytes(&decode_hex_32(&private_key_hex)?);
    let manifest = fs::read(&manifest_path).map_err(|source| SignManifestError::ReadManifest {
        path: manifest_path,
        source,
    })?;
    let signature = signing_key.sign(&manifest);
    let signature_b64 = base64::engine::general_purpose::STANDARD.encode(signature.to_bytes());

    let mut stdout = io::stdout().lock();
    stdout
        .write_all(signature_b64.as_bytes())
        .and_then(|()| stdout.write_all(b"\n"))
        .map_err(SignManifestError::WriteSignature)
}

fn decode_hex_32(hex: &str) -> Result<[u8; 32], SignManifestError> {
    let hex = hex.trim();
    if hex.len() != 64 {
        return Err(SignManifestError::InvalidHexLength(hex.len()));
    }
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&hex[offset..offset + 2], 16)
            .map_err(|source| SignManifestError::InvalidHexByte { index, source })?;
    }
    Ok(out)
}
