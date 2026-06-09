//! Startup connection parameters for Cloudtoid IPC.
//!
//! Matches the managed host's argument convention (see `RenderingManager.GetConnectionParameters`).

use std::env::{self, VarError};
use std::mem::size_of;
use std::net::UdpSocket;
use std::num::TryFromIntError;
use std::string::FromUtf8Error;
use std::sync::atomic::{AtomicBool, Ordering};

use thiserror::Error;

/// Loopback UDP port used by debug renderer attach handshakes.
const ATTACH_RENDERER_PORT: u16 = 42_512;
/// Maximum accepted attach handshake payload size.
const ATTACH_RENDERER_PACKET_MAX_BYTES: usize = 4096;
/// Environment variable carrying the debug-attach v2 nonce.
pub const ATTACH_RENDERER_NONCE_ENV: &str = "RENDERIDE_ATTACH_NONCE";
/// Magic prefix for authenticated debug-attach v2 packets.
const ATTACH_RENDERER_V2_MAGIC: &[u8] = b"RenderideAttachV2\0";
/// Minimum nonce length accepted for debug attach.
const ATTACH_RENDERER_MIN_NONCE_BYTES: usize = 16;
/// Maximum nonce length accepted for debug attach.
const ATTACH_RENDERER_MAX_NONCE_BYTES: usize = 128;
/// Maximum number of bytes in a .NET 7-bit encoded i32.
const MAX_7BIT_ENCODED_I32_BYTES: usize = 5;
/// High bit marking another 7-bit length byte.
const SEVEN_BIT_CONTINUATION: u8 = 0x80;
/// Data bits carried by each 7-bit length byte.
const SEVEN_BIT_VALUE_MASK: u8 = 0x7f;
/// Valid data bits in the fifth byte of a non-negative .NET i32 length.
const SEVEN_BIT_FINAL_I32_MASK: u8 = 0x07;

/// Error returned when renderer initialization fails (singleton or IPC connect).
#[derive(Debug, Error)]
pub enum InitError {
    /// Only one renderer session may initialize the singleton guard.
    #[error("renderer singleton already initialized")]
    SingletonAlreadyExists,
    /// Opening a subscriber or publisher failed.
    #[error("IPC connect: {0}")]
    IpcConnect(String),
}

/// Error returned when the attach renderer UDP handshake cannot be decoded.
#[derive(Debug, Error)]
enum AttachConnectionError {
    /// The attach listener could not bind to the loopback port.
    #[error("failed to bind UDP attach listener on 127.0.0.1:{ATTACH_RENDERER_PORT}: {0}")]
    Bind(#[source] std::io::Error),
    /// The attach listener failed while waiting for the host datagram.
    #[error("failed to receive attach renderer parameters: {0}")]
    Receive(#[source] std::io::Error),
    /// An authenticated v2 attach packet was received without a configured nonce.
    #[error("attach renderer v2 packet requires {ATTACH_RENDERER_NONCE_ENV}")]
    MissingNonce,
    /// Attach nonce was outside the accepted size range.
    #[error("attach renderer nonce length {len} is outside the accepted range")]
    InvalidNonceLength {
        /// Nonce byte length.
        len: usize,
    },
    /// Attach nonce could not be read as a Unicode environment value.
    #[error("attach renderer nonce is not valid Unicode")]
    InvalidNonceEncoding,
    /// The packet was not an authenticated v2 attach packet.
    #[error("attach renderer packet is not an authenticated v2 packet")]
    MissingV2Magic,
    /// The packet nonce does not match the expected launch nonce.
    #[error("attach renderer nonce mismatch")]
    NonceMismatch,
    /// The payload ended before the queue name length prefix completed.
    #[error("attach renderer payload ended before the queue name length prefix completed")]
    TruncatedStringLength,
    /// The queue name length prefix is not a valid non-negative .NET i32.
    #[error("attach renderer queue name length prefix is malformed")]
    MalformedStringLength,
    /// The queue name length does not fit the current target.
    #[error("attach renderer queue name length {length} does not fit this target")]
    QueueNameTooLong {
        /// Queue name byte length decoded from the payload.
        length: u64,
        /// Integer conversion failure.
        #[source]
        source: TryFromIntError,
    },
    /// The payload ended before the declared queue name bytes completed.
    #[error("attach renderer queue name needs {expected} bytes, but payload has {remaining}")]
    TruncatedQueueName {
        /// Expected UTF-8 queue name byte count.
        expected: usize,
        /// Bytes remaining after the length prefix.
        remaining: usize,
    },
    /// The queue name bytes are not valid UTF-8.
    #[error("attach renderer queue name is not valid UTF-8: {source}")]
    InvalidQueueName {
        /// UTF-8 decoder error.
        #[source]
        source: FromUtf8Error,
    },
    /// The payload ended before the queue capacity completed.
    #[error("attach renderer queue capacity needs 8 bytes, but payload has {remaining}")]
    TruncatedQueueCapacity {
        /// Bytes remaining after the queue name.
        remaining: usize,
    },
    /// The queue capacity was not a positive byte count.
    #[error("attach renderer queue capacity must be positive, got {queue_capacity}")]
    InvalidQueueCapacity {
        /// Decoded queue capacity in bytes.
        queue_capacity: i64,
    },
    /// The queue capacity exceeded the renderer policy cap.
    #[error("attach renderer queue capacity {queue_capacity} exceeds maximum {max_capacity}")]
    QueueCapacityTooLarge {
        /// Decoded queue capacity in bytes.
        queue_capacity: i64,
        /// Maximum accepted queue capacity in bytes.
        max_capacity: i64,
    },
}

/// Default queue capacity (8 MiB), matching `MessagingManager.DEFAULT_CAPACITY`.
pub const DEFAULT_QUEUE_CAPACITY: i64 = 8_388_608;

/// Parsed connection parameters for IPC with the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectionParams {
    /// Base queue name (without `Primary`/`Background` or `A`/`S` suffixes).
    pub queue_name: String,
    /// Ring capacity in bytes (user payload; excludes queue header).
    pub queue_capacity: i64,
}

/// Process-wide guard ensuring only one renderer initializes the IPC singleton.
static RENDERIDE_SINGLETON_CLAIMED: AtomicBool = AtomicBool::new(false);

/// Reserves the single-renderer process guard (Unity: one `RenderingManager`).
///
/// Call once at startup; subsequent calls return [`InitError::SingletonAlreadyExists`].
pub fn try_claim_renderer_singleton() -> Result<(), InitError> {
    if RENDERIDE_SINGLETON_CLAIMED.swap(true, Ordering::SeqCst) {
        return Err(InitError::SingletonAlreadyExists);
    }
    Ok(())
}

/// Parses `-QueueName` / `-QueueCapacity` from `std::env::args`, if present.
///
/// If `-AttachRenderer` was passed, blocks and listens for the host's attach
/// datagram instead.
///
/// Returns [`None`] when arguments are missing or invalid so the process can run without IPC.
pub fn get_connection_parameters() -> Option<ConnectionParams> {
    let args: Vec<String> = env::args().collect();

    if has_attach_renderer_arg(&args) {
        logger::info!("Waiting for Resonite to attach debug renderer.");
        return get_connection_parameters_from_attach_renderer();
    }

    parse_connection_args(&args)
}

/// Returns true when an argument selects the debug attach renderer path.
fn has_attach_renderer_arg(args: &[impl AsRef<str>]) -> bool {
    args.iter()
        .any(|arg| arg_has_ascii_suffix(arg.as_ref(), "attachrenderer"))
}

/// Returns true when `arg` ends with `suffix`, ignoring ASCII case.
fn arg_has_ascii_suffix(arg: &str, suffix: &str) -> bool {
    let arg = arg.as_bytes();
    let suffix = suffix.as_bytes();
    arg.len() >= suffix.len() && arg[arg.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
}

/// Waits for connection parameters from the debug attach UDP handshake.
fn get_connection_parameters_from_attach_renderer() -> Option<ConnectionParams> {
    let nonce = match attach_renderer_nonce_from_env() {
        Ok(nonce) => nonce,
        Err(error) => {
            logger::warn!("Attach renderer handshake failed: {error}");
            return None;
        }
    };
    if nonce.is_none() {
        logger::warn!(
            "Attach renderer nonce not configured; accepting unauthenticated legacy attach payloads"
        );
    }
    match receive_attach_renderer_parameters(nonce.as_deref()) {
        Ok(params) => Some(params),
        Err(error) => {
            logger::warn!("Attach renderer handshake failed: {error}");
            None
        }
    }
}

fn attach_renderer_nonce_from_env() -> Result<Option<Vec<u8>>, AttachConnectionError> {
    let nonce = match env::var(ATTACH_RENDERER_NONCE_ENV) {
        Ok(nonce) => nonce.into_bytes(),
        Err(VarError::NotPresent) => return Ok(None),
        Err(VarError::NotUnicode(_nonce)) => {
            return Err(AttachConnectionError::InvalidNonceEncoding);
        }
    };
    validate_attach_renderer_nonce(&nonce)?;
    Ok(Some(nonce))
}

fn validate_attach_renderer_nonce(nonce: &[u8]) -> Result<(), AttachConnectionError> {
    if !(ATTACH_RENDERER_MIN_NONCE_BYTES..=ATTACH_RENDERER_MAX_NONCE_BYTES).contains(&nonce.len()) {
        return Err(AttachConnectionError::InvalidNonceLength { len: nonce.len() });
    }
    Ok(())
}

/// Receives and parses the debug attach UDP datagram.
fn receive_attach_renderer_parameters(
    expected_nonce: Option<&[u8]>,
) -> Result<ConnectionParams, AttachConnectionError> {
    let socket = UdpSocket::bind(("127.0.0.1", ATTACH_RENDERER_PORT))
        .map_err(AttachConnectionError::Bind)?;
    logger::info!(
        "Listening on UDP port {ATTACH_RENDERER_PORT}. Launch Resonite with -AttachRenderer"
    );

    let mut buf = [0u8; ATTACH_RENDERER_PACKET_MAX_BYTES];
    let (len, _) = socket
        .recv_from(&mut buf)
        .map_err(AttachConnectionError::Receive)?;
    parse_attach_renderer_packet(&buf[..len], expected_nonce)
}

/// Parses the authenticated host attach datagram.
fn parse_attach_renderer_packet(
    packet: &[u8],
    expected_nonce: Option<&[u8]>,
) -> Result<ConnectionParams, AttachConnectionError> {
    let Some(rest) = packet.strip_prefix(ATTACH_RENDERER_V2_MAGIC) else {
        if expected_nonce.is_some() {
            return Err(AttachConnectionError::MissingV2Magic);
        }
        logger::warn!("Attach renderer accepted unauthenticated legacy payload");
        return parse_attach_renderer_legacy_payload(packet);
    };
    let Some(expected_nonce) = expected_nonce else {
        return Err(AttachConnectionError::MissingNonce);
    };
    validate_attach_renderer_nonce(expected_nonce)?;
    let (nonce_len, nonce_offset) = read_7bit_encoded_usize(rest)?;
    let nonce_end = nonce_offset.checked_add(nonce_len).ok_or_else(|| {
        AttachConnectionError::TruncatedQueueName {
            expected: nonce_len,
            remaining: rest.len().saturating_sub(nonce_offset),
        }
    })?;
    if nonce_end > rest.len() {
        return Err(AttachConnectionError::TruncatedQueueName {
            expected: nonce_len,
            remaining: rest.len().saturating_sub(nonce_offset),
        });
    }
    if &rest[nonce_offset..nonce_end] != expected_nonce {
        return Err(AttachConnectionError::NonceMismatch);
    }
    parse_attach_renderer_legacy_payload(&rest[nonce_end..])
}

/// Parses the queue portion of the host attach datagram encoded by `.NET BinaryWriter`.
fn parse_attach_renderer_legacy_payload(
    packet: &[u8],
) -> Result<ConnectionParams, AttachConnectionError> {
    let (queue_name_len, queue_name_offset) = read_7bit_encoded_usize(packet)?;
    let Some(queue_name_end) = queue_name_offset.checked_add(queue_name_len) else {
        return Err(AttachConnectionError::TruncatedQueueName {
            expected: queue_name_len,
            remaining: packet.len().saturating_sub(queue_name_offset),
        });
    };

    if queue_name_end > packet.len() {
        return Err(AttachConnectionError::TruncatedQueueName {
            expected: queue_name_len,
            remaining: packet.len().saturating_sub(queue_name_offset),
        });
    }

    let queue_name = String::from_utf8(packet[queue_name_offset..queue_name_end].to_vec())
        .map_err(|source| AttachConnectionError::InvalidQueueName { source })?;
    let capacity_end = queue_name_end + size_of::<i64>();
    if capacity_end > packet.len() {
        return Err(AttachConnectionError::TruncatedQueueCapacity {
            remaining: packet.len().saturating_sub(queue_name_end),
        });
    }

    let mut queue_capacity_bytes = [0u8; size_of::<i64>()];
    queue_capacity_bytes.copy_from_slice(&packet[queue_name_end..capacity_end]);
    let queue_capacity = i64::from_le_bytes(queue_capacity_bytes);
    if queue_capacity <= 0 {
        return Err(AttachConnectionError::InvalidQueueCapacity { queue_capacity });
    }
    if queue_capacity > interprocess::QueueOptions::MAX_CAPACITY {
        return Err(AttachConnectionError::QueueCapacityTooLarge {
            queue_capacity,
            max_capacity: interprocess::QueueOptions::MAX_CAPACITY,
        });
    }

    Ok(ConnectionParams {
        queue_name,
        queue_capacity,
    })
}

/// Reads a .NET 7-bit encoded non-negative i32 length from `packet`.
fn read_7bit_encoded_usize(packet: &[u8]) -> Result<(usize, usize), AttachConnectionError> {
    let mut value = 0u32;

    for byte_index in 0..MAX_7BIT_ENCODED_I32_BYTES {
        let byte = *packet
            .get(byte_index)
            .ok_or(AttachConnectionError::TruncatedStringLength)?;

        if byte_index == MAX_7BIT_ENCODED_I32_BYTES - 1 && byte & !SEVEN_BIT_FINAL_I32_MASK != 0 {
            return Err(AttachConnectionError::MalformedStringLength);
        }

        value |= u32::from(byte & SEVEN_BIT_VALUE_MASK) << (byte_index * 7);

        if byte & SEVEN_BIT_CONTINUATION == 0 {
            let length = usize::try_from(value).map_err(|source| {
                AttachConnectionError::QueueNameTooLong {
                    length: u64::from(value),
                    source,
                }
            })?;
            return Ok((length, byte_index + 1));
        }
    }

    Err(AttachConnectionError::MalformedStringLength)
}

/// Scans `args` for the first complete `-QueueName` / `-QueueCapacity` pair (case-insensitive
/// flag suffix).
/// Requires QueueCapacity to be a positive integer within [`interprocess::QueueOptions::MAX_CAPACITY`].
/// Returns [`None`] if either flag is missing, malformed, or duplicated before
/// the pair completes.
fn parse_connection_args(args: &[impl AsRef<str>]) -> Option<ConnectionParams> {
    if args.is_empty() {
        return None;
    }

    let mut queue_name: Option<String> = None;
    let mut queue_capacity: Option<i64> = None;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let next_i = i + 1;
        if next_i >= args.len() {
            break;
        }

        if arg_has_ascii_suffix(arg.as_ref(), "queuename") {
            if queue_name.is_some() {
                return None;
            }
            queue_name = Some(args[next_i].as_ref().to_owned());
            i = next_i;
        } else if arg_has_ascii_suffix(arg.as_ref(), "queuecapacity") {
            if queue_capacity.is_some_and(|c| c > 0) {
                return None;
            }
            queue_capacity = args[next_i]
                .as_ref()
                .parse()
                .ok()
                .filter(|&c| c > 0 && c <= interprocess::QueueOptions::MAX_CAPACITY);
            i = next_i;
        }

        i += 1;

        if let Some(name) = queue_name.as_ref()
            && let Some(cap) = queue_capacity
            && cap > 0
            && cap <= interprocess::QueueOptions::MAX_CAPACITY
        {
            return Some(ConnectionParams {
                queue_name: name.clone(),
                queue_capacity: cap,
            });
        }
    }

    queue_name.and_then(|name| {
        queue_capacity
            .filter(|&c| c > 0 && c <= interprocess::QueueOptions::MAX_CAPACITY)
            .map(|cap| ConnectionParams {
                queue_name: name,
                queue_capacity: cap,
            })
    })
}

/// Subscriber queue name for the renderer (non-authority -> `...A` side).
pub fn subscriber_queue_name(base: &str, channel: &str) -> String {
    format!("{base}{channel}A")
}

/// Publisher queue name for the renderer (non-authority -> `...S` side).
pub fn publisher_queue_name(base: &str, channel: &str) -> String {
    format!("{base}{channel}S")
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_NONCE: &[u8] = b"0123456789abcdef";

    fn attach_packet(queue_name: &str, queue_capacity: i64) -> Vec<u8> {
        v2_packet(&legacy_attach_packet(queue_name, queue_capacity))
    }

    fn legacy_attach_packet(queue_name: &str, queue_capacity: i64) -> Vec<u8> {
        let mut packet = Vec::new();
        write_7bit_encoded_usize(queue_name.len(), &mut packet);
        packet.extend_from_slice(queue_name.as_bytes());
        packet.extend_from_slice(&queue_capacity.to_le_bytes());
        packet
    }

    fn v2_packet(legacy_payload: &[u8]) -> Vec<u8> {
        let mut packet = Vec::new();
        packet.extend_from_slice(ATTACH_RENDERER_V2_MAGIC);
        write_7bit_encoded_usize(TEST_NONCE.len(), &mut packet);
        packet.extend_from_slice(TEST_NONCE);
        packet.extend_from_slice(legacy_payload);
        packet
    }

    fn write_7bit_encoded_usize(mut value: usize, packet: &mut Vec<u8>) {
        while value >= usize::from(SEVEN_BIT_CONTINUATION) {
            packet.push((value as u8 & SEVEN_BIT_VALUE_MASK) | SEVEN_BIT_CONTINUATION);
            value >>= 7;
        }
        packet.push(value as u8);
    }

    #[test]
    fn has_attach_renderer_arg_matches_case_insensitive_suffix() {
        assert!(has_attach_renderer_arg(&["renderide", "-AttachRenderer"]));
        assert!(has_attach_renderer_arg(&["renderide", "/attachrenderer"]));
        assert!(has_attach_renderer_arg(&[
            "renderide",
            "--renderide-ATTACHRENDERER"
        ]));
        assert!(!has_attach_renderer_arg(&[
            "renderide",
            "-AttachRendererSuffix"
        ]));
    }

    #[test]
    fn parse_attach_renderer_packet_accepts_authenticated_binary_writer_payload() {
        assert_eq!(
            parse_attach_renderer_packet(
                &attach_packet("RenderideQueue", 8_388_608),
                Some(TEST_NONCE),
            )
            .expect("attach packet should parse"),
            ConnectionParams {
                queue_name: "RenderideQueue".into(),
                queue_capacity: 8_388_608,
            }
        );
    }

    #[test]
    fn parse_attach_renderer_packet_accepts_multibyte_string_length() {
        let queue_name = "q".repeat(130);
        assert_eq!(
            parse_attach_renderer_packet(&attach_packet(&queue_name, 4096), Some(TEST_NONCE))
                .expect("attach packet should parse"),
            ConnectionParams {
                queue_name,
                queue_capacity: 4096,
            }
        );
    }

    #[test]
    fn parse_attach_renderer_packet_rejects_legacy_payload_when_nonce_is_configured() {
        let error = parse_attach_renderer_packet(
            &legacy_attach_packet("RenderideQueue", 8_388_608),
            Some(TEST_NONCE),
        )
        .expect_err("legacy attach packet should be rejected");
        assert!(matches!(error, AttachConnectionError::MissingV2Magic));
    }

    #[test]
    fn parse_attach_renderer_packet_accepts_legacy_payload_when_nonce_is_absent() {
        assert_eq!(
            parse_attach_renderer_packet(&legacy_attach_packet("RenderideQueue", 8_388_608), None,)
                .expect("legacy attach packet should parse"),
            ConnectionParams {
                queue_name: "RenderideQueue".into(),
                queue_capacity: 8_388_608,
            }
        );
    }

    #[test]
    fn parse_attach_renderer_packet_rejects_v2_payload_when_nonce_is_absent() {
        let error = parse_attach_renderer_packet(&attach_packet("RenderideQueue", 8_388_608), None)
            .expect_err("v2 attach packet without expected nonce should be rejected");
        assert!(matches!(error, AttachConnectionError::MissingNonce));
    }

    #[test]
    fn parse_attach_renderer_packet_rejects_nonce_mismatch() {
        let error = parse_attach_renderer_packet(
            &attach_packet("RenderideQueue", 8_388_608),
            Some(b"fedcba9876543210"),
        )
        .expect_err("wrong nonce should be rejected");
        assert!(matches!(error, AttachConnectionError::NonceMismatch));
    }

    #[test]
    fn parse_attach_renderer_packet_rejects_truncated_string_length() {
        let error =
            parse_attach_renderer_packet(&v2_packet(&[SEVEN_BIT_CONTINUATION]), Some(TEST_NONCE))
                .expect_err("length prefix should be incomplete");
        assert!(matches!(
            error,
            AttachConnectionError::TruncatedStringLength
        ));
    }

    #[test]
    fn parse_attach_renderer_packet_rejects_malformed_string_length() {
        let error = parse_attach_renderer_packet(
            &v2_packet(&[0xff, 0xff, 0xff, 0xff, 0x08]),
            Some(TEST_NONCE),
        )
        .expect_err("length prefix should be malformed");
        assert!(matches!(
            error,
            AttachConnectionError::MalformedStringLength
        ));
    }

    #[test]
    fn parse_attach_renderer_packet_rejects_truncated_queue_name() {
        let error = parse_attach_renderer_packet(&v2_packet(&[4, b'n']), Some(TEST_NONCE))
            .expect_err("queue name should be incomplete");
        assert!(matches!(
            error,
            AttachConnectionError::TruncatedQueueName {
                expected: 4,
                remaining: 1
            }
        ));
    }

    #[test]
    fn parse_attach_renderer_packet_rejects_invalid_utf8_queue_name() {
        let mut packet = vec![1, 0xff];
        packet.extend_from_slice(&4096_i64.to_le_bytes());

        let error = parse_attach_renderer_packet(&v2_packet(&packet), Some(TEST_NONCE))
            .expect_err("queue name should reject invalid UTF-8");
        assert!(matches!(
            error,
            AttachConnectionError::InvalidQueueName { .. }
        ));
    }

    #[test]
    fn parse_attach_renderer_packet_rejects_truncated_queue_capacity() {
        let error = parse_attach_renderer_packet(
            &v2_packet(&[4, b'n', b'a', b'm', b'e']),
            Some(TEST_NONCE),
        )
        .expect_err("queue capacity should be incomplete");
        assert!(matches!(
            error,
            AttachConnectionError::TruncatedQueueCapacity { remaining: 0 }
        ));
    }

    #[test]
    fn parse_attach_renderer_packet_rejects_non_positive_queue_capacity() {
        let zero = parse_attach_renderer_packet(&attach_packet("queue", 0), Some(TEST_NONCE))
            .expect_err("zero capacity should be invalid");
        assert!(matches!(
            zero,
            AttachConnectionError::InvalidQueueCapacity { queue_capacity: 0 }
        ));

        let negative = parse_attach_renderer_packet(&attach_packet("queue", -1), Some(TEST_NONCE))
            .expect_err("negative capacity should be invalid");
        assert!(matches!(
            negative,
            AttachConnectionError::InvalidQueueCapacity { queue_capacity: -1 }
        ));
    }

    #[test]
    fn parse_attach_renderer_packet_rejects_oversized_queue_capacity() {
        let error = parse_attach_renderer_packet(
            &attach_packet("queue", interprocess::QueueOptions::MAX_CAPACITY + 8),
            Some(TEST_NONCE),
        )
        .expect_err("oversized capacity should be invalid");
        assert!(matches!(
            error,
            AttachConnectionError::QueueCapacityTooLarge { .. }
        ));
    }

    #[test]
    fn parses_queue_name_and_capacity_case_insensitive() {
        let cmd = [
            "renderide",
            "-QueueName",
            "TestSession",
            "-QueueCapacity",
            "8388608",
        ];
        assert_eq!(
            parse_connection_args(&cmd),
            Some(ConnectionParams {
                queue_name: "TestSession".to_string(),
                queue_capacity: 8_388_608,
            })
        );
    }

    #[test]
    fn parse_args_accepts_queue_capacity_before_queue_name() {
        let cmd = [
            "renderide",
            "-QueueCapacity",
            "4096",
            "-QueueName",
            "LaterName",
        ];
        assert_eq!(
            parse_connection_args(&cmd),
            Some(ConnectionParams {
                queue_name: "LaterName".into(),
                queue_capacity: 4096,
            })
        );
    }

    #[test]
    fn parse_args_accepts_prefixed_flag_spellings_by_suffix() {
        let cmd = [
            "renderide",
            "--renderide-QueueName",
            "Prefixed",
            "/QueueCapacity",
            "2048",
        ];
        assert_eq!(
            parse_connection_args(&cmd),
            Some(ConnectionParams {
                queue_name: "Prefixed".into(),
                queue_capacity: 2048,
            })
        );
    }

    #[test]
    fn parse_args_returns_none_when_flag_value_is_missing() {
        assert_eq!(parse_connection_args(&["renderide", "-QueueName"]), None);
        assert_eq!(
            parse_connection_args(&["renderide", "-QueueName", "Name", "-QueueCapacity"]),
            None
        );
    }

    #[test]
    fn parse_args_allows_replacing_invalid_capacity_with_later_valid_capacity() {
        let cmd = [
            "renderide",
            "-QueueName",
            "Recover",
            "-QueueCapacity",
            "bad",
            "-QueueCapacity",
            "1024",
        ];

        assert_eq!(
            parse_connection_args(&cmd),
            Some(ConnectionParams {
                queue_name: "Recover".into(),
                queue_capacity: 1024,
            })
        );
    }

    #[test]
    fn parse_args_rejects_duplicate_queue_name() {
        let cmd = [
            "renderide",
            "-QueueName",
            "First",
            "-QueueName",
            "Second",
            "-QueueCapacity",
            "4096",
        ];
        assert_eq!(parse_connection_args(&cmd), None);
    }

    #[test]
    fn parse_args_returns_first_complete_pair_and_ignores_later_flags() {
        let cmd = [
            "renderide",
            "-QueueName",
            "S",
            "-QueueCapacity",
            "4096",
            "-QueueCapacity",
            "8192",
        ];
        assert_eq!(
            parse_connection_args(&cmd),
            Some(ConnectionParams {
                queue_name: "S".into(),
                queue_capacity: 4096,
            })
        );
    }

    #[test]
    fn parse_args_rejects_non_numeric_or_non_positive_capacity() {
        assert_eq!(
            parse_connection_args(&["r", "-QueueName", "n", "-QueueCapacity", "not_a_number"]),
            None
        );
        assert_eq!(
            parse_connection_args(&["r", "-QueueName", "n", "-QueueCapacity", "0"]),
            None
        );
        assert_eq!(
            parse_connection_args(&["r", "-QueueName", "n", "-QueueCapacity", "-100"]),
            None
        );
        assert_eq!(
            parse_connection_args(&[
                "r",
                "-QueueName",
                "n",
                "-QueueCapacity",
                &(interprocess::QueueOptions::MAX_CAPACITY + 8).to_string()
            ]),
            None
        );
    }

    #[test]
    fn parse_args_returns_none_for_empty_argv() {
        assert_eq!(parse_connection_args(&Vec::<String>::new()), None);
    }

    #[test]
    fn ipc_suffixes_match_cloudtoid_non_authority() {
        let p = ConnectionParams {
            queue_name: "Foo".to_string(),
            queue_capacity: 1024,
        };
        assert_eq!(
            subscriber_queue_name(&p.queue_name, "Primary"),
            "FooPrimaryA"
        );
        assert_eq!(
            publisher_queue_name(&p.queue_name, "Primary"),
            "FooPrimaryS"
        );
        assert_eq!(
            subscriber_queue_name(&p.queue_name, "Background"),
            "FooBackgroundA"
        );
        assert_eq!(
            publisher_queue_name(&p.queue_name, "Background"),
            "FooBackgroundS"
        );
    }
}
