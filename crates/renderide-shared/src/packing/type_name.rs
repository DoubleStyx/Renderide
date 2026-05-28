//! Shared type-name formatting for packing diagnostics.

/// Returns the unqualified Rust type name of `T`.
pub(super) fn short_type_name<T>() -> &'static str {
    let full = std::any::type_name::<T>();
    full.rsplit("::").next().unwrap_or(full)
}
