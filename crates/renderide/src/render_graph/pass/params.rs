//! Pass-parameter and blackboard declaration metadata.

use std::any::{TypeId, type_name};

use crate::render_graph::blackboard::BlackboardSlot;
use crate::render_graph::error::SetupError;

use super::builder::PassBuilder;

/// Stable identity for a typed blackboard slot declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlackboardSlotKey {
    /// Runtime type id for fast matching.
    pub(crate) type_id: TypeId,
    /// Fully qualified Rust type name for diagnostics.
    pub type_name: &'static str,
}

impl BlackboardSlotKey {
    /// Builds a key for blackboard slot type `S`.
    pub fn of<S: BlackboardSlot>() -> Self {
        Self {
            type_id: TypeId::of::<S>(),
            type_name: type_name::<S>(),
        }
    }
}

/// Blackboard access kind declared by a graph pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlackboardAccessKind {
    /// Slot must be present before the pass records.
    RequiredRead,
    /// Slot may be read if available and does not require a producer.
    OptionalRead,
    /// Slot is written by the pass.
    Write,
}

impl BlackboardAccessKind {
    /// Returns whether this access reads slot contents.
    pub const fn reads(self) -> bool {
        matches!(self, Self::RequiredRead | Self::OptionalRead)
    }

    /// Returns whether this access requires a producer or seed.
    pub const fn requires_value(self) -> bool {
        matches!(self, Self::RequiredRead)
    }

    /// Returns whether this access writes slot contents.
    pub const fn writes(self) -> bool {
        matches!(self, Self::Write)
    }
}

/// One declared blackboard access from pass setup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlackboardAccessDecl {
    /// Slot being accessed.
    pub slot: BlackboardSlotKey,
    /// Access semantics.
    pub kind: BlackboardAccessKind,
}

impl BlackboardAccessDecl {
    /// Creates a declaration for slot `S`.
    pub fn new<S: BlackboardSlot>(kind: BlackboardAccessKind) -> Self {
        Self {
            slot: BlackboardSlotKey::of::<S>(),
            kind,
        }
    }
}

/// Graph-level blackboard seed declared by the graph assembler or executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlackboardSeedDecl {
    /// Slot made available before graph pass recording.
    pub slot: BlackboardSlotKey,
    /// Human-readable producer label for diagnostics.
    pub producer: &'static str,
}

impl BlackboardSeedDecl {
    /// Creates a seed declaration for slot `S`.
    pub fn new<S: BlackboardSlot>(producer: &'static str) -> Self {
        Self {
            slot: BlackboardSlotKey::of::<S>(),
            producer,
        }
    }
}

/// One field in a graph pass parameter struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PassParameterField {
    /// Rust field name or logical declaration label.
    pub name: &'static str,
    /// Short semantic role for diagnostics.
    pub role: &'static str,
}

impl PassParameterField {
    /// Creates a field schema entry.
    pub const fn new(name: &'static str, role: &'static str) -> Self {
        Self { name, role }
    }
}

/// Static schema for one pass's setup-time parameter payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassParameterSchema {
    /// Schema name, usually the parameter struct name.
    pub name: String,
    /// Declared fields in stable display order.
    pub fields: Vec<PassParameterField>,
}

impl PassParameterSchema {
    /// Creates an empty schema with the supplied name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
        }
    }

    /// Adds one field entry and returns the updated schema.
    pub fn with_field(mut self, field: PassParameterField) -> Self {
        self.fields.push(field);
        self
    }
}

/// Trait implemented by pass parameter structs that can declare their graph-facing resources.
pub trait GraphPassParameters {
    /// Static parameter schema for diagnostics and tooling.
    fn schema(&self) -> PassParameterSchema;

    /// Declares resources and blackboard access against a pass builder.
    fn declare(&self, builder: &mut PassBuilder<'_>) -> Result<(), SetupError>;
}
