//! Common algebra for finite enum config types: [`LabeledEnum`] trait plus the [`labeled_enum!`]
//! declarative macro that emits the enum, [`Default`], [`serde::Serialize`] / [`serde::Deserialize`],
//! and the trait impl from a single declarative listing of variants x persist string x label x
//! optional aliases.
//!
//! ## Why
//!
//! Every renderer config enum (vsync mode, MSAA sample count, scene-color format, cluster
//! assignment, record parallelism, power preference, tonemap mode, bloom composite mode, watchdog
//! action) used to repeat the same quadruplet of helpers by hand:
//!
//! - `const ALL: [Self; N]` for ImGui pickers and round-trip tests,
//! - `fn label(self) -> &'static str` for the renderer config window,
//! - `fn as_persist_str(self) -> &'static str` and `fn from_persist_str(&str) -> Option<Self>`
//!   for stable TOML round-trip with case-insensitive parsing and legacy aliases.
//!
//! The macro generates all four plus the serde glue from one declaration so each enum's
//! definition stays focused on what's domain-specific (variant set, persist tokens, labels) and
//! aliases stay co-located with their canonical variant.
//!
//! ## Bool-shape configs
//!
//! Some legacy `config.toml` files use boolean shorthand for an enum (for example the original
//! `vsync = true / false` syntax). The bool-aware [`labeled_enum!`] arm routes each boolean to a
//! variant during deserialization, preserving those configs without manual migration.

/// Common metadata for finite enum config types.
///
/// Every `labeled_enum!`-generated type implements this trait. The trait is also useful for
/// generic UI/round-trip code that wants to iterate variants without knowing the concrete enum.
pub trait LabeledEnum: Sized + Copy + 'static {
    /// Every variant in declaration order. ImGui pickers and round-trip tests iterate through this.
    const ALL: &'static [Self];

    /// The variant returned by [`Default::default`]. Provided as an associated function so generic
    /// callers can compute it without invoking [`Default`] (which requires `Sized` plus a concrete
    /// type).
    fn default_variant() -> Self;

    /// Human-readable label for the renderer config window (ImGui combo / radio rows).
    fn label(self) -> &'static str;

    /// Stable string used in `config.toml` and structured logs.
    fn persist_str(self) -> &'static str;

    /// Parses a case-insensitive token into a variant. Accepts the canonical persist string plus
    /// every alias declared on the variant.
    fn parse_persist(s: &str) -> Option<Self>;
}

/// Declarative shortcut for [`LabeledEnum`] enums.
///
/// Generates the enum (with `Clone, Copy, Debug, PartialEq, Eq, Hash`), [`Default`],
/// [`serde::Serialize`], [`serde::Deserialize`] (string visitor -- plus a bool visitor in the
/// bool-aware arm), and the [`LabeledEnum`] impl from one declaration. Variants list their
/// canonical persist string, label, and any alias tokens accepted on input.
///
/// Aliases are matched after lowercasing and trimming, so they should be written lowercase.
///
/// ## Plain string-shape
///
/// ```ignore
/// labeled_enum! {
///     /// MSAA sample count for the main desktop swapchain forward path.
///     pub enum MsaaSampleCount: "MSAA sample count" {
///         default => Off;
///         Off => { persist: "off", label: "1x (off)", aliases: ["1", "1x", "none"] },
///         X2  => { persist: "x2",  label: "2x",       aliases: ["2", "2x"] },
///         X4  => { persist: "x4",  label: "4x",       aliases: ["4", "4x"] },
///         X8  => { persist: "x8",  label: "8x",       aliases: ["8", "8x", "x16", "16", "16x"] },
///     }
/// }
/// ```
///
/// ## Bool-shape (legacy `vsync = true / false`)
///
/// ```ignore
/// labeled_enum! {
///     /// Master vsync mode persisted as `[rendering] vsync`.
///     pub enum VsyncMode: "vsync mode (`off` / `on` / `auto`)" {
///         default    => Off;
///         bool_true  => On;
///         bool_false => Off;
///         Off  => { persist: "off",  label: "Off",  aliases: ["false", "0", "no", "none"] },
///         On   => { persist: "on",   label: "On",   aliases: ["true", "1", "yes", "vsync", "fifo"] },
///         Auto => { persist: "auto", label: "Auto", aliases: ["adaptive", "fifo_relaxed", "fiforelaxed", "relaxed"] },
///     }
/// }
/// ```
#[macro_export]
macro_rules! labeled_enum {
    (
        $(#[$enum_attr:meta])*
        $vis:vis enum $Name:ident : $expecting:literal {
            default => $Default:ident;
            $(
                $(#[$variant_attr:meta])*
                $Variant:ident => {
                    persist: $persist:literal,
                    label: $label:literal
                    $(, aliases: [ $($alias:literal),* $(,)? ])?
                    $(,)?
                }
            ),* $(,)?
        }
    ) => {
        $crate::__labeled_enum_emit! {
            attrs: [$(#[$enum_attr])*],
            vis: $vis,
            name: $Name,
            expecting: $expecting,
            default: $Default,
            bool: (),
            variants: [
                $(
                    {
                        attrs: [$(#[$variant_attr])*],
                        ident: $Variant,
                        persist: $persist,
                        label: $label,
                        aliases: [ $($($alias),*)? ],
                    }
                ),*
            ],
        }
    };

    (
        $(#[$enum_attr:meta])*
        $vis:vis enum $Name:ident : $expecting:literal {
            default    => $Default:ident;
            bool_true  => $BoolTrue:ident;
            bool_false => $BoolFalse:ident;
            $(
                $(#[$variant_attr:meta])*
                $Variant:ident => {
                    persist: $persist:literal,
                    label: $label:literal
                    $(, aliases: [ $($alias:literal),* $(,)? ])?
                    $(,)?
                }
            ),* $(,)?
        }
    ) => {
        $crate::__labeled_enum_emit! {
            attrs: [$(#[$enum_attr])*],
            vis: $vis,
            name: $Name,
            expecting: $expecting,
            default: $Default,
            bool: ($BoolTrue, $BoolFalse),
            variants: [
                $(
                    {
                        attrs: [$(#[$variant_attr])*],
                        ident: $Variant,
                        persist: $persist,
                        label: $label,
                        aliases: [ $($($alias),*)? ],
                    }
                ),*
            ],
        }
    };
}

/// Internal emission helper for [`labeled_enum!`]. Both top-level arms route here with a
/// uniform shape so the generated code is written once.
#[doc(hidden)]
#[macro_export]
macro_rules! __labeled_enum_emit {
    (
        attrs: [$($enum_attr:tt)*],
        vis: $vis:vis,
        name: $Name:ident,
        expecting: $expecting:literal,
        default: $Default:ident,
        bool: ( $($BoolTrue:ident, $BoolFalse:ident)? ),
        variants: [
            $(
                {
                    attrs: [$($variant_attr:tt)*],
                    ident: $Variant:ident,
                    persist: $persist:literal,
                    label: $label:literal,
                    aliases: [ $($alias:literal),* $(,)? ],
                }
            ),* $(,)?
        ],
    ) => {
        $($enum_attr)*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        $vis enum $Name {
            $(
                $($variant_attr)*
                $Variant,
            )*
        }

        impl ::core::default::Default for $Name {
            fn default() -> Self {
                Self::$Default
            }
        }

        impl $Name {
            /// Every variant in declaration order. ImGui pickers and round-trip tests iterate
            /// through this. Mirrors [`crate::config::labeled_enum::LabeledEnum::ALL`] as an
            /// inherent constant so call sites don't need to import the trait.
            pub const ALL: &'static [Self] = &[$(Self::$Variant),*];

            /// Human-readable label for the renderer config window (ImGui combo / radio rows).
            ///
            /// Mirrors [`crate::config::labeled_enum::LabeledEnum::label`] without requiring
            /// the trait to be in scope at call sites.
            pub fn label(self) -> &'static str {
                match self {
                    $(Self::$Variant => $label,)*
                }
            }

            /// Stable string used in `config.toml` and structured logs.
            ///
            /// Mirrors [`crate::config::labeled_enum::LabeledEnum::persist_str`] without
            /// requiring the trait to be in scope at call sites.
            pub fn persist_str(self) -> &'static str {
                match self {
                    $(Self::$Variant => $persist,)*
                }
            }

            /// Parses a case-insensitive token (canonical persist string or any declared alias).
            ///
            /// Matching uses `eq_ignore_ascii_case` so the canonical persist string keeps its
            /// declared casing on serialize while inputs in any case still resolve. Separator
            /// characters (e.g. underscores) must match exactly -- when an enum needs to accept
            /// multiple separator conventions, list each as an explicit alias.
            ///
            /// Mirrors [`crate::config::labeled_enum::LabeledEnum::parse_persist`] without
            /// requiring the trait to be in scope at call sites.
            pub fn parse_persist(s: &str) -> ::core::option::Option<Self> {
                let s = s.trim();
                $(
                    if s.eq_ignore_ascii_case($persist)
                        $(|| s.eq_ignore_ascii_case($alias))*
                    {
                        return ::core::option::Option::Some(Self::$Variant);
                    }
                )*
                ::core::option::Option::None
            }
        }

        impl $crate::config::labeled_enum::LabeledEnum for $Name {
            const ALL: &'static [Self] = <$Name>::ALL;

            fn default_variant() -> Self {
                Self::$Default
            }

            fn label(self) -> &'static str {
                Self::label(self)
            }

            fn persist_str(self) -> &'static str {
                Self::persist_str(self)
            }

            fn parse_persist(s: &str) -> ::core::option::Option<Self> {
                Self::parse_persist(s)
            }
        }

        impl ::serde::Serialize for $Name {
            fn serialize<S: ::serde::Serializer>(
                &self,
                serializer: S,
            ) -> ::core::result::Result<S::Ok, S::Error> {
                serializer.serialize_str(
                    <Self as $crate::config::labeled_enum::LabeledEnum>::persist_str(*self),
                )
            }
        }

        impl<'de> ::serde::Deserialize<'de> for $Name {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> ::core::result::Result<Self, D::Error> {
                struct __Visitor;

                impl<'de> ::serde::de::Visitor<'de> for __Visitor {
                    type Value = $Name;

                    fn expecting(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        f.write_str($expecting)
                    }

                    fn visit_str<E: ::serde::de::Error>(
                        self,
                        v: &str,
                    ) -> ::core::result::Result<Self::Value, E> {
                        <$Name as $crate::config::labeled_enum::LabeledEnum>::parse_persist(v)
                            .ok_or_else(|| E::custom(format!("unknown {}: `{}`", $expecting, v)))
                    }

                    fn visit_string<E: ::serde::de::Error>(
                        self,
                        v: String,
                    ) -> ::core::result::Result<Self::Value, E> {
                        self.visit_str(&v)
                    }

                    $(
                        fn visit_bool<E: ::serde::de::Error>(
                            self,
                            v: bool,
                        ) -> ::core::result::Result<Self::Value, E> {
                            ::core::result::Result::Ok(if v {
                                $Name::$BoolTrue
                            } else {
                                $Name::$BoolFalse
                            })
                        }
                    )?
                }

                deserializer.deserialize_any(__Visitor)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::LabeledEnum;

    labeled_enum! {
        /// Two-variant fixture used by the trait/macro tests below.
        pub enum Fruit: "fruit (`apple`, `banana`)" {
            default => Apple;
            Apple  => { persist: "apple",  label: "Apple",  aliases: ["a", "apl"] },
            Banana => { persist: "banana", label: "Banana" },
        }
    }

    labeled_enum! {
        /// Bool-shape fixture: `true` => `On`, `false` => `Off`.
        pub enum Switch: "switch (`on` / `off`)" {
            default    => Off;
            bool_true  => On;
            bool_false => Off;
            Off => { persist: "off", label: "Off", aliases: ["0", "no", "false"] },
            On  => { persist: "on",  label: "On",  aliases: ["1", "yes", "true"] },
        }
    }

    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    struct FruitHolder {
        v: Fruit,
    }

    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    struct SwitchHolder {
        v: Switch,
    }

    #[test]
    fn default_variant_matches_default_clause() {
        assert_eq!(Fruit::default(), Fruit::Apple);
        assert_eq!(<Fruit as LabeledEnum>::default_variant(), Fruit::Apple);
    }

    #[test]
    fn all_iterates_in_declaration_order() {
        assert_eq!(Fruit::ALL, &[Fruit::Apple, Fruit::Banana]);
    }

    #[test]
    fn parse_persist_accepts_canonical_and_aliases() {
        assert_eq!(Fruit::parse_persist("apple"), Some(Fruit::Apple));
        assert_eq!(Fruit::parse_persist(" APPLE "), Some(Fruit::Apple));
        assert_eq!(Fruit::parse_persist("a"), Some(Fruit::Apple));
        assert_eq!(Fruit::parse_persist("banana"), Some(Fruit::Banana));
        assert_eq!(Fruit::parse_persist("cherry"), None);
    }

    #[test]
    fn serde_roundtrips_through_canonical_persist_str() {
        for variant in Fruit::ALL.iter().copied() {
            let serialized = toml::to_string(&FruitHolder { v: variant }).expect("serialize");
            assert!(
                serialized.contains(&format!("v = \"{}\"", variant.persist_str())),
                "expected canonical persist string in TOML, got: {serialized}"
            );
            let back: FruitHolder = toml::from_str(&serialized).expect("deserialize");
            assert_eq!(back.v, variant);
        }
    }

    #[test]
    fn deserialize_accepts_aliases() {
        let h: FruitHolder = toml::from_str("v = \"a\"").expect("alias");
        assert_eq!(h.v, Fruit::Apple);
    }

    #[test]
    fn bool_shape_routes_bool_visitor() {
        let on: SwitchHolder = toml::from_str("v = true").expect("bool true");
        assert_eq!(on.v, Switch::On);
        let off: SwitchHolder = toml::from_str("v = false").expect("bool false");
        assert_eq!(off.v, Switch::Off);
    }

    #[test]
    fn deserialize_canonical_is_case_insensitive() {
        let h: FruitHolder = toml::from_str("v = \"BANANA\"").expect("uppercase canonical");
        assert_eq!(h.v, Fruit::Banana);
    }
}
