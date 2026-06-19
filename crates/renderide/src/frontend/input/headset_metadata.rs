//! Host-facing headset identity derived from OpenXR runtime and HMD system metadata.

use crate::shared::HeadsetConnection;

/// Host-facing headset metadata forwarded through [`crate::shared::HeadsetState`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HeadsetMetadata {
    /// Host headset connection class.
    pub(crate) connection_type: HeadsetConnection,
    /// Runtime or source string forwarded as the host headset manufacturer/source field.
    pub(crate) headset_manufacturer: Option<String>,
    /// OpenXR system name forwarded as the host headset model field.
    pub(crate) headset_model: Option<String>,
}

impl HeadsetMetadata {
    /// Returns the historical fallback metadata used before OpenXR metadata forwarding.
    pub(crate) fn fallback() -> Self {
        Self {
            connection_type: HeadsetConnection::Wired,
            headset_manufacturer: Some("Renderide".to_string()),
            headset_model: Some("SteamVR".to_string()),
        }
    }

    /// Builds host-facing metadata from OpenXR runtime and HMD system names.
    pub(crate) fn from_openxr(runtime_name: Option<&str>, system_name: Option<&str>) -> Self {
        let headset_manufacturer = clean_metadata_string(runtime_name);
        let headset_model = clean_metadata_string(system_name);

        if headset_manufacturer.is_none() && headset_model.is_none() {
            return Self::fallback();
        }

        Self {
            connection_type: classify_connection(runtime_name, system_name),
            headset_manufacturer,
            headset_model,
        }
    }
}

fn clean_metadata_string(value: Option<&str>) -> Option<String> {
    let trimmed = value?.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn classify_connection(runtime_name: Option<&str>, system_name: Option<&str>) -> HeadsetConnection {
    let runtime_key = normalized_metadata_key(runtime_name.unwrap_or_default());
    let system_key = normalized_metadata_key(system_name.unwrap_or_default());

    if is_steam_link_compatible_identifier(&runtime_key)
        || is_steam_link_compatible_identifier(&system_key)
        || (runtime_key.contains("wivrn") && system_key.contains("questpro"))
    {
        return HeadsetConnection::WirelessSteamLink;
    }

    if is_wireless_streaming_identifier(&runtime_key)
        || is_wireless_streaming_identifier(&system_key)
    {
        return HeadsetConnection::WirelessGeneral;
    }

    HeadsetConnection::Wired
}

fn normalized_metadata_key(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn is_steam_link_compatible_identifier(value: &str) -> bool {
    value.contains("steamlink") || value.contains("vrlink")
}

fn is_wireless_streaming_identifier(value: &str) -> bool {
    value.contains("wivrn")
        || value.contains("alvr")
        || value.contains("aapvr")
        || value.contains("ivry")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steam_link_runtime_maps_to_steam_link_wireless() {
        let metadata = HeadsetMetadata::from_openxr(Some("Steam Link"), Some("Quest Pro"));
        assert_eq!(
            metadata.connection_type,
            HeadsetConnection::WirelessSteamLink
        );
    }

    #[test]
    fn vrlink_quest_pro_maps_to_steam_link_wireless() {
        let metadata = HeadsetMetadata::from_openxr(Some("SteamVR"), Some("VRLINKHMDQUESTPRO"));
        assert_eq!(
            metadata.connection_type,
            HeadsetConnection::WirelessSteamLink
        );
    }

    #[test]
    fn wivrn_quest_pro_maps_to_steam_link_wireless() {
        let metadata = HeadsetMetadata::from_openxr(Some("WiVRn"), Some("Meta Quest Pro"));
        assert_eq!(
            metadata.connection_type,
            HeadsetConnection::WirelessSteamLink
        );
    }

    #[test]
    fn wivrn_other_headset_maps_to_general_wireless() {
        let metadata = HeadsetMetadata::from_openxr(Some("WiVRn"), Some("Quest 3"));
        assert_eq!(metadata.connection_type, HeadsetConnection::WirelessGeneral);
    }

    #[test]
    fn unknown_runtime_maps_to_wired() {
        let metadata = HeadsetMetadata::from_openxr(Some("Monado"), Some("Generic HMD"));
        assert_eq!(metadata.connection_type, HeadsetConnection::Wired);
    }

    #[test]
    fn empty_openxr_strings_use_fallback_metadata() {
        let metadata = HeadsetMetadata::from_openxr(Some(" "), Some(""));
        assert_eq!(metadata, HeadsetMetadata::fallback());
    }

    #[test]
    fn metadata_strings_are_trimmed_before_forwarding() {
        let metadata = HeadsetMetadata::from_openxr(Some("  WiVRn  "), Some("  Quest 3  "));
        assert_eq!(metadata.headset_manufacturer.as_deref(), Some("WiVRn"));
        assert_eq!(metadata.headset_model.as_deref(), Some("Quest 3"));
    }
}
