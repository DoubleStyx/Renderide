//! Startup graphics API preference (`[rendering] graphics_api`).

use crate::labeled_enum;

labeled_enum! {
    /// Startup graphics API preference persisted in `config.toml` as `[rendering] graphics_api`.
    ///
    /// `Auto` preserves wgpu's normal backend discovery. Explicit API choices constrain the first
    /// adapter-selection attempt to that backend; if no compatible adapter exists, startup retries
    /// automatic backend discovery. The setting is read only when the GPU stack is created, so
    /// changes made through the config HUD apply on the next renderer restart.
    pub enum GraphicsApiSetting: "graphics API (`auto` / `vulkan` / `dx12` / `metal` / `gl`)" {
        default => Auto;

        /// Let wgpu enumerate every enabled backend.
        Auto => {
            persist: "auto",
            label: "Auto",
            aliases: ["default", "automatic"],
        },
        /// Prefer Vulkan.
        Vulkan => {
            persist: "vulkan",
            label: "Vulkan",
            aliases: ["vk"],
        },
        /// Prefer Direct3D 12.
        Dx12 => {
            persist: "dx12",
            label: "DirectX 12",
            aliases: ["d3d12", "directx12", "direct3d12"],
        },
        /// Prefer Metal.
        Metal => {
            persist: "metal",
            label: "Metal",
            aliases: ["mtl"],
        },
        /// Prefer OpenGL / OpenGL ES.
        Gl => {
            persist: "gl",
            label: "OpenGL",
            aliases: ["opengl", "gles"],
        },
    }
}

impl GraphicsApiSetting {
    /// Stable string for TOML / logs. Historical-style alias for [`Self::persist_str`].
    pub fn as_persist_str(self) -> &'static str {
        self.persist_str()
    }

    /// Parses case-insensitive persisted or UI tokens. Historical-style alias for
    /// [`Self::parse_persist`].
    #[cfg(test)]
    pub fn from_persist_str(s: &str) -> Option<Self> {
        Self::parse_persist(s)
    }

    /// Initial backend set requested from wgpu before `WGPU_BACKEND` environment overrides apply.
    pub fn requested_backends(self) -> wgpu::Backends {
        match self {
            Self::Auto => wgpu::Backends::all(),
            Self::Vulkan => wgpu::Backends::VULKAN,
            Self::Dx12 => wgpu::Backends::DX12,
            Self::Metal => wgpu::Backends::METAL,
            Self::Gl => wgpu::Backends::GL,
        }
    }

    /// Whether startup should retry automatic backend selection when this choice cannot produce a
    /// usable adapter for the active target.
    pub fn should_retry_auto_on_adapter_failure(self) -> bool {
        self != Self::Auto
    }

    /// Whether the current OpenXR path can directly honor this API preference.
    pub fn is_openxr_compatible(self) -> bool {
        matches!(self, Self::Auto | Self::Vulkan)
    }
}

#[cfg(test)]
mod tests {
    use super::GraphicsApiSetting;
    use crate::config::RendererSettings;

    #[test]
    fn tokens_parse_to_graphics_api() {
        assert_eq!(
            GraphicsApiSetting::from_persist_str("auto"),
            Some(GraphicsApiSetting::Auto)
        );
        assert_eq!(
            GraphicsApiSetting::from_persist_str("VK"),
            Some(GraphicsApiSetting::Vulkan)
        );
        assert_eq!(
            GraphicsApiSetting::from_persist_str("d3d12"),
            Some(GraphicsApiSetting::Dx12)
        );
        assert_eq!(
            GraphicsApiSetting::from_persist_str("mtl"),
            Some(GraphicsApiSetting::Metal)
        );
        assert_eq!(
            GraphicsApiSetting::from_persist_str("opengl"),
            Some(GraphicsApiSetting::Gl)
        );
        assert_eq!(GraphicsApiSetting::from_persist_str(""), None);
    }

    #[test]
    fn maps_to_expected_wgpu_backend_sets() {
        assert_eq!(
            GraphicsApiSetting::Auto.requested_backends(),
            wgpu::Backends::all()
        );
        assert_eq!(
            GraphicsApiSetting::Vulkan.requested_backends(),
            wgpu::Backends::VULKAN
        );
        assert_eq!(
            GraphicsApiSetting::Dx12.requested_backends(),
            wgpu::Backends::DX12
        );
        assert_eq!(
            GraphicsApiSetting::Metal.requested_backends(),
            wgpu::Backends::METAL
        );
        assert_eq!(
            GraphicsApiSetting::Gl.requested_backends(),
            wgpu::Backends::GL
        );
    }

    #[test]
    fn explicit_apis_retry_auto_but_auto_does_not() {
        assert!(!GraphicsApiSetting::Auto.should_retry_auto_on_adapter_failure());
        for api in [
            GraphicsApiSetting::Vulkan,
            GraphicsApiSetting::Dx12,
            GraphicsApiSetting::Metal,
            GraphicsApiSetting::Gl,
        ] {
            assert!(api.should_retry_auto_on_adapter_failure());
        }
    }

    #[test]
    fn openxr_accepts_auto_or_vulkan_only() {
        assert!(GraphicsApiSetting::Auto.is_openxr_compatible());
        assert!(GraphicsApiSetting::Vulkan.is_openxr_compatible());
        assert!(!GraphicsApiSetting::Dx12.is_openxr_compatible());
        assert!(!GraphicsApiSetting::Metal.is_openxr_compatible());
        assert!(!GraphicsApiSetting::Gl.is_openxr_compatible());
    }

    #[test]
    fn graphics_api_toml_roundtrip() {
        for api in GraphicsApiSetting::ALL.iter().copied() {
            let mut s = RendererSettings::default();
            s.rendering.graphics_api = api;
            let toml = toml::to_string(&s).expect("serialize");
            let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
            assert_eq!(back.rendering.graphics_api, api);
        }
    }

    #[test]
    fn missing_graphics_api_loads_as_auto() {
        let toml = "[rendering]\npresentation_mode = \"fifo\"\n";
        let parsed: RendererSettings = toml::from_str(toml).expect("config without field");
        assert_eq!(parsed.rendering.graphics_api, GraphicsApiSetting::Auto);
    }
}
