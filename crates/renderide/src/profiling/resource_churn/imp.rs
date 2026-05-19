//! Tracy-enabled implementation of the resource-churn counters.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::Mutex;

use super::ResourceChurnKind;

static REGISTRY: OnceLock<Mutex<Vec<&'static ResourceChurnSite>>> = OnceLock::new();
static TOTAL_PLOTS_CONFIGURED: AtomicBool = AtomicBool::new(false);

/// Static counter state for one resource creation site.
pub(crate) struct ResourceChurnSite {
    kind: ResourceChurnKind,
    label: &'static str,
    count: AtomicU64,
    registered: AtomicBool,
    plot_name: OnceLock<tracy_client::PlotName>,
    plot_configured: AtomicBool,
}

impl ResourceChurnSite {
    /// Creates a resource-churn site for a static resource creation call.
    pub(crate) const fn new(kind: ResourceChurnKind, label: &'static str) -> Self {
        Self {
            kind,
            label,
            count: AtomicU64::new(0),
            registered: AtomicBool::new(false),
            plot_name: OnceLock::new(),
            plot_configured: AtomicBool::new(false),
        }
    }

    /// Records one resource creation at this site.
    #[inline]
    pub(crate) fn note(&'static self) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.register();
    }

    fn register(&'static self) {
        if self
            .registered
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            registry().lock().push(self);
        }
    }

    fn take_count(&self) -> u64 {
        self.count.swap(0, Ordering::AcqRel)
    }

    #[cfg(test)]
    fn pending_count(&self) -> u64 {
        self.count.load(Ordering::Acquire)
    }

    fn plot_name(&self) -> tracy_client::PlotName {
        *self.plot_name.get_or_init(|| {
            tracy_client::PlotName::new_leak(format!(
                "resource_churn::{}::{}",
                self.kind.plot_segment(),
                self.label
            ))
        })
    }

    fn configure_plot_if_needed(&self, client: &tracy_client::Client) {
        if self
            .plot_configured
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            client.plot_config(
                self.plot_name(),
                tracy_client::PlotConfiguration::default()
                    .line_style(tracy_client::PlotLineStyle::Stepped),
            );
        }
    }

    fn plot(&self, client: &tracy_client::Client, count: u64) {
        self.configure_plot_if_needed(client);
        client.plot(self.plot_name(), count as f64);
    }
}

impl ResourceChurnKind {
    fn plot_segment(self) -> &'static str {
        match self {
            Self::Buffer => "buffer",
            Self::BindGroup => "bind_group",
            Self::Texture => "texture",
            Self::TextureView => "texture_view",
            Self::Sampler => "sampler",
            Self::RenderPipeline => "render_pipeline",
            Self::ComputePipeline => "compute_pipeline",
        }
    }
}

/// Plots all registered resource creation sites and resets their counters for the next frame.
#[inline]
pub(crate) fn flush_resource_churn_plots() {
    let mut totals = ResourceChurnTotals::default();
    let Some(client) = tracy_client::Client::running() else {
        reset_without_plotting(&mut totals);
        return;
    };

    if let Some(registry) = REGISTRY.get() {
        for site in registry.lock().iter().copied() {
            let count = site.take_count();
            totals.add(site.kind, count);
            site.plot(&client, count);
        }
    }
    configure_total_plots_if_needed(&client);
    totals.plot(&client);
}

fn reset_without_plotting(totals: &mut ResourceChurnTotals) {
    if let Some(registry) = REGISTRY.get() {
        for site in registry.lock().iter().copied() {
            totals.add(site.kind, site.take_count());
        }
    }
}

fn registry() -> &'static Mutex<Vec<&'static ResourceChurnSite>> {
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

fn configure_total_plots_if_needed(client: &tracy_client::Client) {
    if TOTAL_PLOTS_CONFIGURED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    client.plot_config(
        tracy_client::plot_name!("resource_churn::buffers_total"),
        stepped_plot_config(),
    );
    client.plot_config(
        tracy_client::plot_name!("resource_churn::bind_groups_total"),
        stepped_plot_config(),
    );
    client.plot_config(
        tracy_client::plot_name!("resource_churn::textures_total"),
        stepped_plot_config(),
    );
    client.plot_config(
        tracy_client::plot_name!("resource_churn::texture_views_total"),
        stepped_plot_config(),
    );
    client.plot_config(
        tracy_client::plot_name!("resource_churn::samplers_total"),
        stepped_plot_config(),
    );
    client.plot_config(
        tracy_client::plot_name!("resource_churn::render_pipelines_total"),
        stepped_plot_config(),
    );
    client.plot_config(
        tracy_client::plot_name!("resource_churn::compute_pipelines_total"),
        stepped_plot_config(),
    );
    client.plot_config(
        tracy_client::plot_name!("resource_churn::pipelines_total"),
        stepped_plot_config(),
    );
}

fn stepped_plot_config() -> tracy_client::PlotConfiguration {
    tracy_client::PlotConfiguration::default().line_style(tracy_client::PlotLineStyle::Stepped)
}

#[derive(Clone, Copy, Debug, Default)]
struct ResourceChurnTotals {
    buffers: u64,
    bind_groups: u64,
    textures: u64,
    texture_views: u64,
    samplers: u64,
    render_pipelines: u64,
    compute_pipelines: u64,
}

impl ResourceChurnTotals {
    fn add(&mut self, kind: ResourceChurnKind, count: u64) {
        match kind {
            ResourceChurnKind::Buffer => {
                self.buffers = self.buffers.saturating_add(count);
            }
            ResourceChurnKind::BindGroup => {
                self.bind_groups = self.bind_groups.saturating_add(count);
            }
            ResourceChurnKind::Texture => {
                self.textures = self.textures.saturating_add(count);
            }
            ResourceChurnKind::TextureView => {
                self.texture_views = self.texture_views.saturating_add(count);
            }
            ResourceChurnKind::Sampler => {
                self.samplers = self.samplers.saturating_add(count);
            }
            ResourceChurnKind::RenderPipeline => {
                self.render_pipelines = self.render_pipelines.saturating_add(count);
            }
            ResourceChurnKind::ComputePipeline => {
                self.compute_pipelines = self.compute_pipelines.saturating_add(count);
            }
        }
    }

    fn pipelines(self) -> u64 {
        self.render_pipelines.saturating_add(self.compute_pipelines)
    }

    fn plot(self, client: &tracy_client::Client) {
        client.plot(
            tracy_client::plot_name!("resource_churn::buffers_total"),
            self.buffers as f64,
        );
        client.plot(
            tracy_client::plot_name!("resource_churn::bind_groups_total"),
            self.bind_groups as f64,
        );
        client.plot(
            tracy_client::plot_name!("resource_churn::textures_total"),
            self.textures as f64,
        );
        client.plot(
            tracy_client::plot_name!("resource_churn::texture_views_total"),
            self.texture_views as f64,
        );
        client.plot(
            tracy_client::plot_name!("resource_churn::samplers_total"),
            self.samplers as f64,
        );
        client.plot(
            tracy_client::plot_name!("resource_churn::render_pipelines_total"),
            self.render_pipelines as f64,
        );
        client.plot(
            tracy_client::plot_name!("resource_churn::compute_pipelines_total"),
            self.compute_pipelines as f64,
        );
        client.plot(
            tracy_client::plot_name!("resource_churn::pipelines_total"),
            self.pipelines() as f64,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_SITE: ResourceChurnSite =
        ResourceChurnSite::new(ResourceChurnKind::Buffer, "profiling::resource_churn_test");

    #[test]
    fn flush_resets_registered_site_counts() {
        flush_resource_churn_plots();
        TEST_SITE.note();
        TEST_SITE.note();
        assert_eq!(TEST_SITE.pending_count(), 2);

        flush_resource_churn_plots();

        assert_eq!(TEST_SITE.pending_count(), 0);
    }
}
