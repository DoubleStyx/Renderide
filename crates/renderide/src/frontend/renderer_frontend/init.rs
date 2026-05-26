//! Init handshake and session-flag methods on [`RendererFrontend`].

use crate::shared::RendererInitData;

use super::super::init_state::InitState;
use super::RendererFrontend;

impl RendererFrontend {
    /// Host requested an orderly renderer exit.
    pub fn shutdown_requested(&self) -> bool {
        self.session.shutdown_requested()
    }

    /// Records a host shutdown request.
    pub fn set_shutdown_requested(&mut self, value: bool) {
        self.session.set_shutdown_requested(value);
    }

    /// Unrecoverable IPC/init ordering error; stops begin-frame until reset.
    pub fn fatal_error(&self) -> bool {
        self.session.fatal_error()
    }

    /// Marks a fatal IPC/init error.
    pub fn set_fatal_error(&mut self, value: bool) {
        self.session.set_fatal_error(value);
    }

    /// Current host/renderer init handshake phase.
    pub fn init_state(&self) -> InitState {
        self.session.init_state()
    }

    /// Updates the init handshake phase.
    pub fn set_init_state(&mut self, state: InitState) {
        self.session.set_init_state(state);
    }

    /// Host [`RendererInitData`] waiting to be consumed after the SHM accessor is ready.
    pub fn pending_init(&self) -> Option<&RendererInitData> {
        self.session.pending_init()
    }

    /// Stores init payload until the runtime attaches shared memory and finalizes setup.
    pub fn set_pending_init(&mut self, data: RendererInitData) {
        self.session.set_pending_init(data);
    }

    /// Removes and returns pending init data once the consumer is ready.
    pub fn take_pending_init(&mut self) -> Option<RendererInitData> {
        self.session.take_pending_init()
    }

    /// Marks init received after `renderer_init_data`.
    pub fn on_init_received(&mut self) {
        self.session.mark_init_received();
        self.lockstep.mark_init_received();
    }

    /// Marks the host engine as ready for strict frame lockstep gating.
    pub fn on_renderer_engine_ready(&mut self) {
        self.lockstep.activate_host_lockstep();
    }
}
