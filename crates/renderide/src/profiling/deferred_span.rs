//! Deferred CPU span handles for profiler scopes that cross stack-frame boundaries.

/// CPU profiling span that can stay open after the creating function returns.
#[derive(Default)]
pub(crate) struct DeferredCpuSpan {
    #[cfg(feature = "tracy")]
    span: Option<tracy_client::Span>,
}

impl DeferredCpuSpan {
    /// Starts the span when no span is already open.
    #[inline]
    pub(crate) fn begin_if_empty(&mut self, name: &'static str) {
        if self.is_open() {
            return;
        }
        self.begin(name);
    }

    /// Starts the span, ending any span already held by this handle.
    #[inline]
    pub(crate) fn begin(&mut self, name: &'static str) {
        self.end();
        #[cfg(feature = "tracy")]
        {
            let function = "renderide::profiling::DeferredCpuSpan";
            if let Some(client) = tracy_client::Client::running() {
                self.span = Some(client.span_alloc(Some(name), function, file!(), line!(), 0));
            }
        }
        #[cfg(not(feature = "tracy"))]
        {
            let _ = name;
        }
    }

    /// Ends the span if one is currently open.
    #[inline]
    pub(crate) fn end(&mut self) {
        #[cfg(feature = "tracy")]
        {
            self.span = None;
        }
    }

    /// Returns whether this handle currently owns an open span.
    #[inline]
    pub(crate) fn is_open(&self) -> bool {
        #[cfg(feature = "tracy")]
        {
            self.span.is_some()
        }
        #[cfg(not(feature = "tracy"))]
        {
            false
        }
    }
}
