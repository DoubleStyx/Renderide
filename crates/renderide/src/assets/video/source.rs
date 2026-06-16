//! Host video source normalization for GStreamer playbin.

use std::path::{Path, PathBuf};

/// Returns `true` when `source` already has a URI scheme.
pub(super) fn is_uri_source(source: &str) -> bool {
    source.contains("://")
}

/// Converts a host source string into a playbin URI.
pub(super) fn source_uri(source: Option<&str>) -> Result<Option<String>, gstreamer::glib::Error> {
    let Some(source) = source else {
        return Ok(None);
    };
    if is_uri_source(source) {
        return Ok(Some(source.to_owned()));
    }
    gstreamer::glib::filename_to_uri(local_source_path(source), None)
        .map(|uri| Some(uri.to_string()))
}

/// Returns an absolute local path for GLib URI conversion when possible.
pub(super) fn local_source_path(source: &str) -> PathBuf {
    let path = Path::new(source);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    match std::env::current_dir() {
        Ok(cwd) => cwd.join(path),
        Err(_) => path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_source_yields_no_uri() {
        assert_eq!(source_uri(None).unwrap(), None);
    }

    #[test]
    fn uri_sources_are_preserved_directly() {
        for source in [
            "https://93.184.216.34/movie.mp4",
            "rtsp://192.168.1.20/stream",
            "rtmp://example.com/live/stream",
            "file:///tmp/video.mp4",
        ] {
            assert_eq!(source_uri(Some(source)).unwrap(), Some(source.to_owned()));
        }
    }

    #[test]
    fn absolute_local_path_converts_to_file_uri() {
        let path = std::env::current_dir().unwrap().join("renderide-video.mp4");
        let expected = gstreamer::glib::filename_to_uri(&path, None)
            .unwrap()
            .to_string();

        assert_eq!(
            source_uri(Some(path.to_str().unwrap())).unwrap(),
            Some(expected)
        );
    }

    #[test]
    fn relative_local_path_converts_to_absolute_file_uri() {
        let source = "relative/video.mp4";
        let path = local_source_path(source);
        let expected = gstreamer::glib::filename_to_uri(&path, None)
            .unwrap()
            .to_string();

        assert!(path.is_absolute());
        assert!(path.ends_with(source));
        assert_eq!(source_uri(Some(source)).unwrap(), Some(expected));
    }
}
