//! Host video source normalization for GStreamer playbin.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};

use url::{Host, Url};

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
        return Ok(allowed_network_uri(source));
    }
    logger::warn!("video texture source rejected: local paths are not allowed by default");
    Ok(None)
}

fn allowed_network_uri(source: &str) -> Option<String> {
    allowed_network_uri_with_resolver(source, resolve_domain_host)
}

fn allowed_network_uri_with_resolver<F>(source: &str, mut resolve_domain: F) -> Option<String>
where
    F: FnMut(&str, u16) -> Option<Vec<IpAddr>>,
{
    let Ok(url) = Url::parse(source) else {
        logger::warn!("video texture URI rejected: malformed URI");
        return None;
    };
    if !matches!(url.scheme(), "http" | "https") {
        logger::warn!(
            "video texture URI rejected: unsupported scheme {}",
            url.scheme()
        );
        return None;
    }
    let Some(host) = url.host() else {
        logger::warn!("video texture URI rejected: missing host");
        return None;
    };
    if host_is_blocked(&host) {
        logger::warn!("video texture URI rejected: blocked host");
        return None;
    }
    if let Host::Domain(domain) = host {
        let Some(port) = url.port_or_known_default() else {
            logger::warn!("video texture URI rejected: missing known port");
            return None;
        };
        let domain = domain.trim_end_matches('.');
        let Some(resolved) = resolve_domain(domain, port) else {
            logger::warn!("video texture URI rejected: host resolution failed");
            return None;
        };
        if resolved.is_empty() {
            logger::warn!("video texture URI rejected: host resolution returned no addresses");
            return None;
        }
        if resolved.iter().copied().any(ip_is_blocked) {
            logger::warn!("video texture URI rejected: host resolves to a blocked address");
            return None;
        }
    }
    Some(url.to_string())
}

fn resolve_domain_host(domain: &str, port: u16) -> Option<Vec<IpAddr>> {
    (domain, port)
        .to_socket_addrs()
        .ok()
        .map(|addrs| addrs.map(|addr| addr.ip()).collect())
}

fn host_is_blocked(host: &Host<&str>) -> bool {
    match host {
        Host::Domain(domain) => {
            let domain = domain.trim_end_matches('.').to_ascii_lowercase();
            domain == "localhost" || domain.ends_with(".localhost")
        }
        Host::Ipv4(ip) => ipv4_is_blocked(*ip),
        Host::Ipv6(ip) => ipv6_is_blocked(*ip),
    }
}

fn ip_is_blocked(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ipv4_is_blocked(ip),
        IpAddr::V6(ip) => ipv6_is_blocked(ip),
    }
}

fn ipv4_is_blocked(ip: Ipv4Addr) -> bool {
    ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ipv4_is_documentation(ip)
        || ip.is_unspecified()
}

fn ipv4_is_documentation(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    matches!(
        octets,
        [192, 0, 2, _] | [198, 51, 100, _] | [203, 0, 113, _]
    )
}

fn ipv6_is_blocked(ip: Ipv6Addr) -> bool {
    if let Some(ipv4) = ip.to_ipv4_mapped() {
        return ipv4_is_blocked(ipv4);
    }
    ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_unique_local()
        || ip.is_unicast_link_local()
        || ipv6_is_documentation(ip)
}

fn ipv6_is_documentation(ip: Ipv6Addr) -> bool {
    ip.segments()[0] == 0x2001 && ip.segments()[1] == 0x0db8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_source_yields_no_uri() {
        assert_eq!(source_uri(None).unwrap(), None);
    }

    #[test]
    fn public_http_uri_source_is_preserved_directly() {
        assert_eq!(
            source_uri(Some("https://93.184.216.34/movie.mp4")).unwrap(),
            Some(String::from("https://93.184.216.34/movie.mp4"))
        );
    }

    #[test]
    fn local_path_is_rejected() {
        assert_eq!(source_uri(Some("/tmp/renderide-video.mp4")).unwrap(), None);
        assert_eq!(source_uri(Some("relative/video.mp4")).unwrap(), None);
    }

    #[test]
    fn unsafe_uri_sources_are_rejected() {
        assert_eq!(source_uri(Some("file:///tmp/video.mp4")).unwrap(), None);
        assert_eq!(
            source_uri(Some("ftp://example.invalid/video.mp4")).unwrap(),
            None
        );
        assert_eq!(
            source_uri(Some("http://127.0.0.1/video.mp4")).unwrap(),
            None
        );
        assert_eq!(source_uri(Some("http://10.0.0.5/video.mp4")).unwrap(), None);
        assert_eq!(
            source_uri(Some("http://169.254.169.254/video.mp4")).unwrap(),
            None
        );
        assert_eq!(
            source_uri(Some("http://localhost/video.mp4")).unwrap(),
            None
        );
        assert_eq!(
            source_uri(Some("http://media.localhost/video.mp4")).unwrap(),
            None
        );
        assert_eq!(
            source_uri(Some("http://[::ffff:127.0.0.1]/video.mp4")).unwrap(),
            None
        );
    }

    #[test]
    fn ip_policy_blocks_local_networks() {
        assert!(ip_is_blocked(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))));
        assert!(ip_is_blocked(IpAddr::V4(Ipv4Addr::LOCALHOST)));
        assert!(ip_is_blocked(IpAddr::V6(Ipv6Addr::LOCALHOST)));
        assert!(ip_is_blocked(IpAddr::V6(
            "::ffff:127.0.0.1".parse().unwrap()
        )));
        assert!(!ip_is_blocked(IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34))));
    }

    #[test]
    fn public_domain_uri_is_allowed_when_resolution_is_public() {
        assert_eq!(
            allowed_network_uri_with_resolver("https://media.example/video.mp4", |domain, port| {
                assert_eq!(domain, "media.example");
                assert_eq!(port, 443);
                Some(vec![IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34))])
            }),
            Some(String::from("https://media.example/video.mp4"))
        );
    }

    #[test]
    fn domain_uri_is_rejected_when_resolution_fails() {
        assert_eq!(
            allowed_network_uri_with_resolver(
                "https://media.example/video.mp4",
                |_domain, _port| { None }
            ),
            None
        );
    }

    #[test]
    fn domain_uri_is_rejected_when_resolution_is_private() {
        assert_eq!(
            allowed_network_uri_with_resolver(
                "https://media.example/video.mp4",
                |_domain, _port| { Some(vec![IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5))]) }
            ),
            None
        );
    }

    #[test]
    fn domain_uri_is_rejected_when_any_resolution_is_private() {
        assert_eq!(
            allowed_network_uri_with_resolver(
                "https://media.example/video.mp4",
                |_domain, _port| {
                    Some(vec![
                        IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34)),
                        IpAddr::V4(Ipv4Addr::new(192, 168, 1, 10)),
                    ])
                }
            ),
            None
        );
    }
}
