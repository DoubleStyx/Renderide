//! Shader default render queue directive parsing.

use super::super::error::BuildError;

const UNITY_RENDER_QUEUE_BACKGROUND: i32 = 1000;
const UNITY_RENDER_QUEUE_GEOMETRY: i32 = 2000;
const UNITY_RENDER_QUEUE_ALPHA_TEST: i32 = 2450;
const UNITY_RENDER_QUEUE_TRANSPARENT: i32 = 3000;
const UNITY_RENDER_QUEUE_OVERLAY: i32 = 4000;

/// Shader default render queue parsed from a `//#render_queue` directive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in super::super) struct RenderQueueDirective {
    /// Unity render queue value after applying the optional named-queue offset.
    pub queue: i32,
}

/// Parses the optional shader default render queue directive from WGSL source.
pub(in super::super) fn parse_render_queue_directive(
    source: &str,
    file: &str,
) -> Result<Option<RenderQueueDirective>, BuildError> {
    let mut directive = None;
    for (line_idx, line) in source.lines().enumerate() {
        let line_no = line_idx + 1;
        let Some(rest) = line.trim_start().strip_prefix("//#render_queue") else {
            continue;
        };
        if directive.is_some() {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: duplicate `//#render_queue` directive"
            )));
        }
        let expression = rest.split_whitespace().collect::<String>();
        if expression.is_empty() {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#render_queue` requires a Unity queue name"
            )));
        }
        directive = Some(RenderQueueDirective {
            queue: parse_render_queue_expression(&expression, file, line_no)?,
        });
    }
    Ok(directive)
}

fn parse_render_queue_expression(value: &str, file: &str, line: usize) -> Result<i32, BuildError> {
    let (queue_name, offset) = if let Some((offset_start, sign)) =
        value.char_indices().find(|(_, c)| *c == '+' || *c == '-')
    {
        let queue_name = &value[..offset_start];
        let offset = value[offset_start..].parse::<i32>().map_err(|err| {
            BuildError::Message(format!(
                "{file}:{line}: `//#render_queue` offset must be an integer, got `{sign}{}`: {err}",
                &value[offset_start + sign.len_utf8()..]
            ))
        })?;
        (queue_name, offset)
    } else {
        (value, 0)
    };
    let base = match queue_name.to_ascii_lowercase().as_str() {
        "background" => UNITY_RENDER_QUEUE_BACKGROUND,
        "geometry" => UNITY_RENDER_QUEUE_GEOMETRY,
        "alphatest" => UNITY_RENDER_QUEUE_ALPHA_TEST,
        "transparent" => UNITY_RENDER_QUEUE_TRANSPARENT,
        "overlay" => UNITY_RENDER_QUEUE_OVERLAY,
        _ => {
            return Err(BuildError::Message(format!(
                "{file}:{line}: unknown `//#render_queue` queue `{queue_name}` (allowed: Background, Geometry, AlphaTest, Transparent, Overlay)"
            )));
        }
    };
    base.checked_add(offset).ok_or_else(|| {
        BuildError::Message(format!(
            "{file}:{line}: `//#render_queue` value `{value}` overflows i32"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_named_render_queue() -> Result<(), BuildError> {
        assert_eq!(
            parse_render_queue_directive("//#render_queue Transparent\n", "test.wgsl")?,
            Some(RenderQueueDirective { queue: 3000 })
        );
        Ok(())
    }

    #[test]
    fn parses_positive_and_negative_offsets() -> Result<(), BuildError> {
        assert_eq!(
            parse_render_queue_directive("//#render_queue AlphaTest+200\n", "test.wgsl")?,
            Some(RenderQueueDirective { queue: 2650 })
        );
        assert_eq!(
            parse_render_queue_directive("//#render_queue Transparent - 100\n", "test.wgsl")?,
            Some(RenderQueueDirective { queue: 2900 })
        );
        Ok(())
    }

    #[test]
    fn rejects_unknown_render_queue() {
        let err = parse_render_queue_directive("//#render_queue Lit\n", "test.wgsl")
            .expect_err("unknown queue should fail");

        assert!(err.to_string().contains("unknown `//#render_queue` queue"));
    }

    #[test]
    fn rejects_duplicate_render_queue_directives() {
        let err = parse_render_queue_directive(
            "//#render_queue Geometry\n//#render_queue Overlay\n",
            "test.wgsl",
        )
        .expect_err("duplicate directives should fail");

        assert!(err.to_string().contains("duplicate `//#render_queue`"));
    }
}
