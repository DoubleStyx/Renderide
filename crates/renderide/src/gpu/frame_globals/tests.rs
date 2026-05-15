//! Layout assertions and packing tests for [`super::FrameGpuUniforms`].

#[cfg(test)]
mod offset_and_packing_tests {
    use super::super::clustered::ClusteredFrameGlobalsParams;
    use super::super::skybox_specular::SkyboxSpecularUniformParams;
    use super::super::uniforms::{
        FRAME_PROJECTION_FLAG_ORTHOGRAPHIC, FRAME_TAIL_AMBIENT_SH_VALID, FrameGpuUniforms,
        frame_tail_sample_count, pack_frame_tail_flags,
    };
    use crate::shared::RenderSH2;
    use glam::Mat4;

    #[test]
    fn frame_globals_size_304() {
        assert_eq!(size_of::<FrameGpuUniforms>(), 304);
        assert_eq!(size_of::<FrameGpuUniforms>() % 16, 0);
    }

    #[test]
    fn frame_globals_offsets_match_wgsl_layout() {
        assert_eq!(std::mem::offset_of!(FrameGpuUniforms, camera_world_pos), 0);
        assert_eq!(
            std::mem::offset_of!(FrameGpuUniforms, camera_world_pos_right),
            16
        );
        assert_eq!(
            std::mem::offset_of!(FrameGpuUniforms, view_space_z_coeffs),
            32
        );
        assert_eq!(
            std::mem::offset_of!(FrameGpuUniforms, view_space_z_coeffs_right),
            48
        );
        assert_eq!(std::mem::offset_of!(FrameGpuUniforms, cluster_count_x), 64);
        assert_eq!(std::mem::offset_of!(FrameGpuUniforms, proj_params_left), 96);
        assert_eq!(
            std::mem::offset_of!(FrameGpuUniforms, proj_params_right),
            112
        );
        assert_eq!(std::mem::offset_of!(FrameGpuUniforms, frame_tail), 128);
        assert_eq!(std::mem::offset_of!(FrameGpuUniforms, skybox_specular), 144);
        assert_eq!(std::mem::offset_of!(FrameGpuUniforms, ambient_sh), 160);
    }

    #[test]
    fn z_coeffs_extracts_third_row_for_translation_only_view() {
        // Translation-only view: world-to-view z = world.z + tz (tz from row 3, w component).
        let tz = 7.0;
        let m = Mat4::from_translation(glam::Vec3::new(0.0, 0.0, tz));
        let coeffs = FrameGpuUniforms::view_space_z_coeffs_from_world_to_view(m);
        assert_eq!(coeffs, [0.0, 0.0, 1.0, tz]);

        // Sanity: dot(coeffs.xyz, p) + coeffs.w matches (m * p).z for a sample point.
        let p = glam::Vec3::new(2.0, -3.0, 4.0);
        let view_z = (m * p.extend(1.0)).z;
        let dotted = coeffs[2].mul_add(p.z, coeffs[0].mul_add(p.x, coeffs[1] * p.y)) + coeffs[3];
        assert!((view_z - dotted).abs() < 1e-6);
    }

    #[test]
    fn z_coeffs_matches_third_component_under_yaw_rotation() {
        // Yaw should leave Z row invariant (rotation about Y keeps Z-basis).
        let m = Mat4::from_rotation_y(std::f32::consts::FRAC_PI_3);
        let coeffs = FrameGpuUniforms::view_space_z_coeffs_from_world_to_view(m);
        let p = glam::Vec3::new(1.5, -0.25, 2.0);
        let view_z = (m * p.extend(1.0)).z;
        let dotted = coeffs[2].mul_add(p.z, coeffs[0].mul_add(p.x, coeffs[1] * p.y)) + coeffs[3];
        assert!((view_z - dotted).abs() < 1e-5);
    }

    #[test]
    fn proj_params_extract_diagonal_and_offcenter_are_zero_for_symmetric() {
        // Symmetric perspective: [0][2] and [1][2] are zero.
        let p = Mat4::perspective_rh(60.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
        let coeffs = FrameGpuUniforms::proj_params_from_proj(p);
        assert!(coeffs[0].abs() > 0.0);
        assert!(coeffs[1].abs() > 0.0);
        assert!(coeffs[2].abs() < 1e-5);
        assert!(coeffs[3].abs() < 1e-5);
    }

    #[test]
    fn new_clustered_populates_fields_including_zero_w_for_camera_pos() {
        let params = ClusteredFrameGlobalsParams {
            camera_world_pos: glam::Vec3::new(1.0, 2.0, 3.0),
            camera_world_pos_right: glam::Vec3::new(4.0, 5.0, 6.0),
            view_space_z_coeffs: [0.1, 0.2, 0.3, 0.4],
            view_space_z_coeffs_right: [0.5, 0.6, 0.7, 0.8],
            cluster_count_x: 16,
            cluster_count_y: 9,
            cluster_count_z: 24,
            near_clip: 0.05,
            far_clip: 1000.0,
            light_count: 42,
            viewport_width: 1920,
            viewport_height: 1080,
            proj_params_left: [1.5, 2.5, 0.0, 0.0],
            proj_params_right: [1.5, 2.5, 0.1, -0.2],
            frame_index: 7,
            projection_flags_left: FRAME_PROJECTION_FLAG_ORTHOGRAPHIC,
            projection_flags_right: 0,
            sample_count: 4,
            ambient_sh_valid: true,
            skybox_specular: SkyboxSpecularUniformParams::from_cubemap_resident_mips(6),
            ambient_sh: [[0.0; 4]; 9],
        };
        let u = FrameGpuUniforms::new_clustered(&params);
        assert_eq!(u.camera_world_pos, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(u.camera_world_pos_right, [4.0, 5.0, 6.0, 0.0]);
        assert_eq!(u.view_space_z_coeffs, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(u.view_space_z_coeffs_right, [0.5, 0.6, 0.7, 0.8]);
        assert_eq!(u.cluster_count_x, 16);
        assert_eq!(u.cluster_count_y, 9);
        assert_eq!(u.cluster_count_z, 24);
        assert_eq!(u.near_clip, 0.05);
        assert_eq!(u.far_clip, 1000.0);
        assert_eq!(u.light_count, 42);
        assert_eq!(u.viewport_width, 1920);
        assert_eq!(u.viewport_height, 1080);
        assert_eq!(u.proj_params_left, [1.5, 2.5, 0.0, 0.0]);
        assert_eq!(u.proj_params_right, [1.5, 2.5, 0.1, -0.2]);
        assert_eq!(
            u.frame_tail,
            [
                7,
                FRAME_PROJECTION_FLAG_ORTHOGRAPHIC,
                0,
                pack_frame_tail_flags(true, 4)
            ]
        );
        assert_eq!(u.skybox_specular, [5.0, 1.0, 1.0, 0.0]);
        assert_eq!(u.ambient_sh, [[0.0; 4]; 9]);
    }

    #[test]
    fn new_clustered_can_pack_same_camera_position_for_mono() {
        let camera_world_pos = glam::Vec3::new(-1.0, 2.5, 8.0);
        let params = ClusteredFrameGlobalsParams {
            camera_world_pos,
            camera_world_pos_right: camera_world_pos,
            view_space_z_coeffs: [0.0, 0.0, 1.0, 0.0],
            view_space_z_coeffs_right: [0.0, 0.0, 1.0, 0.0],
            cluster_count_x: 1,
            cluster_count_y: 1,
            cluster_count_z: 1,
            near_clip: 0.01,
            far_clip: 100.0,
            light_count: 0,
            viewport_width: 1,
            viewport_height: 1,
            proj_params_left: [1.0, 1.0, 0.0, 0.0],
            proj_params_right: [1.0, 1.0, 0.0, 0.0],
            frame_index: 0,
            projection_flags_left: 0,
            projection_flags_right: 0,
            sample_count: 1,
            ambient_sh_valid: false,
            skybox_specular: SkyboxSpecularUniformParams::disabled(),
            ambient_sh: [[0.0; 4]; 9],
        };
        let u = FrameGpuUniforms::new_clustered(&params);

        assert_eq!(u.camera_world_pos, u.camera_world_pos_right);
        assert_eq!(u.frame_tail, [0, 0, 0, 0]);
    }

    #[test]
    fn frame_tail_sample_count_round_trips_supported_msaa_counts() {
        for sample_count in [1, 2, 4, 8] {
            let flags = pack_frame_tail_flags(false, sample_count);
            assert_eq!(frame_tail_sample_count(flags), sample_count);
            assert_eq!(flags & FRAME_TAIL_AMBIENT_SH_VALID, 0);
        }

        let flags = pack_frame_tail_flags(true, 4);
        assert_eq!(frame_tail_sample_count(flags), 4);
        assert_ne!(flags & FRAME_TAIL_AMBIENT_SH_VALID, 0);
    }

    #[test]
    fn frame_tail_sample_count_defaults_unsupported_counts_to_single_sample() {
        for sample_count in [0, 3, 16] {
            let flags = pack_frame_tail_flags(false, sample_count);
            assert_eq!(frame_tail_sample_count(flags), 1);
        }
    }

    #[test]
    fn skybox_specular_params_pack_disabled_and_cubemap() {
        assert_eq!(
            SkyboxSpecularUniformParams::disabled().to_vec4(),
            [0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            SkyboxSpecularUniformParams::from_cubemap_resident_mips(6).to_vec4(),
            [5.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(
            SkyboxSpecularUniformParams::from_cubemap_resident_mips(0).to_vec4(),
            [0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn render_sh2_packs_into_vec4_slots() {
        let sh = RenderSH2 {
            sh0: glam::Vec3::new(1.0, 2.0, 3.0),
            sh8: glam::Vec3::new(4.0, 5.0, 6.0),
            ..RenderSH2::default()
        };

        let packed = FrameGpuUniforms::ambient_sh_from_render_sh2(&sh);

        assert_eq!(packed[0], [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(packed[8], [4.0, 5.0, 6.0, 0.0]);
    }

    #[test]
    fn zero_render_sh2_packs_black_and_is_invalid() {
        let packed = FrameGpuUniforms::ambient_sh_from_render_sh2(&RenderSH2::default());

        assert_eq!(packed[0], [0.0; 4]);
        assert_eq!(packed[1], [0.0; 4]);
        assert!(!FrameGpuUniforms::ambient_sh_is_valid(&RenderSH2::default()));
    }

    #[test]
    fn nonzero_render_sh2_is_valid() {
        let sh = RenderSH2 {
            sh0: glam::Vec3::new(0.01, 0.0, 0.0),
            ..RenderSH2::default()
        };

        assert!(FrameGpuUniforms::ambient_sh_is_valid(&sh));
    }
}
