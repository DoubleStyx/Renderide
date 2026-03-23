Shader "Custom/PBSIntersect"
{
    Properties
    {
		_Color("Color", Color) = (1,1,1,1)
		_MainTex("Albedo (RGB)", 2D) = "white" {}

		_IntersectColor ("Intersect Color", Color) = (1,1,1,1)
		_IntersectEmissionColor ("Intersect Emission Color", Color) = (1, 0, 0, 1)

		_BeginTransitionStart ("Begin Transition Start", Float) = 0
		_BeginTransitionEnd ("Begin Transition End", Float) = 0
		_EndTransitionStart ("End Transition Start", Float) = 0.1
		_EndTransitionEnd ("End Transition End", Float) = 0.1

		_NormalMap("NormalMap", 2D) = "bump" {}
		_NormalScale("Normal Scale", Float) = 1

		_EmissionColor("EmissionColor", Color) = (0,0,0,0)
		_EmissionMap("EmissionMap", 2D) = "black" {}

		_OcclusionMap("Occlusion", 2D) = "white" {}

		_Glossiness("Smoothness", Range(0,1)) = 0.5
		_Metallic("Metallic", Range(0,1)) = 0.0

		_MetallicMap("MetallicMap", 2D) = "black" {}

		_OffsetFactor("Offset Factor", Float) = 0.0
		_OffsetUnits("Offset Units", Float) = 0.0
    }



    SubShader
    {
		Tags { "RenderType" = "Transparent" "Queue" = "Transparent" }
		Cull[_Cull]
		Offset[_OffsetFactor],[_OffsetUnits]
		LOD 200

        CGPROGRAM
        // Physically based Standard lighting model, and enable shadows on all light types
        #pragma surface surf Standard alpha fullforwardshadows vertex:vert
		#define _GLOSSYENV 1

        #pragma target 4.0
		

		#pragma multi_compile _ _ALBEDOTEX
		#pragma multi_compile _ _EMISSIONTEX
		#pragma multi_compile _ _NORMALMAP
		#pragma multi_compile _ _METALLICMAP
		#pragma multi_compile _ _OCCLUSION

		UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);

		float _BeginTransitionStart;
		float _BeginTransitionEnd;
		float _EndTransitionStart;
		float _EndTransitionEnd;

		float4 _IntersectColor;
		float4 _IntersectEmissionColor;

#ifdef _ALBEDOTEX
		sampler2D _MainTex;
#endif

#ifdef _NORMALMAP
		sampler2D _NormalMap;
		float _NormalScale;
#endif

#ifdef _EMISSIONTEX
		sampler2D _EmissionMap;
#endif

#ifdef _METALLICMAP
		sampler2D _MetallicMap;
#else
		half _Glossiness;
		half _Metallic;
#endif

#ifdef _OCCLUSION
		sampler2D _OcclusionMap;
#endif

		fixed4 _Color;
		float4 _EmissionColor;

        struct Input
        {
            float2 uv_MainTex;
			float facing : FACE;
			float4 screenPos;
			float eyeDepth : TEXCOORD1;
        };

        // Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
        // See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
        // #pragma instancing_options assumeuniformscaling
        UNITY_INSTANCING_BUFFER_START(Props)
            // put more per-instance properties here
        UNITY_INSTANCING_BUFFER_END(Props)

		void vert(inout appdata_full v, out Input o)
		{
			UNITY_INITIALIZE_OUTPUT(Input, o);
			COMPUTE_EYEDEPTH(o.eyeDepth);
		}

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
			float rawZ = SAMPLE_DEPTH_TEXTURE_PROJ(_CameraDepthTexture, UNITY_PROJ_COORD(IN.screenPos));
			float sceneZ = LinearEyeDepth(rawZ);
			float partZ = IN.eyeDepth;
			float diff = sceneZ - partZ;

			float intersectLerp;

			if (diff < _EndTransitionStart)
				intersectLerp = (diff - _BeginTransitionStart) / (_BeginTransitionEnd - _BeginTransitionStart);
			else
				intersectLerp = (_EndTransitionEnd - diff) / (_EndTransitionEnd - _EndTransitionStart);

			intersectLerp = saturate(intersectLerp);

			float2 mainUv = IN.uv_MainTex;

			half4 c = lerp(_Color, _IntersectColor, intersectLerp);

#ifdef _ALBEDOTEX
			c *= tex2D(_MainTex, mainUv);
#endif

			o.Albedo = c.rgb;

#ifdef _NORMALMAP
			o.Normal = UnpackScaleNormal(tex2D(_NormalMap, mainUv), _NormalScale);
#endif
			if (IN.facing < 0.5)
				o.Normal.z *= -1;

#ifdef _OCCLUSION
			o.Occlusion = tex2D(_OcclusionMap, mainUv).r;
#endif

#ifdef _METALLICMAP
			half4 m = tex2D(_MetallicMap, mainUv);

			o.Metallic = m.r;
			o.Smoothness = m.a;
#else
			o.Metallic = _Metallic;
			o.Smoothness = _Glossiness;
#endif

			o.Emission = _EmissionColor;
#ifdef _EMISSIONTEX
			o.Emission *= tex2D(_EmissionMap, mainUv).rgb;
#endif
			o.Emission += _IntersectEmissionColor * intersectLerp;

			o.Alpha = c.a;
        }
        ENDCG
    }
    FallBack "Diffuse"
}
