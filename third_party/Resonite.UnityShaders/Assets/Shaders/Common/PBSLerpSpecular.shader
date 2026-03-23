Shader "PBSLerpSpecular" {
	Properties{
		_Color("Color0", Color) = (1,1,1,1)
		_Color1("Color0", Color) = (1,1,1,1)

		_Lerp("Lerp", Float) = 0
		_LerpTex("LerpTexture", 2D) = "white" {}

		_MainTex("Albedo (RGB) 0", 2D) = "white" {}
		_MainTex1("Albedo (RGB) 1", 2D) = "white" {}

		_NormalMap("NormalMap 0", 2D) = "bump" {}
		_NormalMap1("NormalMap 0", 2D) = "bump" {}
		_NormalScale("Normal Scale", Float) = 1
		_NormalScale1("Normal Scale", Float) = 1

		_EmissionColor("EmissionColor0", Color) = (0,0,0,0)
		_EmissionColor1("EmissionColor1", Color) = (0,0,0,0)

		_EmissionMap("EmissionMap0", 2D) = "black" {}
		_EmissionMap1("EmissionMap1", 2D) = "black" {}

		_Occlusion("Occlusion0", 2D) = "white" {}
		_Occlusion1("Occlusion1", 2D) = "white" {}

		// Specular
		_SpecularColor("SpecularColor 0", Color) = (1,1,1,0.5)
		_SpecularColor1("SpecularColor 1", Color) = (1,1,1,0.5)

		_SpecularMap("SpecularMap 0", 2D) = "white" {}
		_SpecularMap1("SpecularMap 1", 2D) = "white" {}

		_AlphaClip("Alpha clip", Float) = 0.5
			_Cull("_Cull", Float) = 2.0
	}



		SubShader{
			Tags { "RenderType" = "Opaque" }
			Offset[_OffsetFactor],[_OffsetUnits]
			Cull[_Cull]
			LOD 200

			CGPROGRAM
			#pragma surface surf StandardSpecular fullforwardshadows addshadow
			#define _GLOSSYENV 1

			#pragma multi_compile _ _LERPTEX

			#pragma multi_compile _ _ALBEDOTEX
			#pragma multi_compile _ _EMISSIONTEX
			#pragma multi_compile _ _NORMALMAP
			#pragma multi_compile _ _OCCLUSION
			#pragma multi_compile _ _SPECULARMAP
#pragma multi_compile _ _MULTI_VALUES
		#pragma multi_compile _ _DUALSIDED
		#pragma multi_compile _ _ALPHACLIP

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0
		

#ifdef _LERPTEX
		sampler2D _LerpTex;
#endif
		half _Lerp;

#ifdef _ALBEDOTEX
		sampler2D _MainTex;
		sampler2D _MainTex1;
#endif

#ifdef _NORMALMAP
		sampler2D _NormalMap;
		sampler2D _NormalMap1;
		float _NormalScale;
		float _NormalScale1;
#endif

#ifdef _OCCLUSION
		sampler2D _Occlusion;
		sampler2D _Occlusion1;
#endif

#ifdef _EMISSIONTEX
		sampler2D _EmissionMap;
		sampler2D _EmissionMap1;
#endif

#ifdef _SPECULARMAP
		sampler2D _SpecularMap;
		sampler2D _SpecularMap1;
#endif

#ifdef _ALPHACLIP
		float _AlphaClip;
#endif

		half4 _SpecularColor;
		half4 _SpecularColor1;
		fixed3 _EmissionColor;
		fixed3 _EmissionColor1;

		fixed4 _Color;
		fixed4 _Color1;



		struct Input {
			float2 uv_LerpTex;
			float2 uv_MainTex;
			float2 uv_MainTex1;

#ifdef _DUALSIDED
			float facing : FACE;
#endif 
		};

		void surf(Input IN, inout SurfaceOutputStandardSpecular o)
		{
			// Compute lerp factor first
#ifdef _LERPTEX
			half l = tex2D(_LerpTex, IN.uv_LerpTex).r;
#ifdef _MULTI_VALUES
			l *= _Lerp;
#endif
#else
			half l = _Lerp;
#endif

			// Albedo

			fixed4 c0 = _Color;
			fixed4 c1 = _Color1;

#ifdef _ALBEDOTEX
			c0 *= tex2D(_MainTex, IN.uv_MainTex);
			c1 *= tex2D(_MainTex1, IN.uv_MainTex1);
#endif

			fixed4 c = lerp(c0, c1, l);

#ifdef _ALPHACLIP
			clip(c.a - _AlphaClip);
#endif

			o.Albedo = c.rgb;
			o.Alpha = c.a;

			// Normal

#ifdef _NORMALMAP
			fixed3 n0 = UnpackScaleNormal(tex2D(_NormalMap, IN.uv_MainTex), _NormalScale);
			fixed3 n1 = UnpackScaleNormal(tex2D(_NormalMap1, IN.uv_MainTex1), _NormalScale1);

			o.Normal = lerp(n0, n1, l);
#endif

#ifdef _DUALSIDED
			if (IN.facing < 0.5)
				o.Normal.z *= -1;
#endif

			// Occlusion (duh)
#ifdef _OCCLUSION
			o.Occlusion = lerp(tex2D(_Occlusion, IN.uv_MainTex).r, tex2D(_Occlusion1, IN.uv_MainTex1).r, l);
#endif

			// Emission

			half3 e0 = _EmissionColor;
			half3 e1 = _EmissionColor1;

#ifdef _EMISSIONTEX
			e0 *= tex2D(_EmissionMap, IN.uv_MainTex).rgb;
			e1 *= tex2D(_EmissionMap1, IN.uv_MainTex1).rgb;
#endif

			o.Emission = lerp(e0, e1, l);

			// Specular and smoothness
#ifdef _SPECULARMAP
			half4 s0 = tex2D(_SpecularMap, IN.uv_MainTex);
			half4 s1 = tex2D(_SpecularMap1, IN.uv_MainTex1);

#ifdef _MULTI_VALUES
			s0 *= _SpecularColor;
			s1 *= _SpecularColor1;
#endif

#else
			half4 s0 = _SpecularColor;
			half4 s1 = _SpecularColor1;
#endif
			half4 s = lerp(s0, s1, l);

			o.Specular = s.rgb;
			o.Smoothness = s.a;
		}
		ENDCG
	}
	FallBack "Transparent/Cutout/VertexLit"
}
