Shader "Billboard/Unlit" {
	Properties{
		_Tex("Texture", 2D) = "white" {}
	_Color("Color", Color) = (1,1,1,1)
		_Cutoff("Alpha cutoff", Range(0,1)) = 0.5

		_PointSize ("PointSize", Vector) = (0.1, 0.1, 0, 0)

		_SrcBlend("SrcBlend", Float) = 1.0
		_DstBlend("DstBlend", Float) = 0.0
		_ZWrite("ZWrite", Float) = 1.0
		_Cull("Cull", Float) = 2.0

		_OffsetTex("Offset Texture", 2D) = "black" {}
		_OffsetMagnitude("Offset Magnitude", Vector) = (0.1, 0.1, 0, 0)

		_PolarPow("Polar Mapping Power", Float) = 1.0

		_ZTest("ZTest", Float) = 2
	}

		

		SubShader{
		Tags{ "Queue" = "AlphaTest+200" "RenderType" = "Transparent" "DisableBatching" = "True" }
		LOD 100

		Pass{

		Blend[_SrcBlend][_DstBlend], One One
		BlendOp Add, Max
		ZWrite[_ZWrite]
		Cull[_Cull]
		ZTest[_ZTest]

		CGPROGRAM


#pragma vertex vert
#pragma fragment frag
#pragma geometry geom
#pragma multi_compile_fog
//#pragma multi_compile_fwdbase_fullshadows

#pragma multi_compile _ _TEXTURE
#pragma multi_compile _ _COLOR
#pragma multi_compile _ _ALPHATEST
#pragma multi_compile _ _VERTEXCOLORS
#pragma multi_compile _ _MUL_ALPHA_INTENSITY
#pragma multi_compile _ _OFFSET_TEXTURE
#pragma multi_compile _ _MUL_RGB_BY_ALPHA
#pragma multi_compile _ _POLARUV
#pragma multi_compile _ _RIGHT_EYE_ST

#pragma multi_compile _ _POINT_ROTATION
#pragma multi_compile _ _POINT_SIZE
#pragma multi_compile _ _POINT_UV

#pragma multi_compile _ _VERTEX_LINEAR_COLOR
#pragma multi_compile _ _VERTEX_SRGB_COLOR
#pragma multi_compile _ _VERTEX_HDRSRGB_COLOR
#pragma multi_compile _ _VERTEX_HDRSRGBALPHA_COLOR

#include "UnityCG.cginc"
#include "..\Common.cginc"

		struct v2g
	{
		float4 pos : POSITION;

#if defined(_TEXTURE) && defined(_POINT_UV)
		float2 texcoord : TEXCOORD0;
		float2 texscale : TEXCOORD1;
#endif

#if defined(_POINT_ROTATION) || defined(_POINT_SIZE)
		float3 pointdata : NORMAL;
#endif

#ifdef _VERTEXCOLORS
		float4 vcolor : COLOR;
#endif

		UNITY_FOG_COORDS(2)
		UNITY_VERTEX_INPUT_INSTANCE_ID
	};

	struct g2f
	{
		float4 pos : SV_POSITION;
		float2 uv : TEXCOORD0;

#ifdef _VERTEXCOLORS
		float4 vcolor : COLOR;
#endif

		UNITY_VERTEX_OUTPUT_STEREO
	};

	float2 _PointSize;

#ifdef _TEXTURE
	sampler2D _Tex;
	float4 _Tex_ST;
#ifdef _RIGHT_EYE_ST
	float4 _RightEye_ST;
#endif
#endif

#ifdef _OFFSET_TEXTURE
	float2 _OffsetMagnitude;

	sampler2D _OffsetTex;
	float4 _OffsetTex_ST;
#endif

#ifdef _POLARUV
	float _PolarPow;
#endif

	#include "..\Billboard.cginc"

		float2 billboard_size(v2g p)
	{
		float2 scale = _PointSize;
#ifdef _POINT_SIZE
		scale *= p.pointdata.xy;
#endif
		return scale;
	}

	void rotate_billboard(v2g p, inout float3 m0, inout float3 m1, inout float3 m2)
	{
#ifdef _POINT_ROTATION
		rotate_by_angle(p.pointdata.z, m0, m1, m2);
#endif
	}

	void setup_vertex(v2g p, inout g2f v)
	{
#if defined(_TEXTURE) && defined(_POINT_UV)
		v.uv = p.texcoord + p.texscale * (v.uv - 0.5);
#endif
		#if defined(_VERTEXCOLORS)
		v.vcolor = p.vcolor;
		#endif
	}

	BILLBOARD_GEOMETRY_SHADER

	v2g vert(v2g v)
	{
		UNITY_TRANSFER_FOG(v,v.pos);
		return v;
	}

	float4 frag(g2f i) : SV_Target
	{
		//UNITY_SETUP_INSTANCE_ID(i);
		UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

#if defined(_TEXTURE)

		float4 _ST;
#if defined(_RIGHT_EYE_ST)
	if (unity_StereoEyeIndex == 0)
		_ST = _Tex_ST;
	else
		_ST = _RightEye_ST;
#else
	_ST = _Tex_ST;
#endif

#if defined(_POLARUV)
	float2 uvddx, uvddy;
	float2 uv = PolarMapping(i.uv * 2 - 1, _ST, _PolarPow, uvddx, uvddy);
#else
	float2 uv = i.uv * _ST.xy + _ST.zw;
#endif

	// TODO!!! Polar mapping for Offset Texture as well?
	// Polar offets?
#if defined(_OFFSET_TEXTURE)
	float4 offset = tex2D(_OffsetTex, TRANSFORM_TEX(i.uv, _OffsetTex));
	uv += offset.xy * _OffsetMagnitude;
#endif

#if defined(_POLARUV)
	float4 col = tex2Dgrad(_Tex, uv, uvddx, uvddy);
#else
	float4 col = tex2D(_Tex, uv);
#endif

#if defined(_COLOR)
	col *= _Color;
#endif

#elif defined(_COLOR)
		float4 col = _Color;
#else
		float4 col = float4(1,1,1,1);
#endif

#if defined(_ALPHATEST)
	clip(col.a - _Cutoff);
#endif

	EVR_APPLY_VERTEX_COLORS_FRAG(col, i);

#ifdef _MUL_RGB_BY_ALPHA
	col.rgb *= col.a;
#endif

	// This is for external camera blending, so additive shaders output alpha
#ifdef _MUL_ALPHA_INTENSITY
	float mulfactor = (col.r + col.g + col.b) * 0.3333333;
	col.a *= mulfactor;
#endif

	UNITY_APPLY_FOG(i.fogCoord, col);
	//UNITY_OPAQUE_ALPHA(col.a);
	return col;
	}
		ENDCG
	}
	}

}
