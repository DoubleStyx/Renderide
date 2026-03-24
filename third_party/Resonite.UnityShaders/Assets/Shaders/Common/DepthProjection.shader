Shader "DepthProjection"
{
	Properties
	{
		_MainTex("Texture", 2D) = "white" {}
		_DepthTex("DepthTexture", 2D) = "black" {}

		_MainTex_ST ("", Vector) = ( 1, 1, 0, 0 )
		_DepthTex_ST("", Vector) = (1, 1, 0, 0)

		_DepthFrom("DepthFrom", Float) = 0
		_DepthTo("DepthTo", Float) = 1
		_Angle("AngleScale", Vector) = (90, 60, 0, 0)

		_NearClip("NearClip", Float) = 0
		_FarClip("FarClip", Float) = 1

		_DiscardThreshold("Discard Threshold", Float) = 0.01
		_DiscardOffset("Discard Offset", Float) = 0.01

		_SrcBlend("SrcBlend", Float) = 1.0
		_DstBlend("DstBlend", Float) = 0.0
		_ZWrite("ZWrite", Float) = 1.0
	}
		SubShader
	{
		Tags{ "Queue" = "AlphaTest+200" "RenderType" = "Opaque"  }

		LOD 100
		Offset[_OffsetFactor],[_OffsetUnits]

		Cull Off
		Blend[_SrcBlend][_DstBlend], One One
		BlendOp Add, Max
		ZWrite[_ZWrite]

		Pass
	{
		CGPROGRAM
#pragma vertex vert
#pragma fragment frag

#pragma exclude_renderers d3d11_9x

#pragma multi_compile DEPTH_GRAYSCALE DEPTH_HUE

#include "UnityCG.cginc"

		struct appdata
	{
		float4 vertex : POSITION;
		float2 uv : TEXCOORD0;

		UNITY_VERTEX_INPUT_INSTANCE_ID
	};

	struct v2f
	{
		float2 uv : TEXCOORD0;
		float4 vertex : SV_POSITION;

		float normDepth : TEXCOORD1;
		float diff : TEXCOORD2;

		UNITY_VERTEX_OUTPUT_STEREO
	};

	sampler2D _MainTex;
	float4 _MainTex_ST;

	sampler2D _DepthTex;
	float4 _DepthTex_ST;

	float _DepthFrom;
	float _DepthTo;
	float4 _Angle;

	float _NearClip;
	float _FarClip;

	float _DiscardThreshold;
	float _DiscardOffset;

	float sampleDepth(float2 uv)
	{
#ifdef DEPTH_GRAYSCALE
		return 1 - tex2Dlod(_DepthTex, float4(TRANSFORM_TEX(uv, _DepthTex), 0, 0)).x;
#elif defined(DEPTH_HUE)
		float3 c = tex2Dlod(_DepthTex, float4(TRANSFORM_TEX(uv, _DepthTex), 0, 0)).xyz;

		// convert from rgb to normalized hue, assumed full saturation
		float cmax = max(max(c.r, c.g), c.b);
		float cmin = min(min(c.r, c.g), c.b);
		float delta = cmax - cmin;

		if (delta == 0)
			return 1;
		if (cmax == c.r)
			return (((c.g - c.b) / delta) % 6) / 6.0;
		if (cmax == c.g)
			return ((c.b - c.r) / delta + 2) / 6.0;
		return ((c.r - c.g) / delta + 4) / 6.0;
#else
		return 0.0;
#endif
	}

	v2f vert(appdata v)
	{
		v2f o;

		UNITY_SETUP_INSTANCE_ID(v);
		UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

		float depth = sampleDepth(v.uv);

		float4 surroundDepth = float4(
			sampleDepth(v.uv + float2(_DiscardOffset, 0)),
			sampleDepth(v.uv + float2(0, _DiscardOffset)),
			sampleDepth(v.uv - float2(_DiscardOffset, 0)),
			sampleDepth(v.uv - float2(0, _DiscardOffset))
			);

		float diff = abs(depth - surroundDepth.x);
		diff = max(diff, abs(depth - surroundDepth.y));
		diff = max(diff, abs(depth - surroundDepth.z));
		diff = max(diff, abs(depth - surroundDepth.w));

		depth = depth + surroundDepth.x + surroundDepth.y + surroundDepth.z + surroundDepth.w;
		depth /= 5;

		o.normDepth = depth;

		//depth *= _DepthFrom;

		depth = log(depth + 1) / log(_DepthTo + 1) * depth;
		depth = (_DepthTo - _DepthFrom) * depth;
		depth += _DepthFrom;

		/*float2 angle = atan((v.uv - 0.5) * tan((_Angle.xy / 180)*3.1415));
		float3 normal = float3(tan(angle), 1);*/

		float2 angle = (v.uv - 0.5) * tan((_Angle.xy / 180)*3.1415);
		float3 normal = float3(angle, 1);

		/*
		// This was dumb
		float3 normal = float3(0, 0, 1);

		float3x3 rot = float3x3(
			cos(angle.x), 0, sin(angle.x),
			0, 1, 0,
			-sin(angle.x), 0, cos(angle.x)
			);

		normal = mul(rot, normal);

		rot = float3x3(
			1, 0, 0,
			0, cos(angle.y), -sin(angle.y),
			0, sin(angle.y), cos(angle.y)
			);

		normal = mul(rot, normal);*/

		float3 offset = normal * depth;

		o.vertex = UnityObjectToClipPos(float4(offset, 1)/*v.vertex*/);
		o.uv = v.uv;
		o.diff = diff;

		return o;
	}

	fixed4 frag(v2f i) : SV_Target
	{
		UNITY_SETUP_INSTANCE_ID(i);
		UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

		// sample the texture
		fixed4 col = tex2D(_MainTex, TRANSFORM_TEX(i.uv, _MainTex));

	if (i.diff > _DiscardThreshold || i.normDepth < _NearClip || i.normDepth > _FarClip)
		discard;

	return col;
	}
		ENDCG
	}
	}
}
