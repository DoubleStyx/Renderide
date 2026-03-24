Shader "Unlit/Circle"
{
	Properties
	{
		_Color("Color", Color) = (1,1,1,1)
	}
		SubShader
	{
		Tags{ "Queue" = "Transparent" "IgnoreProjector" = "True" "RenderType" = "Transparent" }
		Blend SrcAlpha OneMinusSrcAlpha
		ColorMask RGB
		ZWrite Off
		Cull Off

		Pass
	{
		CGPROGRAM

#pragma vertex vert
#pragma fragment frag

#include "UnityCG.cginc"

		// Quality level
		// 2 == high quality
		// 1 == medium quality
		// 0 == low quality
#define QUALITY_LEVEL 2

		struct appdata
	{
		float4 vertex : POSITION;
		float2 texcoord : TEXCOORD0;

		UNITY_VERTEX_INPUT_INSTANCE_ID
	};

	struct v2f
	{
		float4 pos : SV_POSITION;
		float2 uv : TEXCOORD0;

		UNITY_VERTEX_OUTPUT_STEREO
	};

	v2f vert(appdata v)
	{
		v2f o;

		UNITY_SETUP_INSTANCE_ID(v);
		UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

		o.pos = UnityObjectToClipPos(v.vertex);
		o.uv = v.texcoord.xy;
		return o;
	}

	fixed4 _Color;

	fixed4 frag(v2f i) : SV_Target
	{
		//UNITY_SETUP_INSTANCE_ID(i);
		UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

		float2 coord = i.uv;
		float2 center = float2(0.5, 0.5);

		float dst = dot(abs(coord - center), float2(1, 1));
		float aaf = fwidth(dst);

		dst = 1 - smoothstep(0.2 - aaf, 0.2, dst);
	
		return fixed4(1, 1, 1, dst);

	}
		ENDCG
	}
	}
}