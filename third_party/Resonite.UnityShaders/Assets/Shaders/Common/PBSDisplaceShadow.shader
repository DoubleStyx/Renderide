// Upgrade NOTE: upgraded instancing buffer 'Props' to new syntax.

Shader "PBSDisplaceShadow"
{
	Properties{
		_Color("Color", Color) = (1,1,1,1)
		_MainTex("Albedo (RGB)", 2D) = "white" {}

		_VertexOffsetMap ("Vertex Offset Map", 2D) = "black" {}
		_VertexOffsetMagnitude ("Vertex Offset Magnitude", Float) = 0.1
		_VertexOffsetBias ("Vertex Offset Bias", Float) = 0
	}
			SubShader{
				Tags{ "RenderType" = "Opaque" }
				//Tags { "RenderType"="Transparent" "Queue"="Transparent" }
				LOD 200

				CGPROGRAM
			// Physically based Standard lighting model, and enable shadows on all light types
			#pragma surface surf Standard fullforwardshadows vertex:vert addshadow

			// Use shader model 3.0 target, to get nicer looking lighting
			#pragma target 3.0
			

		sampler2D _VertexOffsetMap;
		float4 _VertexOffsetMap_ST;
		float _VertexOffsetMagnitude;
		float _VertexOffsetBias;

		sampler2D _MainTex;

		struct Input
		{
			float2 uv_MainTex;
		};

		void vert(inout appdata_full v)
		{
			float vertOffset = tex2Dlod(_VertexOffsetMap, float4(TRANSFORM_TEX(v.texcoord, _VertexOffsetMap), 0, 0)).x;
			vertOffset = vertOffset * _VertexOffsetMagnitude + _VertexOffsetBias;

			v.vertex.xyz += v.normal.xyz * vertOffset;
		}

		void surf (Input IN, inout SurfaceOutputStandard o)
		{
		}
		ENDCG
	}
	FallBack "Diffuse"
}
