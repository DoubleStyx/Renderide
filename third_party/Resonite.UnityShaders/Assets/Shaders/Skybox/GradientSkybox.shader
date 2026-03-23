// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "GradientSkybox"
{
	Properties
	{
		_BaseColor ("Base Color", Color) = (1,1,1,1)
		_Gradients ("Gradient Count", Float) = 0
	}
	SubShader
	{
		Tags{ "Queue" = "Background" "RenderType" = "Background" "PreviewType" = "Skybox" }
		Cull Off ZWrite Off

		Pass
		{
			CGPROGRAM
			
			#pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"
			#include "Lighting.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
			};

			struct v2f
			{
				float4	pos	: SV_POSITION;
				float3	vertex : TEXCOORD0;
			};

			uniform half4 _BaseColor;

			uniform int _Gradients;

			uniform half4 _DirsSpread[16]; // xyz - dir, w - spread;
			uniform half4 _Color0[16];
			uniform half4 _Color1[16];
			uniform half4 _Params[16]; // x - blendmode, y - exp, z - from, w - to
			
			v2f vert (appdata v)
			{
				v2f OUT;
				OUT.pos = UnityObjectToClipPos(v.vertex);
				OUT.vertex = v.vertex;

				return OUT;
			}
			
			half4 frag (v2f IN) : SV_Target
			{
				half3 col = _BaseColor;
				float3 ray = normalize(mul((float3x3)unity_ObjectToWorld, IN.vertex));

#if !defined(SHADER_API_D3D11_9X)
				for (int i = 0; i < _Gradients; i++)
				{
					float r = 0.5 - dot(ray, _DirsSpread[i].xyz)*0.5;

					half spread = _DirsSpread[i].w;
					half exp = _Params[i].y;
					half from = _Params[i].z;
					half to = _Params[i].w;

					// rescale ratio to the target spread
					r /= spread;

					if (r > 1)
						continue;

					r = pow(r, _Params[i].y);

					// rescale to subrange
					r -= from;
					r /= (to - from);
					r = clamp(r, 0, 1);

					half4 c = lerp(_Color0[i], _Color1[i], r);

					if (_Params[i].x == 0) // alpha
						col = col.rgb * (1 - c.a) + c.rgb * c.a;
					else
						col = col.rgb + c.rgb * c.a; // Additive
				}
#endif

				return half4(col, 1);
			}
			ENDCG
		}
	}
}
