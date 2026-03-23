Shader "Null"
{
	Properties
	{
	}
	SubShader
	{
		Tags { "RenderType"="Opaque" }
		Offset 2,2
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#pragma multi_compile_instancing
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				//float2 uv : TEXCOORD0;

				//UNITY_VERTEX_INPUT_INSTANCE_ID
			};

			struct v2f
			{
				/*float2 uv : TEXCOORD0;
				UNITY_FOG_COORDS(1)*/
				float4 vertex : SV_POSITION;
				float3 checker : TEXCOORD0;

				UNITY_VERTEX_OUTPUT_STEREO
			};
			
			v2f vert (appdata v)
			{
				v2f o;

				//UNITY_SETUP_INSTANCE_ID(v);
				UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

				o.checker = v.vertex.xyz * 5;
				o.vertex = UnityObjectToClipPos(v.vertex);
				return o;
			}

			#define TRANSITION 50

			void transition(float p, inout float a, inout float b)
			{
				p *= TRANSITION;

				if(p < TRANSITION * 0.25)
					p = saturate(p + 0.5);
				else if(p < TRANSITION * 0.75)
					p = 1 - saturate(p - TRANSITION * 0.5 - 0.5);
				else
					p = 1 - saturate(TRANSITION - p + 0.5);

				float _a = lerp(a, b, p);
				float _b = lerp(b, a, p);

				a = _a;
				b = _b;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				//UNITY_SETUP_INSTANCE_ID(i);
				UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

				i.checker = frac(i.checker);

				float a = 0;
				float b = 0.05;

				transition(i.checker.x, a, b);
				transition(i.checker.y, a, b);
				transition(i.checker.z, a, b);

				float intensity = a;

				return fixed4(intensity.xxx, 1);
			}
			ENDCG
		}
	}
}
