Shader "Unlit/UnlitPolarMapping"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
		_Pow ("Pow", Float) = 1.0
	}
	SubShader
	{
		Tags { "RenderType"="Opaque" }
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#pragma target 3.0
			// make fog work
			#pragma multi_compile_fog
			
			#include "UnityCG.cginc"
			#include "..\Common.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				UNITY_FOG_COORDS(1)
				float4 vertex : SV_POSITION;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;

			uniform float _Pow;
			
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv;
				UNITY_TRANSFER_FOG(o,o.vertex);
				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				// shift UV coordinates so they cover -1 to 1 range
				i.uv = (i.uv * 2) - 1;

				float2 uvddx, uvddy;
				i.uv = PolarMapping(i.uv, _MainTex_ST, _Pow, uvddx, uvddy);

				fixed4 col = tex2Dgrad(_MainTex, i.uv, uvddx, uvddy);
				// apply fog
				UNITY_APPLY_FOG(i.fogCoord, col);
				return col;
			}
			ENDCG
		}
	}
}
