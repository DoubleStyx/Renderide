Shader "PlaneTransition/WireframeUnlit"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
		_LineColor ("Line Color", Color) = (0, 1, 1, 1)
		_FillColor ("Fill Color", Color) = (0, 0.2, 0.2, 1)

		_Thickness ("Thickness", Float) = 2

		_PlaneDir("Plane Direction", Vector) = (0, 1, 0, 0)

		_FillTransitionRange ("Fill Transition Range", Float) = 0.1
		_FillTransitionExp("Fill Transition Exp", Float) = 1
		_FillPlaneOffset("Fill Plane Offset", Float) = 0

		_WireTransitionRange("Wire Transition Range", Float) = 0.1
		_WireTransitionExp("Wire Transition Exp", Float) = 1
		_WirePlaneOffset("Wire Plane Offset", Float) = 0
	}
	SubShader
	{
		Pass
		{
			ZWrite On
			ColorMask 0
		}

		Pass
		{
			Tags{ "RenderType" = "Transparent" "Queue" = "Transparent" }
			LOD 200
			Blend SrcAlpha OneMinusSrcAlpha

			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
				float distance : TEXCOORD1;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;

			float3 _PlaneDir;

			float _FillPlaneOffset;
			float _FillTransitionRange;
			float _FillTransitionExp;

			float3 project(float3 a, float3 b)
			{
				return b * dot(a,b);
			}
			
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _MainTex);
				o.distance = dot(normalize(_PlaneDir), v.vertex);

				// only uses the fill, so can transform here
				o.distance += _FillPlaneOffset;
				o.distance /= _FillTransitionRange * 0.5;
				o.distance = (o.distance + 1) * 0.5f; 

				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				// sample the texture
				fixed4 col = tex2D(_MainTex, i.uv);

				float lerp = pow(saturate(i.distance), _FillTransitionExp);

				if (lerp == 0)
					discard;

				col.a *= lerp;
				
				return col;
			}
			ENDCG
		}

		Pass
		{
			Tags{ "RenderType" = "Transparent" "Queue" = "Transparent" }
			LOD 200
			Blend One One

			CGPROGRAM
			#pragma target 5.0
			#include "UnityCG.cginc"
			#include "WireframeShared.cginc"
			#pragma vertex vert
			#pragma fragment frag
			#pragma geometry geom

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2g
			{
				float4 pos : SV_POSITION;
				float2 uv : TEXCOORD0;
				float distance : TEXCOORD1;

				float4 posWorld : TEXCOORD2;
			};

			struct g2f
			{
				float4 pos : SV_POSITION;
				float2 uv : TEXCOORD0;
				float distance : TEXCOORD1;

				float4 posWorld : TEXCOORD2;
				float3 dist : TEXCOORD3;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;

			float3 _PlaneDir;
			
			float _FillPlaneOffset;
			float _FillTransitionRange;
			float _FillTransitionExp;

			float _WirePlaneOffset;
			float _WireTransitionRange;
			float _WireTransitionExp;

			float4 _FillColor;
			float4 _LineColor;

			float3 project(float3 a, float3 b)
			{
				return b * dot(a,b);
			}

			v2g vert(appdata v)
			{
				v2g o;
				o.pos = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _MainTex);

				o.distance = dot(normalize(_PlaneDir), v.vertex);

				o.posWorld = mul(unity_ObjectToWorld, v.vertex);

				return o;
			}

			// Geometry Shader
			[maxvertexcount(3)]
			void geom(triangle v2g p[3], inout TriangleStream<g2f> triStream)
			{
				float3 dist;
#ifdef _SCREENSPACE
				dist = geom_screenspace(p[0].pos, p[1].pos, p[2].pos);
#else
				dist = geom_objectspace(p[0].posWorld, p[1].posWorld, p[2].posWorld);
#endif

				g2f pIn;

				//add the first point
				pIn.pos = p[0].pos;
				pIn.uv = p[0].uv;
				pIn.dist = float3(dist.x, 0, 0);
				pIn.distance = p[0].distance;
				pIn.posWorld = p[0].posWorld;
				//pIn.normal = p[0].normal;
				triStream.Append(pIn);

				//add the second point
				pIn.pos = p[1].pos;
				pIn.uv = p[1].uv;
				pIn.dist = float3(0, dist.y, 0);
				pIn.distance = p[1].distance;
				pIn.posWorld = p[1].posWorld;
				//pIn.normal = p[1].normal;
				triStream.Append(pIn);

				//add the third point
				pIn.pos = p[2].pos;
				pIn.uv = p[2].uv;
				pIn.dist = float3(0, 0, dist.z);
				pIn.distance = p[2].distance;
				pIn.posWorld = p[2].posWorld;
				//pIn.normal = p[2].normal;
				triStream.Append(pIn);
			}

			fixed4 frag(g2f i) : SV_Target
			{
				float l = line_lerp(i.dist);

				float fillLerp = i.distance;

				fillLerp += _FillPlaneOffset;
				fillLerp /= _FillTransitionRange * 0.5;
				fillLerp = saturate(1 - (fillLerp + 1) * 0.5);
				fillLerp = pow(fillLerp, _FillTransitionExp);

				float wireLerp = i.distance;

				wireLerp += _WirePlaneOffset;
				wireLerp /= _WireTransitionRange * 0.5;
				wireLerp = saturate(1 - (wireLerp + 1) * 0.5);
				wireLerp = pow(wireLerp, _WireTransitionExp);

				float4 col = float4(0, 0, 0, 0);

				col += _FillColor * fillLerp;
				col += _LineColor * wireLerp * l;

				return col;
			}
			ENDCG
		}
	}
}
