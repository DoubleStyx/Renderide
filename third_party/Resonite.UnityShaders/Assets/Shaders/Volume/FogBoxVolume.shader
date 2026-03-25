// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Volume/FogBox"
{
	Properties
	{
		_BaseColor("Base Color", Color) = (0, 0, 0, 0)

		_AccumulationColor("Accumulation Color", Color) = (0.1,0.1,0.1,0.1)

		_AccumulationColorBottom("Accumulation Color Bottom", Color) = (0.1,0.1,0.0,0.1)
		_AccumulationColorTop("Accumulation Color Top", Color) = (0.1,0.1,0.1,0.1)

		_AccumulationRate("Accumulation Rate", Float) = 0.1
		_GammaCurve("Fog Power", Float) = 2.2

		_FogStart("Fog Start", Float) = 0
		_FogEnd("Fog End", Float) = 10000000

		_FogDensity("Fog Density", Range(0,1)) = 0.1

		_SrcBlend("SrcBlend", Float) = 1.0
		_DstBlend("DstBlend", Float) = 0.0
	}

		SubShader
	{
		Tags { "RenderType" = "Transparent+1000" "IgnoreProjector" = "True"
		 "Queue" = "Transparent" "DisableBatching" = "True" }

		Blend[_SrcBlend][_DstBlend], One One
		BlendOp Add, Max
		Cull Front
		ZWrite Off
		ZTest Always

		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			

			#pragma multi_compile _ SATURATE_ALPHA SATURATE_COLOR
			#pragma multi_compile COLOR_CONSTANT COLOR_VERT_GRADIENT
			#pragma multi_compile OBJECT_SPACE WORLD_SPACE
			#pragma multi_compile FOG_LINEAR FOG_EXP FOG_EXP2

			#include "UnityCG.cginc"
			#include "../Common.cginc"

			float4 _BaseColor;
			float _AccumulationRate;
			float _GammaCurve;

			float4 _AccumulationColor;
			float4 _AccumulationColorBottom;
			float4 _AccumulationColorTop;

			float _FogStart;
			float _FogEnd;
			float _FogDensity;

			struct a2v
			{
				half4 vertex : POSITION;

				UNITY_VERTEX_INPUT_INSTANCE_ID
			};

			struct v2f
			{
				half4 vertex : POSITION;
				half4 origin : TEXCOORD0;
				half4 projPos : TEXCOORD1;
				//float3 scale : TEXCOORD2;

				UNITY_VERTEX_OUTPUT_STEREO
			};

			float3 LinePlaneIntersection(float3 linePoint, float3 lineDirection, float3 planePoint, float3 planeNormal)
			{
				float3 diff = linePoint - planePoint;
				float prod1 = dot(diff, planeNormal);
				float prod2 = dot(lineDirection, planeNormal);
				float prod3 = prod1 / prod2;

				return linePoint - lineDirection * prod3;
			}

			void FilterPoint(float3 refPoint, float3 newPoint, float2 checkSquare, inout float3 lastPoint, inout float lastDistance)
			{
				if (any(abs(checkSquare) > 0.5))
					return;

				float dist = distance(refPoint, newPoint);

				if (dist < lastDistance)
				{
					lastDistance = dist;
					lastPoint = newPoint;
				}
			}

			float3 IntersectUnitCube(float3 linePoint, float3 lineDirection)
			{
				float3 i0 = LinePlaneIntersection(linePoint, lineDirection, float3(-0.5, 0, 0), float3(-1, 0, 0));
				float3 i1 = LinePlaneIntersection(linePoint, lineDirection, float3(0.5, 0, 0), float3(1, 0, 0));
				float3 i2 = LinePlaneIntersection(linePoint, lineDirection, float3(0, -0.5, 0), float3(0, -1, 0));
				float3 i3 = LinePlaneIntersection(linePoint, lineDirection, float3(0, 0.5, 0), float3(0, 1, 0));
				float3 i4 = LinePlaneIntersection(linePoint, lineDirection, float3(0, 0, -0.5), float3(0, 0, -1));
				float3 i5 = LinePlaneIntersection(linePoint, lineDirection, float3(0, 0, 0.5), float3(0, 0, 1));

				float dist = 65000;
				float3 p = linePoint;

				FilterPoint(linePoint, i0, i0.yz, p, dist);
				FilterPoint(linePoint, i1, i1.yz, p, dist);
				FilterPoint(linePoint, i2, i2.xz, p, dist);
				FilterPoint(linePoint, i3, i3.xz, p, dist);
				FilterPoint(linePoint, i4, i4.xy, p, dist);
				FilterPoint(linePoint, i5, i5.xy, p, dist);

				return p;
			}

			v2f vert(a2v v)
			{
				v2f o;

				UNITY_SETUP_INSTANCE_ID(v);
				UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

				o.vertex = UnityObjectToClipPos(v.vertex);
				o.origin = v.vertex;

				o.projPos = ComputeScreenPos(o.vertex);
				COMPUTE_EYEDEPTH(o.projPos.z);

				/*float4x4 m = unity_ObjectToWorld;
				o.scale = float3(
					sqrt(m._m00 * m._m00 + m._m10 * m._m10 + m._m20 * m._m20),
					sqrt(m._m01 * m._m01 + m._m11 * m._m11 + m._m21 * m._m21),
					sqrt(m._m02 * m._m02 + m._m12 * m._m12 + m._m22 * m._m22)
					);*/

				return o;
			}

			float distance_sqr(float3 a, float3 b)
			{
				float3 v = a - b;
				return dot(v, v);
			}

			float3 clamp_inside_unit_cube(float3 pos, float3 dir)
			{
				if (all(abs(pos) <= 0.5))
					return pos;
				else
					return IntersectUnitCube(pos, dir);
			}

			half4 frag(v2f i) : COLOR
			{
				UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

				half3 ndir;

				float sceneZ = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE_PROJ(_CameraDepthTexture, UNITY_PROJ_COORD(i.projPos)));
				float partZ = i.projPos.z;

				float3 start;
				float3 end;
				float3 camPos;

#ifdef OBJECT_SPACE
				camPos = mul(unity_WorldToObject, float4(_WorldSpaceCameraPos.xyz, 1)).xyz;
				float3 endPos = i.origin.xyz;

				ndir = normalize(endPos - camPos);

				// check if the camera position is inside of the cube
				start = clamp_inside_unit_cube(camPos, ndir);

				float maxDist = distance(camPos, endPos);
				float endRatio = min(sceneZ / partZ, 1);

				end = camPos + ndir * maxDist * endRatio;
#else
				// TODO!!! Can this code be factored out into the vertex shader instead? Particularly all the unit cube clipping
				float3 clampedStartPos = mul(unity_WorldToObject, float4(_WorldSpaceCameraPos.xyz, 1)).xyz;
				ndir = normalize(i.origin.xyz - clampedStartPos);
				clampedStartPos = clamp_inside_unit_cube(clampedStartPos, -ndir);

				float clampedDepth = -UnityObjectToViewPos(clampedStartPos).z;

				//return half4(clampedDepth.xxx, 1);
				
				if (clampedDepth > sceneZ)
					discard;

				// compute the sample origin position
				camPos = _WorldSpaceCameraPos.xyz;
				float3 endPos = mul(unity_ObjectToWorld, float4(i.origin.xyz, 1)).xyz;

				ndir = normalize(endPos - camPos);

				start = camPos;
				// TODO!!! This is technically not fully correct and will result in spherical distance, but calculating actual world position
				// requires inverse matrix, for computing clip to world transformation
				end = camPos + sceneZ * ndir;
#endif

				if (distance_sqr(camPos, end) < distance_sqr(camPos, start))
					discard;

				// total distance that needs to be travelled
				float dist = distance(start, end);

#ifdef FOG_LINEAR
				dist = min(_FogEnd, dist);
				dist -= _FogStart;
				dist = max(0, dist);
#endif

#ifdef FOG_EXP
				dist = 1 - (1 / exp(dist * _FogDensity));
#endif

#ifdef FOG_EXP2
				float d = dist * _FogDensity;
				dist = 1 - (1 / exp(d * d));
#endif
				float4 accColor;

#ifdef COLOR_CONSTANT
				accColor = _AccumulationColor;
#elif COLOR_VERT_GRADIENT
				// need to compute the start and end points within the cube volume
				float startY;
				float endY;
#ifdef OBJECT_SPACE
				startY = start.y;
				endY = end.y;
#else
				float3 localStart = mul(unity_WorldToObject, float4(start, 1)).xyz;
				float3 localEnd = mul(unity_WorldToObject, float4(end, 1)).xyz;

				localStart = clamp_inside_unit_cube(localStart, normalize(localStart));
				localEnd = clamp_inside_unit_cube(localEnd, normalize(localEnd));

				startY = localStart.y;
				endY = localEnd.y;
#endif
				float avgY = (startY + endY) * 0.5 + 0.5;

				accColor = lerp(_AccumulationColorBottom, _AccumulationColorTop, saturate(avgY));
#endif

				float4 acc = pow(dist * _AccumulationRate, _GammaCurve) * accColor;

				float4 resultColor = _BaseColor + acc;

#ifdef SATURATE_ALPHA
				resultColor.a = saturate(resultColor.a);
#elif SATURATE_COLOR
				resultColor = saturate(resultColor);
#endif

				return resultColor;
			}


			ENDCG
		}
	}
}
