Shader "Volume/Unlit"
{
	Properties {
		_Volume ("Volume", 3D) = "" {}
		_StepSize ("Step Size", Float) = 0.1
		_Gain ("Gain", Float) = 0.1

		_Exp ("Exponent", Float) = 1

		_LowClip ("Low Clip", Float) = 0
		_HighClip ("High Clip", Float) = 1

		_AccumulationCutoff ("Accumulation Cutoff", Float) = 100
		_HitThreshold ("Display Threshold", Float) = 0.5
	}
	SubShader {
		Tags { "RenderType"="Transparent" "IgnoreProjector"="True"
		 "Queue"="Transparent" "DisableBatching"="True" }

		Blend[_SrcBlend][_DstBlend], One One
		BlendOp Add, Max
		Cull Front
		ZWrite Off
		ZTest Always
		LOD 200

		Pass
		{
		
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment frag

		#pragma multi_compile _ ALPHA_CHANNEL
		#pragma multi_compile ADDITIVE ADDITIVE_CUTOFF HIT_THRESHOLD

		#pragma multi_compile HIGHLIGHT0 HIGHLIGHT1 HIGHLIGHT2 HIGHLIGHT3 HIGHLIGHT4
		#pragma multi_compile SLICE0 SLICE1 SLICE2 SLICE3 SLICE4
		
		
		#include "UnityCG.cginc"

		half _StepSize;
		half _Gain;

		half _AccumulationCutoff;
		half _HitThreshold;

		//#ifndef HIGHLIGHT0
		uniform half3 _HighlightNormal[4];
		uniform half _HighlightOffset[4];
		uniform half _HighlightRange[4];
		uniform half4 _HighlightColor[4];
		//#endif

		//#ifndef SLICE0
		uniform half3 _SlicerNormal[4];
		uniform half _SlicerOffset[4];
		//#endif

		uniform half _Exp;

		uniform half _LowClip;
		uniform half _HighClip;

		UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);
		sampler3D _Volume;

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
			float3 scale : TEXCOORD2;

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

		half cut(half v)
		{
			if(v < 0 || v > 1)
				return 0;
			return v;
		}

		half4 rescale(half4 c)
		{
			c -= _LowClip;
			c /= _HighClip - _LowClip;
			return half4(cut(c.x), cut(c.y), cut(c.z), cut(c.w));
			/*
			c = saturate(c);
			return c;*/
		}

		half surf_distance(half3 p, half3 normal, half offset)
		{
			return dot(p, normal) + offset;
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

			float4x4 m = unity_ObjectToWorld;
			o.scale = float3(
				sqrt(m._m00 * m._m00 + m._m10 * m._m10 + m._m20 * m._m20),
				sqrt(m._m01 * m._m01 + m._m11 * m._m11 + m._m21 * m._m21),
				sqrt(m._m02 * m._m02 + m._m12 * m._m12 + m._m22 * m._m22)
				);

			return o;
		}

		fixed4 frag(v2f i) : COLOR
		{
			//UNITY_SETUP_INSTANCE_ID(i);
			UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

#if	   defined(HIGHLIGHT1)
			const int HIGHLIGHT_ITERATIONS = 1;
#elif  defined(HIGHLIGHT2)
			const int HIGHLIGHT_ITERATIONS = 2;
#elif  defined(HIGHLIGHT3)
			const int HIGHLIGHT_ITERATIONS = 3;
#elif  defined(HIGHLIGHT4)
			const int HIGHLIGHT_ITERATIONS = 4;
#endif

#if    defined(SLICE1)
		const int SLICE_ITERATIONS = 1;
#elif  defined(SLICE2)
		const int SLICE_ITERATIONS = 2;
#elif  defined(SLICE3)
		const int SLICE_ITERATIONS = 3;
#elif  defined(SLICE4)
		const int SLICE_ITERATIONS = 4;
#endif

			half3 pos;
			half3 dir; 
			half3 ndir;

			float sceneZ = LinearEyeDepth (SAMPLE_DEPTH_TEXTURE_PROJ(_CameraDepthTexture, UNITY_PROJ_COORD(i.projPos)));
			float partZ = i.projPos.z;

			float endRatio = min(sceneZ / partZ, 1);

			// compute the sample origin position
			float3 camPos = mul(unity_WorldToObject, float4(_WorldSpaceCameraPos.xyz, 1)).xyz;

			ndir = normalize(i.origin - camPos);

			// check if the camera position is inside of the cube
			if (all(abs(camPos) <= 0.5))
				pos = camPos;
			else
				pos = IntersectUnitCube(camPos, ndir);
			//pos = LinePlaneIntersection(camPos, ndir, float3(0, 0, 0), float3(0, 0, -1));

			//return fixed4(distance(pos, camPos).xxx / 10, 1);
			//return fixed4((abs(pos * 4) % 1) > 0.5, 1);

			//pos = i.origin;
			dir = ndir * _StepSize;

			// move the origin based on the depth
			//float zdiff = partZ - sceneZ;
			//zdiff = max(0, zdiff); // make it zero if the scene is behind the back face

			//zdiff /= length(ndir * i.scale);

			//pos += ndir * zdiff;

			float3 start = pos;
			float3 end = i.origin;

			float maxDist = distance(camPos, end);
			end = camPos + ndir * maxDist * endRatio;

			// total distance that needs to be travelled
			float dist = distance(start, end);

			if (distance(camPos, end) < distance(camPos, start))
				discard;

			half steps = dist / _StepSize;

			// start the sampling, continuing until the other side is reached
			half4 acc = half4(0,0,0,0);

#ifdef HIT_THRESHOLD 
			acc = half4(0, 0, 0, -1);
#endif

			steps = min(steps, 1024); 

			half gain = _Gain;

#if defined(ADDITIVE) || defined(ADDITIVE_CUTOFF)
			gain *= length(ndir * normalize(i.scale)) * _StepSize;
#endif
			

#if !defined(SHADER_API_GLES)
			[loop]
#endif
			for(int f = 0; f < steps; f++)
			{
				//if (abs(pos.x) > 0.5001 || abs(pos.y) > 0.5001 || abs(pos.z) > 0.5001)
				//	break;

				//// TODO, optimize to compare against precomputed distance
				//if (distance(pos, camPos) < _StepSize*1.5)
				//	break;

				#if ALPHA_CHANNEL
					half4 c = tex3D(_Volume, pos + 0.5).aaaa;
				#else
					half4 c = tex3D(_Volume, pos + 0.5);
				#endif

				//c = 1;

	/*			half3 spos = pos * i.scale;
				float v = sin(spos.x*20) * cos(spos.y*10) * tan(spos.z*5);
				c = half4(v.xxx, 1);*/

				c = pow( rescale(c), _Exp) * gain;

				#if !defined(SLICE0)
				for (int i = 0; i < SLICE_ITERATIONS; i++)
					if (surf_distance(pos, _SlicerNormal[i], _SlicerOffset[i]) < 0)
						c *= 0;
				#endif

				#ifndef HIGHLIGHT0
				for (int i = 0; i < HIGHLIGHT_ITERATIONS; i++)
					if (abs(surf_distance(pos, _HighlightNormal[i], _HighlightOffset[i])) < _HighlightRange[i])
						c *= _HighlightColor[i];
				#endif

#if defined(ADDITIVE) || defined(ADDITIVE_CUTOFF)
				acc += c;
#elif defined(HIT_THRESHOLD)
				if ((c.x + c.y + c.z) * 0.3333 >= _HitThreshold)
				{
					acc = c;
					break;
				}
#endif

#ifdef ADDITIVE_CUTOFF
				if (((acc.x + acc.y + acc.z) * 0.3333) > _AccumulationCutoff)
					break;
#endif

				pos += dir;
			}

			//return fixed4(dist.xxx / 100, 1);

#ifdef HIT_THRESHOLD
			if (acc.a < 0)
				discard;
#endif

#ifdef ADDITIVE_CUTOFF
			acc.xyz /= min(1, _AccumulationCutoff);
#endif

			return fixed4(acc.xyz, 1);
		}

		ENDCG

		}
	} 
	FallBack "Diffuse"
}
