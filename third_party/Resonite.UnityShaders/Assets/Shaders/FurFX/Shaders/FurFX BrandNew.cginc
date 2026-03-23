// Upgrade NOTE: replaced '_World2Object' with 'unity_WorldToObject'

#include "AutoLight.cginc"
#include "UnityCG.cginc"

struct appdata {
	float4 vertex : POSITION;
	fixed3 normal : NORMAL;
	half4 texcoord : TEXCOORD0;
};

struct v2f {
	float4 pos : SV_POSITION;
	half4 uv : TEXCOORD0;
	half4 uv2 : TEXCOORD1;
	fixed4 color : COLOR0;	
};

      struct Input {
          float2 uv_MainTex;
          float2 uv_BumpMap;
          float3 worldRefl;
          float3 viewDir;
      };

uniform float _FurLength;
uniform float _EdgeFade;
uniform float4 _ForceGlobal;
uniform float4 _ForceLocal;
uniform float _HairHardness;
uniform float _HairThinness;
uniform float _SkinAlpha;
uniform float _HairShading;
uniform float _HairColoring;
uniform float4 _Color;
//uniform fixed4 _SpecColor;
uniform float _Shininess;
uniform float _Gloss;

uniform float _ShininessFur;
uniform float _GlossFur;

uniform float4 _ReflColor;
uniform float _Reflection;
uniform float _ReflMinLevel;

uniform float4 _RimColor;
uniform float _RimPower;

uniform float4 _BonusAmbient;

//uniform fixed4 _LightColor0; 

sampler2D _MainTex;
sampler2D _BumpMap;
sampler2D _NoiseTex;
samplerCUBE _Cube;

//uniform half4 _MainTex_ST;
//uniform half4 _NoiseTex_ST;

float4 tessFixed()
{
    return _HairThinness;
}  

void surfBase (Input IN, inout SurfaceOutput o) {
	fixed4 tex = tex2D(_MainTex, IN.uv_MainTex);
	o.Albedo = tex.rgb * _Color.rgb;
	o.Gloss = _Gloss;
	o.Alpha = 1.0;
	o.Specular = _Shininess;
	o.Normal = UnpackNormal(tex2D(_BumpMap, IN.uv_BumpMap));
	o.Albedo += _BonusAmbient;
}

void surfFur (Input IN, inout SurfaceOutput o) {
	fixed4 tex = tex2D(_MainTex, IN.uv_MainTex);
	
	clip ( FUR_MULTIPLIER > max(tex.a,_SkinAlpha) ? -1 : 1 );
	
	fixed3 noise = tex2D(_NoiseTex, IN.uv_MainTex * _HairThinness).rgb;
	
	o.Albedo = tex.rgb * _Color.rgb;
	o.Gloss = _Gloss * noise.r;
	//o.Alpha = 1.0;
	//o.Alpha = clamp(noise - (FUR_MULTIPLIER*FUR_MULTIPLIER)*_EdgeFade,0,1);
	o.Alpha = noise.r;
	o.Specular = _Shininess;
	
	half rim = 1.0 - saturate(dot (normalize(IN.viewDir), normalize(o.Normal)));
	o.Emission = _RimColor.rgb * pow (rim, _RimPower);
	
	float4 reflTex = texCUBE(_Cube, IN.worldRefl) * _ReflColor;	
	reflTex = lerp(float4(0,0,0,0),reflTex,max(FUR_MULTIPLIER,_ReflMinLevel));
	o.Albedo += reflTex.rgb * _Reflection * noise.r;
	
	//o.Albedo -= (pow(1-(FUR_MULTIPLIER/tex.a),1)) * _HairShading;
	
	fixed3 shadow = tex2D(_NoiseTex, IN.uv_MainTex * _HairThinness + (0.1,0.1)).rgb;
	o.Albedo += ((shadow.rgb*2)-1) * 0.5 * tex * _HairColoring;
	
	o.Albedo -= (FUR_MULTIPLIER) * _HairShading * tex * 2;	
	
	o.Albedo += _BonusAmbient;
}

void vert (inout appdata_full v) {
     v.vertex.xyz += v.normal * _FurLength * FUR_MULTIPLIER * _HairHardness;
     v.vertex.xyz += (mul(unity_WorldToObject,clamp(_ForceGlobal,-1,1)).xyz + mul(UNITY_MATRIX_P,clamp(_ForceLocal,-1,1)).xyz) * pow(FUR_MULTIPLIER,2) * _FurLength;
}
   
