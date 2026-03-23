using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

using System.IO;
using Random = UnityEngine.Random;

public class PoissonDiscGenerator : ScriptableWizard
{
    public int Samples = 32;
    public bool ShaderCode;
    public bool CSharpCode;
    public bool OneDimension;
    public bool TwoDimensions;
    public bool ThreeDimensions;
    public string ShaderPath = "Assets/APP/Unity Environment/Shaders/EVRPoissonDisc.cginc";

    [MenuItem("Assets/Generate Poisson Disc Data")]
    static void CreateWizard()
    {
        ScriptableWizard.DisplayWizard<PoissonDiscGenerator>("Generate Poisson Disc Samples", "Generate");
    }

    void OnWizardCreate()
    {
        var oneDimension = new List<float>();
        var twoDimensions = new List<Vector2>();
        var threeDimensions = new List<Vector3>();

        if (OneDimension)
        {
            oneDimension = OneDimensionalSamples(Samples);
        }

        if (TwoDimensions)
        {
            twoDimensions = TwoDimensionalSamples(Samples);
        }

        if (ThreeDimensions)
        {
            threeDimensions = ThreeDimensionalSamples(Samples);
        }
        
        // Generate some code.

        if (ShaderCode)
        {
            if (File.Exists(ShaderPath))
                File.Delete(ShaderPath);
            
            var file = File.CreateText(ShaderPath);
            var output = "";

            if (OneDimension)
            {
                file.WriteLine($"static const float POISSON1D_SAMPLES[{Samples}] = " +"{");

                for (int i = 0; i < oneDimension.Count; i++)
                {
                    string line = $"    {oneDimension[i]}";

                    if (i < oneDimension.Count - 1)
                    {
                        line += ",";
                    }

                    file.WriteLine(line);
                }
                file.WriteLine("};");
            }

            if (TwoDimensions)
            {
                file.WriteLine("static const half2 POISSON2D_SAMPLES[" + Samples + "] = {");

                for (int i = 0; i < twoDimensions.Count; i++)
                {
                    string line = "    {" + twoDimensions[i].x + ", " + twoDimensions[i].y + "}";

                    if (i < twoDimensions.Count - 1)
                    {
                        line += ",";
                    }
                    
                    file.WriteLine(line);
                }

                file.WriteLine("};");
            }

            if (ThreeDimensions)
            {
                file.WriteLine("static const half3 POISSON3D_SAMPLES[" + Samples + "] = {");
                for (int i = 0; i < threeDimensions.Count; i++)
                {
                    string line = "    {" + threeDimensions[i].x + ", " + threeDimensions[i].y + ", " + threeDimensions[i].z + "}";

                    if (i < threeDimensions.Count - 1)
                    {
                        line += ",";
                    }
                    
                    file.WriteLine(line);
                }

                file.WriteLine("};");
            }
            
            file.Close();
        }
        
        AssetDatabase.Refresh();
    }
    
    public int TryLimit = 1000;
    public float Factor = 0.75f;
    public float Epsilon = 0.5f;
    public Vector2 MinMax = new Vector2(0, 1);

    List<float> OneDimensionalSamples(int samples)
    {
        List<float> points = new List<float>();

        int tries;
        float newPoint;
        float epsilon = Epsilon;

        while (points.Count < samples)
        {
            tries = 0;
            var rand = new System.Random();
            do
            {
                newPoint = (float)rand.Next(Int32.MaxValue) / (float)Int32.MaxValue;
                tries++;
            } while (points.Exists(x => Mathf.Abs(x - newPoint) <= epsilon) && tries < TryLimit);

            if (tries < TryLimit)
            {
                points.Add(newPoint);
            }
            else
            {
                epsilon *= Factor;
                
                if (epsilon < 0.00001f)
                    epsilon = Epsilon;
            }
        }

        return points;
    }
    
    List<Vector2> TwoDimensionalSamples(int samples)
    {
        List<Vector2> points = new List<Vector2>();

        int tries;
        Vector2 newPoint;
        float epsilon = Epsilon;

        while (points.Count < samples)
        {
            tries = 0;

            do
            {
                newPoint = new Vector2(Random.Range(MinMax.x, MinMax.y), Random.Range(MinMax.x, MinMax.y));
                tries++;
            } while (points.Exists(x => Vector2.Distance(x, newPoint) <= epsilon) && tries < TryLimit);

            if (tries < TryLimit)
            {
                points.Add(newPoint);
            }
            else
            {
                epsilon *= Factor;
            }
        }

        return points;
    }
    
    List<Vector3> ThreeDimensionalSamples(int samples)
    {
        List<Vector3> points = new List<Vector3>();

        int tries;
        Vector3 newPoint;
        float epsilon = Epsilon;

        while (points.Count < samples)
        {
            tries = 0;

            do
            {
                newPoint = new Vector3(Random.Range(MinMax.x, MinMax.y), Random.Range(MinMax.x, MinMax.y), Random.Range(MinMax.x, MinMax.y));
                tries++;
            } while (points.Exists(x => Vector2.Distance(x, newPoint) <= epsilon) && tries < TryLimit);

            if (tries < TryLimit)
            {
                points.Add(newPoint);
            }
            else
            {
                epsilon *= Factor;
            }
        }

        return points;
    }

    void OnWizardUpdate()
    {
        helpString = "Please set the color of the light!";
    }
}
