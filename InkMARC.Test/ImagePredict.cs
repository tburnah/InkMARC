using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace InkMARC.Test
{
    public class ImageData
    {
        // Rename property to match your ONNX model's input name:
        [VectorType(448, 448, 3)]
        public float[] inputs { get; set; }
    }

    public class ImagePrediction
    {
        [ColumnName("output_0")] // or whatever your ONNX output node is called
        public float[] Prediction { get; set; }
    }

    public class OnnxImagePredictor : IDisposable
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;
        private readonly PredictionEngine<ImageData, ImagePrediction> _predEngine;

        public OnnxImagePredictor()
        {
            _mlContext = new MLContext();

            string onnxModelPath = @"C:\Users\tburnah\OneDrive - Massey University\Documents\my_model.onnx";

            // Because your C# property is "inputs", it matches the ONNX model input "inputs"
            var pipeline = _mlContext.Transforms.ApplyOnnxModel(
                modelFile: onnxModelPath,
                inputColumnNames: new[] { "inputs" },
                outputColumnNames: new[] { "output_0" }
            );

            var emptyData = _mlContext.Data.LoadFromEnumerable(new List<ImageData>());
            _model = pipeline.Fit(emptyData);

            _predEngine = _mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(_model);
        }

        public float PredictPressure(Bitmap image)
        {
            // Convert the Bitmap to the float[] shape [448*448*3]
            var imageData = new ImageData
            {
                inputs = ConvertBitmapToFloatArray(image)
            };

            // Predict
            ImagePrediction prediction = _predEngine.Predict(imageData);

            // If your ONNX model produces shape (1,) for a single output, 
            // you can return prediction[0].
            return prediction.Prediction[0];
        }

        public void Dispose()
        {
            _predEngine?.Dispose();
        }

        // Helper: Resize + normalize image to shape [448, 448, 3].
        private float[] ConvertBitmapToFloatArray(Bitmap bitmap)
        {
            int targetWidth = 448;
            int targetHeight = 448;

            using Bitmap resizedBitmap = new Bitmap(targetWidth, targetHeight);
            using (Graphics graphics = Graphics.FromImage(resizedBitmap))
            {
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.DrawImage(bitmap, 0, 0, targetWidth, targetHeight);
            }

            float[] tensorData = new float[targetWidth * targetHeight * 3];
            int index = 0;
            for (int y = 0; y < targetHeight; y++)
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    Color pixel = resizedBitmap.GetPixel(x, y);
                    tensorData[index++] = pixel.R / 255f;
                    tensorData[index++] = pixel.G / 255f;
                    tensorData[index++] = pixel.B / 255f;
                }
            }
            return tensorData;
        }
    }
}
