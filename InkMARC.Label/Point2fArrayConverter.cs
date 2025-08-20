using System;
using System.Text.Json;
using System.Text.Json.Serialization;
using OpenCvSharp;

namespace InkMARC.Label
{
    public class Point2fArrayConverter : JsonConverter<Point2f[]>
    {
        public override Point2f[] Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            var points = new List<Point2f>();

            if (reader.TokenType != JsonTokenType.StartArray)
                throw new JsonException("Expected StartArray");

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndArray)
                    break;

                if (reader.TokenType != JsonTokenType.StartObject)
                    throw new JsonException("Expected StartObject for Point2f");

                float x = 0, y = 0;

                while (reader.Read())
                {
                    if (reader.TokenType == JsonTokenType.EndObject)
                        break;

                    if (reader.TokenType != JsonTokenType.PropertyName)
                        throw new JsonException();

                    string propName = reader.GetString();
                    reader.Read();

                    switch (propName)
                    {
                        case "X":
                            x = reader.GetSingle();
                            break;
                        case "Y":
                            y = reader.GetSingle();
                            break;
                    }
                }

                points.Add(new Point2f(x, y));
            }

            return points.ToArray();
        }

        public override void Write(Utf8JsonWriter writer, Point2f[] value, JsonSerializerOptions options)
        {
            writer.WriteStartArray();

            foreach (var pt in value)
            {
                writer.WriteStartObject();
                writer.WriteNumber("X", pt.X);
                writer.WriteNumber("Y", pt.Y);
                writer.WriteEndObject();
            }

            writer.WriteEndArray();
        }
    }
}
