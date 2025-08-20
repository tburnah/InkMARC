using OpenCvSharp;
using System.Text.Json.Serialization;
using System.Text.Json;

namespace InkMARC.Label
{
    public class Point2fDictionaryConverter : JsonConverter<Dictionary<int, Point2f[]>>
    {
        public override Dictionary<int, Point2f[]> Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            Console.WriteLine($"Token at entry: {reader.TokenType}");

            if (reader.TokenType == JsonTokenType.Null)
                return new Dictionary<int, Point2f[]>(); // Gracefully handle null value

            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException($"Expected StartObject but got {reader.TokenType}");

            var result = new Dictionary<int, Point2f[]>();

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    break;

                if (reader.TokenType != JsonTokenType.PropertyName)
                    throw new JsonException();

                string propertyName = reader.GetString();
                if (!int.TryParse(propertyName, out int key))
                    throw new JsonException($"Invalid dictionary key: {propertyName}");

                reader.Read();

                using (var doc = JsonDocument.ParseValue(ref reader))
                {
                    var points = JsonSerializer.Deserialize<Point2f[]>(doc.RootElement.GetRawText(), options);
                    result[key] = points!;
                }
            }

            return result;
        }

        public override void Write(Utf8JsonWriter writer, Dictionary<int, Point2f[]> value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();

            foreach (var kvp in value)
            {
                writer.WritePropertyName(kvp.Key.ToString());
                JsonSerializer.Serialize(writer, kvp.Value, options);
            }

            writer.WriteEndObject();
        }
    }
}
