
using System.Text.Json;
using System.Text.Json.Serialization;

namespace InkMARC.Label.Services
{
    public sealed class IntPairValueTupleConverter : JsonConverter<(int x, int y)>
    {
        public override (int x, int y) Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType == JsonTokenType.StartObject)
            {
                int x = 0, y = 0;
                bool hasX = false, hasY = false;

                using var doc = JsonDocument.ParseValue(ref reader);
                var obj = doc.RootElement;

                if (obj.TryGetProperty("x", out var xProp)) { x = xProp.GetInt32(); hasX = true; }
                if (obj.TryGetProperty("y", out var yProp)) { y = yProp.GetInt32(); hasY = true; }

                // Legacy Tuple<T1,T2> shape
                if (obj.TryGetProperty("Item1", out var i1)) { x = i1.GetInt32(); hasX = true; }
                if (obj.TryGetProperty("Item2", out var i2)) { y = i2.GetInt32(); hasY = true; }

                if (!hasX || !hasY) throw new JsonException("Expected properties for a 2-tuple.");

                return (x, y);
            }

            if (reader.TokenType == JsonTokenType.StartArray)
            {
                reader.Read(); var x = reader.GetInt32();
                reader.Read(); var y = reader.GetInt32();

                // consume any remaining items (shouldn’t be any) and the EndArray
                while (reader.TokenType != JsonTokenType.EndArray) reader.Read();

                return (x, y);
            }

            throw new JsonException("Expected object or array for a 2-tuple.");
        }

        public override void Write(Utf8JsonWriter writer, (int x, int y) value, JsonSerializerOptions options)
        {
            // Write with explicit names for clarity going forward
            writer.WriteStartObject();
            writer.WriteNumber("x", value.x);
            writer.WriteNumber("y", value.y);
            writer.WriteEndObject();
        }
    }
}
