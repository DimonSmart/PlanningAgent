using System.Text.Json.Nodes;

namespace PlanningAgentDemo.Execution;

public static class JsonPathEvaluator
{
    // Minimal MVP jsonPath support: $.a.b[0]
    public static JsonNode? Select(JsonNode? root, string jsonPath)
    {
        if (root is null)
        {
            return null;
        }

        if (string.IsNullOrWhiteSpace(jsonPath) || jsonPath == "$")
        {
            return root;
        }

        if (!jsonPath.StartsWith("$", StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"Unsupported jsonPath '{jsonPath}'.");
        }

        var current = root;
        var i = 1;

        while (i < jsonPath.Length)
        {
            if (jsonPath[i] == '.')
            {
                i++;
                var start = i;
                while (i < jsonPath.Length && jsonPath[i] != '.' && jsonPath[i] != '[')
                {
                    i++;
                }

                var propertyName = jsonPath[start..i];
                current = (current as JsonObject)?[propertyName];
                continue;
            }

            if (jsonPath[i] == '[')
            {
                i++;
                var start = i;
                while (i < jsonPath.Length && jsonPath[i] != ']')
                {
                    i++;
                }

                if (i >= jsonPath.Length)
                {
                    throw new InvalidOperationException($"Invalid jsonPath '{jsonPath}'.");
                }

                var indexText = jsonPath[start..i];
                i++;

                if (!int.TryParse(indexText, out var index))
                {
                    throw new InvalidOperationException($"Only numeric indexes are supported in jsonPath '{jsonPath}'.");
                }

                current = (current as JsonArray)?[index];
                continue;
            }

            throw new InvalidOperationException($"Unsupported token in jsonPath '{jsonPath}' at position {i}.");
        }

        return current;
    }
}
