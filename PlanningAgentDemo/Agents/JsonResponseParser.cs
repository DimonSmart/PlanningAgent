using System.Text.Json;
using System.Text.Json.Nodes;
using DimonSmart.AiUtils;

namespace PlanningAgentDemo.Agents;

public static class JsonResponseParser
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public static JsonNode ParseNodeFromLlm(string rawResponse)
    {
        var json = JsonExtractor.ExtractJson(rawResponse);
        var node = JsonNode.Parse(json);
        return node ?? throw new InvalidOperationException("LLM returned empty JSON.");
    }

    public static T ParseTypedFromLlm<T>(string rawResponse)
    {
        var node = ParseNodeFromLlm(rawResponse);
        var typed = node.Deserialize<T>(JsonOptions);
        return typed ?? throw new InvalidOperationException($"Failed to deserialize JSON into {typeof(T).Name}.");
    }
}
