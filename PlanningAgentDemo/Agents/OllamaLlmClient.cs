using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace PlanningAgentDemo.Agents;

public sealed class OllamaLlmClient(HttpClient httpClient, string? model = null) : ILlmClient
{
    public const string DefaultModel = "qwen3.5:latest";
    public const string DevModel = "gpt-oss:120b-cloud";

    public async Task<string> GenerateAsync(string systemPrompt, string userPrompt, CancellationToken cancellationToken = default)
    {
        var requestJson = new JsonObject
        {
            ["model"] = model ?? DefaultModel,
            ["stream"] = false,
            ["messages"] = new JsonArray
            {
                new JsonObject { ["role"] = "system", ["content"] = systemPrompt },
                new JsonObject { ["role"] = "user", ["content"] = userPrompt }
            }
        };

        using var request = new HttpRequestMessage(HttpMethod.Post, "/api/chat")
        {
            Content = new StringContent(requestJson.ToJsonString(), Encoding.UTF8, "application/json")
        };

        using var response = await httpClient.SendAsync(request, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(cancellationToken);
            throw new HttpRequestException(
                $"Ollama /api/chat failed with {(int)response.StatusCode} {response.StatusCode}. Body: {errorBody}");
        }

        var payload = await response.Content.ReadAsStringAsync(cancellationToken);
        var root = JsonNode.Parse(payload) as JsonObject;
        var content = root?["message"]?["content"]?.GetValue<string>();

        if (string.IsNullOrWhiteSpace(content))
        {
            throw new InvalidOperationException("Ollama response does not contain message.content.");
        }

        return content;
    }
}
