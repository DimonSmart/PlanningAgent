using System.Text.Json;
using System.Text.Json.Nodes;
using PlanningAgentDemo.Common;
using PlanningAgentDemo.Planning;

namespace PlanningAgentDemo.Agents;

public interface IAgentStepRunner
{
    Task<ResultEnvelope<JsonNode?>> ExecuteAsync(
        PlanStep step,
        JsonObject resolvedInputs,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Executes an agent step by calling the LLM with the prompts embedded in the plan step.
/// There is no agent registry — the planner decides system/user prompts when it creates the plan.
/// </summary>
public sealed class AgentStepRunner(ILlmClient llmClient) : IAgentStepRunner
{
    public async Task<ResultEnvelope<JsonNode?>> ExecuteAsync(
        PlanStep step,
        JsonObject resolvedInputs,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(step.Llm))
            return ResultEnvelope<JsonNode?>.Failure("llm_missing", $"Step '{step.Id}' has no llm label.");

        var systemPrompt = step.SystemPrompt ?? "You are a helpful assistant. Follow the user instruction carefully.";
        // Enforce JSON output at the system-prompt level so the LLM always returns parseable JSON.
        if (step.Out == "json" && !systemPrompt.Contains("JSON", StringComparison.OrdinalIgnoreCase))
            systemPrompt += " Output ONLY valid JSON. No explanation, no markdown, no code fences.";
        var userInstruction = step.UserPrompt ?? "Process the input and return the result.";

        var payload = new JsonObject { ["inputs"] = resolvedInputs };
        var fullUserPrompt = $"{userInstruction}\n\nInput:\n{payload.ToJsonString(new JsonSerializerOptions { WriteIndented = true })}";

        try
        {
            var raw = await llmClient.GenerateAsync(systemPrompt, fullUserPrompt, cancellationToken);

            if (step.Out == "string")
                return ResultEnvelope<JsonNode?>.Success(JsonValue.Create(raw.Trim()));

            try
            {
                var node = JsonResponseParser.ParseNodeFromLlm(raw);
                return ResultEnvelope<JsonNode?>.Success(node);
            }
            catch
            {
                return ResultEnvelope<JsonNode?>.Failure(
                    "json_parse_error",
                    $"LLM step '{step.Llm}' returned non-JSON. Raw: {raw[..Math.Min(raw.Length, 300)]}");
            }
        }
        catch (Exception ex)
        {
            return ResultEnvelope<JsonNode?>.Failure("llm_error", ex.Message);
        }
    }
}
