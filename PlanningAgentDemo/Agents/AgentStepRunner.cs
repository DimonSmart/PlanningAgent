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
        systemPrompt += BuildExecutionContract(step.Out);
        var userInstruction = step.UserPrompt ?? "Process the input and return the result.";

        var payload = new JsonObject { ["inputs"] = resolvedInputs };
        var fullUserPrompt = $"{userInstruction}\n\nInput:\n{payload.ToJsonString(new JsonSerializerOptions { WriteIndented = true })}";

        try
        {
            var raw = await llmClient.GenerateAsync(systemPrompt, fullUserPrompt, cancellationToken);

            if (TryParseExecutionIssue(raw, step.Out, out var issueEnvelope))
                return issueEnvelope;

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

    private static string BuildExecutionContract(string? outputType)
    {
        var resultHint = string.Equals(outputType, "string", StringComparison.OrdinalIgnoreCase)
            ? "a plain string when successful"
            : "the requested JSON payload when successful";

        return $"\n\nIf the task cannot be completed reliably from the provided input, return valid JSON instead of guessing. Use this exact top-level shape: {{\"_execution\":{{\"status\":\"blocked\"|\"partial\",\"needsReplan\":true,\"errors\":[{{\"code\":\"short_code\",\"message\":\"human readable message\"}}]}},\"result\":null}}. If you have a partial but still useful result, put it into 'result'. If you can complete the task reliably, return {resultHint}.";
    }

    private static bool TryParseExecutionIssue(string raw, string? outputType, out ResultEnvelope<JsonNode?> envelope)
    {
        envelope = default!;

        JsonObject? root;
        try
        {
            root = JsonResponseParser.ParseNodeFromLlm(raw) as JsonObject;
        }
        catch
        {
            return false;
        }

        if (root is null || root["_execution"] is not JsonObject execution)
            return false;

        var status = execution["status"]?.GetValue<string>() ?? "blocked";
        var needsReplan = execution["needsReplan"]?.GetValue<bool>() ?? false;
        var errors = execution["errors"] as JsonArray;
        var hasErrors = errors is { Count: > 0 };

        if (!needsReplan && !hasErrors && string.Equals(status, "ok", StringComparison.OrdinalIgnoreCase))
        {
            var resultNode = root["result"]?.DeepClone();
            envelope = ResultEnvelope<JsonNode?>.Success(NormalizeWrappedResult(resultNode, outputType));
            return true;
        }

        var message = hasErrors
            ? string.Join("; ", errors!.Select(error => error?["message"]?.GetValue<string>() ?? "unknown issue"))
            : $"LLM reported status '{status}'.";

        envelope = ResultEnvelope<JsonNode?>.Failure(
            "llm_reported_issue",
            message,
            new JsonObject
            {
                ["status"] = status,
                ["needsReplan"] = needsReplan,
                ["errors"] = errors?.DeepClone(),
                ["result"] = root["result"]?.DeepClone()
            });
        return true;
    }

    private static JsonNode? NormalizeWrappedResult(JsonNode? resultNode, string? outputType)
    {
        if (!string.Equals(outputType, "string", StringComparison.OrdinalIgnoreCase))
            return resultNode;

        if (resultNode is JsonValue value && value.TryGetValue<string>(out var text))
            return JsonValue.Create(text);

        return JsonValue.Create(resultNode?.ToJsonString() ?? string.Empty);
    }
}
