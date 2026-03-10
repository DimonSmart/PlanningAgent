using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using PlanningAgentDemo.Common;
using PlanningAgentDemo.Execution;
using PlanningAgentDemo.Tools;

namespace PlanningAgentDemo.Planning;

public sealed class LlmReplanner(
    IChatClient chatClient,
    IToolRegistry toolRegistry,
    IExecutionLogger? executionLogger = null) : IReplanner
{
    private const int MaxRounds = 6;
    private const int MaxResponseAttempts = 3;
    private readonly IExecutionLogger _log = executionLogger ?? NullExecutionLogger.Instance;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        TypeInfoResolver = new DefaultJsonTypeInfoResolver()
    };

    public async Task<PlanDefinition> ReplanAsync(PlannerReplanRequest request, CancellationToken cancellationToken = default)
    {
        var workflowTools = toolRegistry.ListPlannerMetadata();
        var session = new PlanEditingSession(request.Plan);
        var systemPrompt = BuildSystemPrompt(workflowTools);
        var agent = new ChatClientAgent(chatClient, systemPrompt, "replanner", null, null, null, null);
        JsonArray? lastActionResults = null;

        _log.Log($"[replan] start attempt={request.AttemptNumber} reason={Shorten(request.GoalVerdict.Reason, 240)}");

        for (var round = 1; round <= MaxRounds; round++)
        {
            var roundPrompt = BuildRoundPrompt(request, session, round, lastActionResults);
            var actionBatch = await GenerateActionBatchAsync(agent, roundPrompt, cancellationToken);
            var done = actionBatch["done"]?.GetValue<bool>() ?? false;
            var reason = actionBatch["reason"]?.GetValue<string>()?.Trim() ?? string.Empty;
            var actions = actionBatch["actions"] as JsonArray ?? [];

            _log.Log($"[replan] round={round} done={done} actions={actions.Count} reason={Shorten(reason, 240)}");
            _log.Log($"[replan] round={round} actionBatch={actionBatch.ToJsonString(new JsonSerializerOptions { WriteIndented = true })}");

            lastActionResults = ExecuteActions(session, request, actions);
            _log.Log($"[replan] round={round} actionResults={lastActionResults.ToJsonString(new JsonSerializerOptions { WriteIndented = true })}");

            if (!done)
                continue;

            try
            {
                var replanned = session.BuildPlan();
                PlanValidator.ValidateOrThrow(replanned, workflowTools);
                _log.Log($"[replan] success steps={replanned.Steps.Count} goal={Shorten(replanned.Goal, 240)}");
                _log.Log($"[replan] json {JsonSerializer.Serialize(replanned, new JsonSerializerOptions { WriteIndented = true })}");
                return replanned;
            }
            catch (Exception ex) when (round < MaxRounds)
            {
                lastActionResults.Add(new JsonObject
                {
                    ["tool"] = "plan.validateDraft",
                    ["ok"] = false,
                    ["error"] = new JsonObject
                    {
                        ["code"] = "invalid_plan",
                        ["message"] = ex.Message
                    }
                });
            }
        }

        throw new InvalidOperationException($"Replanner could not produce a valid plan after {MaxRounds} rounds.");
    }

    private async Task<JsonObject> GenerateActionBatchAsync(
        ChatClientAgent agent,
        string roundPrompt,
        CancellationToken cancellationToken)
    {
        Exception? lastError = null;
        var currentPrompt = roundPrompt;

        for (var attempt = 1; attempt <= MaxResponseAttempts; attempt++)
        {
            var response = await agent.RunAsync<JsonNode?>(currentPrompt, null, JsonOptions, null, cancellationToken);

            try
            {
                return DeserializeActionBatch(response.Result?.DeepClone());
            }
            catch (Exception ex) when (attempt < MaxResponseAttempts)
            {
                lastError = ex;
                var invalidJson = response.Result?.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) ?? "null";
                _log.Log($"[replan] response:retry attempt={attempt} error={Shorten(ex.Message, 240)}");
                currentPrompt = BuildRepairPrompt(roundPrompt, invalidJson, ex.Message);
            }
            catch (Exception ex)
            {
                lastError = ex;
                break;
            }
        }

        throw new InvalidOperationException(
            $"Replanner did not return a valid action batch after {MaxResponseAttempts} attempts. Last error: {lastError?.Message}",
            lastError);
    }

    private static JsonObject DeserializeActionBatch(JsonNode? responseNode)
    {
        if (responseNode is not JsonObject root)
            throw new InvalidOperationException("Replanner did not return a JSON object.");

        root = UnwrapActionRoot(root);
        if (root["done"] is null)
            throw new InvalidOperationException("Replanner response must include 'done'.");

        if (root["actions"] is not JsonArray actions)
            throw new InvalidOperationException("Replanner response must include 'actions' as an array.");

        if (root["done"]?.GetValue<bool>() == false && actions.Count == 0)
            throw new InvalidOperationException("Replanner must return at least one action when 'done' is false.");

        foreach (var actionNode in actions)
        {
            if (actionNode is not JsonObject action)
                throw new InvalidOperationException("Each action must be a JSON object.");

            var toolName = action["tool"]?.GetValue<string>()?.Trim();
            if (string.IsNullOrWhiteSpace(toolName))
                throw new InvalidOperationException("Each action must include 'tool'.");

            action["tool"] = toolName;
            action["in"] = action["in"] is JsonObject input ? input : new JsonObject();
        }

        root["reason"] = root["reason"]?.GetValue<string>()?.Trim() ?? string.Empty;
        return root;
    }

    private static JsonObject UnwrapActionRoot(JsonObject root)
    {
        if (HasActionShape(root))
            return root;

        foreach (var key in new[] { "result", "response", "data" })
        {
            if (root[key] is JsonObject nested && HasActionShape(nested))
                return nested;
        }

        foreach (var property in root)
        {
            if (property.Value is JsonObject nested && HasActionShape(nested))
                return nested;
        }

        return root;
    }

    private static bool HasActionShape(JsonObject node) =>
        node["done"] is not null || node["actions"] is JsonArray;

    private static string BuildSystemPrompt(IReadOnlyCollection<ToolPlannerMetadata> workflowTools)
    {
        var sb = new StringBuilder();
        sb.AppendLine("You are a replanning agent.");
        sb.AppendLine("You do NOT generate a full plan directly.");
        sb.AppendLine("You repair the working plan by calling plan-editing tools.");
        sb.AppendLine("Return ONLY JSON. No markdown. No prose outside the JSON.");
        sb.AppendLine();
        sb.AppendLine("Your response must have this exact shape:");
        sb.AppendLine("{");
        sb.AppendLine("  \"done\": true|false,");
        sb.AppendLine("  \"reason\": \"short explanation\",");
        sb.AppendLine("  \"actions\": [");
        sb.AppendLine("    { \"tool\": \"plan.readStep|plan.replaceStep|plan.addSteps|plan.resetFrom|runtime.readFailedTrace\", \"in\": { ... } }");
        sb.AppendLine("  ]");
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("Rules:");
        sb.AppendLine("- Use the existing working plan as the source of truth.");
        sb.AppendLine("- The working plan already contains per-step execution state in s/res/err.");
        sb.AppendLine("- Prefer the smallest correct repair.");
        sb.AppendLine("- Reuse successful upstream steps whenever possible.");
        sb.AppendLine("- Do not repeat a failed extraction/comparison prompt unchanged when the failure data explains what was missing.");
        sb.AppendLine("- Use runtime.readFailedTrace(stepId) when you need the compact structured details of a failed step.");
        sb.AppendLine("- If the failed trace identifies missingFacts and your edit removes that requirement or makes it optional, finish with done=true instead of iterating further.");
        sb.AppendLine("- plan.resetFrom(stepId) resets execution state for that step and all downstream steps so the executor will rerun them.");
        sb.AppendLine("- plan.replaceStep(stepId, step) replaces exactly one existing step in place.");
        sb.AppendLine("- plan.addSteps(afterStepId, steps) inserts new steps after the specified step. Use afterStepId=null only when rebuilding from an empty draft.");
        sb.AppendLine("- plan.readStep(stepId) returns the current JSON of one step.");
        sb.AppendLine("- When you are confident the working draft is ready, set done=true. You may include final edit actions in the same response.");
        sb.AppendLine();
        sb.AppendLine("Plan step rules:");
        sb.AppendLine("- A step must have exactly one of 'tool' or 'llm'.");
        sb.AppendLine("- LLM steps must have systemPrompt and userPrompt.");
        sb.AppendLine("- Use only the workflow tools listed below.");
        sb.AppendLine("- Keep references explicit using $stepId, $stepId.field, $stepId[], $stepId[].field, $stepId[n], or $stepId[n].field.");
        sb.AppendLine("- If a tool takes a scalar input and receives an array ref, the executor fans out automatically.");
        sb.AppendLine("- For extraction tasks, prefer a per-item LLM step with each=true.");
        sb.AppendLine();
        sb.AppendLine("Available workflow tools:");
        foreach (var tool in workflowTools)
        {
            sb.AppendLine($"- name: {tool.Name}");
            sb.AppendLine($"  description: {tool.Description}");
            sb.AppendLine($"  inputSchema: {tool.InputSchema.ToJsonString()}");
            sb.AppendLine($"  outputSchema: {tool.OutputSchema.ToJsonString()}");
        }

        return sb.ToString();
    }

    private static string BuildRoundPrompt(
        PlannerReplanRequest request,
        PlanEditingSession session,
        int round,
        JsonArray? lastActionResults)
    {
        var context = new JsonObject
        {
            ["userQuery"] = request.UserQuery,
            ["attemptNumber"] = request.AttemptNumber,
            ["replanRound"] = round,
            ["goalVerdict"] = JsonSerializer.SerializeToNode(request.GoalVerdict, JsonOptions),
            ["executionSummary"] = BuildExecutionSummary(request.ExecutionResult),
            ["failedTraceHints"] = BuildFailedTraceHints(request.ExecutionResult),
            ["workingPlan"] = session.GetCurrentPlanJson(),
            ["lastActionResults"] = lastActionResults?.DeepClone() ?? new JsonArray()
        };

        return $"Repair the working plan using the plan-editing tools.\n\nReplanning context:\n{context.ToJsonString(new JsonSerializerOptions { WriteIndented = true })}";
    }

    private static string BuildRepairPrompt(string originalPrompt, string invalidResponseJson, string errorMessage) =>
        $"{originalPrompt}\n\nYour previous response was invalid.\nValidation error: {errorMessage}\nInvalid response:\n{invalidResponseJson}\n\nReturn a corrected action batch as JSON only.";

    private static JsonArray ExecuteActions(
        PlanEditingSession session,
        PlannerReplanRequest request,
        JsonArray actions)
    {
        var results = new JsonArray();

        foreach (var actionNode in actions)
        {
            if (actionNode is not JsonObject action)
            {
                results.Add(CreateToolFailure("invalid_action", "Each action must be a JSON object."));
                continue;
            }

            var toolName = action["tool"]?.GetValue<string>()?.Trim();
            results.Add(string.Equals(toolName, "runtime.readFailedTrace", StringComparison.Ordinal)
                ? ExecuteRuntimeReadFailedTrace(request, action)
                : session.ExecuteAction(action));
        }

        return results;
    }

    private static JsonObject ExecuteRuntimeReadFailedTrace(PlannerReplanRequest request, JsonObject action)
    {
        var input = action["in"] as JsonObject ?? [];
        var stepId = input["stepId"]?.GetValue<string>()?.Trim();
        if (string.IsNullOrWhiteSpace(stepId))
            return CreateToolFailure("tool_error", "Action input 'stepId' is required.", "runtime.readFailedTrace");

        var failedTrace = request.ExecutionResult.StepTraces
            .FirstOrDefault(trace => string.Equals(trace.StepId, stepId, StringComparison.Ordinal) && !trace.Success);
        if (failedTrace is null)
        {
            return CreateToolFailure(
                "tool_error",
                $"Failed trace for step '{stepId}' was not found.",
                "runtime.readFailedTrace");
        }

        return new JsonObject
        {
            ["tool"] = "runtime.readFailedTrace",
            ["ok"] = true,
            ["output"] = BuildFailedTraceSummary(failedTrace)
        };
    }

    private static JsonArray BuildExecutionSummary(ExecutionResult executionResult)
    {
        var summary = new JsonArray();
        foreach (var trace in executionResult.StepTraces)
        {
            summary.Add(new JsonObject
            {
                ["stepId"] = trace.StepId,
                ["success"] = trace.Success,
                ["errorCode"] = trace.ErrorCode,
                ["verificationIssues"] = new JsonArray(trace.VerificationIssues.Select(issue => JsonValue.Create(issue.Code)).ToArray())
            });
        }

        return summary;
    }

    private static JsonArray BuildFailedTraceHints(ExecutionResult executionResult)
    {
        var failedTraces = new JsonArray();
        foreach (var trace in executionResult.StepTraces.Where(trace => !trace.Success))
            failedTraces.Add(BuildFailedTraceSummary(trace));

        return failedTraces;
    }

    private static JsonObject BuildFailedTraceSummary(StepExecutionTrace failedTrace)
    {
        var missingFacts = new HashSet<string>(StringComparer.Ordinal);
        var observedEvidence = new HashSet<string>(StringComparer.Ordinal);
        if (failedTrace.ErrorDetails?["errors"] is JsonArray errors)
        {
            foreach (var errorNode in errors.OfType<JsonObject>())
            {
                if (errorNode["details"] is not JsonObject details)
                    continue;

                if (details["missingFacts"] is JsonArray facts)
                {
                    foreach (var factNode in facts)
                    {
                        var fact = factNode?.GetValue<string>()?.Trim();
                        if (!string.IsNullOrWhiteSpace(fact))
                            missingFacts.Add(fact);
                    }
                }

                var evidence = details["observedEvidence"]?.GetValue<string>()?.Trim();
                if (!string.IsNullOrWhiteSpace(evidence))
                    observedEvidence.Add(evidence);
            }
        }

        return new JsonObject
        {
            ["stepId"] = failedTrace.StepId,
            ["errorCode"] = failedTrace.ErrorCode,
            ["errorMessage"] = failedTrace.ErrorMessage,
            ["status"] = failedTrace.ErrorDetails?["status"]?.DeepClone(),
            ["needsReplan"] = failedTrace.ErrorDetails?["needsReplan"]?.DeepClone(),
            ["missingFacts"] = new JsonArray(missingFacts.Select(fact => JsonValue.Create(fact)).ToArray()),
            ["observedEvidence"] = new JsonArray(observedEvidence.Select(evidence => JsonValue.Create(evidence)).ToArray())
        };
    }

    private static JsonObject CreateToolFailure(string code, string message, string? toolName = null) => new()
    {
        ["tool"] = toolName,
        ["ok"] = false,
        ["error"] = new JsonObject
        {
            ["code"] = code,
            ["message"] = message
        }
    };

    private static string Shorten(string? value, int maxLength)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "<empty>";

        var normalized = value.ReplaceLineEndings(" ").Trim();
        return normalized.Length <= maxLength
            ? normalized
            : $"{normalized[..maxLength]}...";
    }
}
