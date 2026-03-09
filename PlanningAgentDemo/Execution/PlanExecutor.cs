using System.Text.Json.Nodes;
using PlanningAgentDemo.Agents;
using PlanningAgentDemo.Common;
using PlanningAgentDemo.Planning;
using PlanningAgentDemo.Tools;
using PlanningAgentDemo.Verification;

namespace PlanningAgentDemo.Execution;

using System.Text.Json;

/// <summary>
/// Executes a PlanDefinition step by step.
///
/// Auto-map rule:
///   If a tool input declared as scalar in its schema receives a JsonArray from a $ref,
///   the executor fans out: calls the tool once per element &#x2192; collects results into an array.
///   Same for agent steps: if ExpectsArray on the step is not set and input is an array,
///   the executor calls the agent once per element.
///   The planner never needs foreach/batch � auto-map happens transparently.
/// </summary>
public sealed class PlanExecutor(
    IToolRegistry toolRegistry,
    IAgentStepRunner agentStepRunner,
    IExecutionLogger? executionLogger = null)
{
    private readonly IExecutionLogger _log = executionLogger ?? NullExecutionLogger.Instance;

    public async Task<ExecutionResult> ExecuteAsync(
        PlanDefinition plan,
        ExecutionStore store,
        CancellationToken cancellationToken = default)
    {
        var traces = new List<StepExecutionTrace>();
        var pending = plan.Steps.ToDictionary(s => s.Id, StringComparer.Ordinal);
        ResultEnvelope<JsonNode?>? lastEnvelope = null;

        while (pending.Count > 0)
        {
            var missingByStep = pending.Values.ToDictionary(s => s, s => GetMissingRefs(s, store));
            var ready = missingByStep
                .Where(e => e.Value.Count == 0)
                .Select(e => e.Key)
                .OrderBy(s => s.Id, StringComparer.Ordinal)
                .ToList();

            if (ready.Count == 0)
            {
                var detail = string.Join("; ", missingByStep
                    .OrderBy(e => e.Key.Id)
                    .Select(e => $"'{e.Key.Id}' missing [{string.Join(", ", e.Value)}]"));
                throw new InvalidOperationException($"Unresolved dependencies: {detail}");
            }

            foreach (var step in ready)
            {
                var (trace, envelope) = await ExecuteStepAsync(step, store, cancellationToken);
                traces.Add(trace);
                lastEnvelope = envelope;

                if (!trace.Success)
                    return new ExecutionResult { StepTraces = traces, LastEnvelope = lastEnvelope };

                if (trace.VerificationIssues.Count > 0)
                    return new ExecutionResult { StepTraces = traces, LastEnvelope = lastEnvelope };

                pending.Remove(step.Id);
            }
        }

        return new ExecutionResult { StepTraces = traces, LastEnvelope = lastEnvelope };
    }

    // -----------------------------------------------------------------------

    private async Task<(StepExecutionTrace trace, ResultEnvelope<JsonNode?>? envelope)> ExecuteStepAsync(
        PlanStep step, ExecutionStore store, CancellationToken ct)
    {
        var calls = new List<JsonObject>();
        var outputs = new List<JsonNode?>();
        ResultEnvelope<JsonNode?>? lastEnv = null;

        // Resolve $ref inputs > detect auto-map (array on scalar parameter)
        var (resolved, autoMapKey, fanOutArray) = ResolveInputs(step, store);

        bool isTool = !string.IsNullOrEmpty(step.Tool);

        _log.Log($"[exec] step:start id={step.Id} kind={(isTool ? "tool" : "llm")} name={(isTool ? step.Tool : step.Llm)} fanOut={fanOutArray?.Count.ToString() ?? "no"} resolvedInputs={SerializeNode(resolved)}");

        if (fanOutArray is not null)
        {
            // Fan-out: call once per array element, substitute scalar value each time
            for (var index = 0; index < fanOutArray.Count; index++)
            {
                var item = fanOutArray[index];
                var singleInput = SubstituteScalar(resolved, autoMapKey!, item);
                _log.Log($"[exec] call:start step={step.Id} callIndex={index} input={SerializeNode(singleInput)}");
                lastEnv = isTool
                    ? await RunToolAsync(step.Tool!, singleInput, calls, ct)
                    : await RunAgentAsync(step, singleInput, calls, ct);

                _log.Log($"[exec] call:end step={step.Id} callIndex={index} ok={lastEnv.Ok} output={SerializeNode(lastEnv.Data)} error={Shorten(lastEnv.Error?.Message, 240)}");

                if (!lastEnv.Ok) break;
                outputs.Add(lastEnv.Data?.DeepClone());
            }
        }
        else
        {
            _log.Log($"[exec] call:start step={step.Id} callIndex=0 input={SerializeNode(resolved)}");
            lastEnv = isTool
                ? await RunToolAsync(step.Tool!, resolved, calls, ct)
                : await RunAgentAsync(step, resolved, calls, ct);

            _log.Log($"[exec] call:end step={step.Id} callIndex=0 ok={lastEnv.Ok} output={SerializeNode(lastEnv.Data)} error={Shorten(lastEnv.Error?.Message, 240)}");

            if (lastEnv.Ok) outputs.Add(lastEnv.Data?.DeepClone());
        }

        // Write aggregated result to store under step.Id
        if (outputs.Count > 0)
        {
            store.Set(step.Id, fanOutArray is not null
                ? new JsonArray(outputs.Select(o => o?.DeepClone()).ToArray())
                : outputs[0]?.DeepClone());

            store.TryGet(step.Id, out var storedOutput);
            _log.Log($"[exec] step:stored id={step.Id} output={SerializeNode(storedOutput)}");
        }

        bool success = lastEnv?.Ok ?? false;
        List<StepVerificationIssue> verificationIssues = [];

        if (success)
        {
            store.TryGet(step.Id, out var storedOutput);
            verificationIssues = StepOutputVerifier.Verify(step, storedOutput);
            foreach (var issue in verificationIssues)
                _log.Log($"[verify] step={step.Id} code={issue.Code} message={issue.Message}");
        }

        _log.Log($"[exec] step:end id={step.Id} success={success} calls={calls.Count} error={Shorten(lastEnv?.Error?.Message, 240)}");

        var trace = new StepExecutionTrace
        {
            StepId = step.Id,
            Success = success,
            ErrorCode = success ? null : lastEnv?.Error?.Code,
            ErrorMessage = success ? null : lastEnv?.Error?.Message,
            ErrorDetails = success ? null : lastEnv?.Error?.Details,
            VerificationIssues = verificationIssues
        };
        foreach (var c in calls) trace.Calls.Add(c);

        return (trace, lastEnv);
    }

    // -----------------------------------------------------------------------

    private async Task<ResultEnvelope<JsonNode?>> RunToolAsync(
        string toolName, JsonObject input, List<JsonObject> calls, CancellationToken ct)
    {
        var tool = toolRegistry.GetRequired(toolName);
        var env = await tool.ExecuteAsync(input, ct);
        calls.Add(new JsonObject { ["tool"] = toolName, ["input"] = input.DeepClone(), ["ok"] = env.Ok, ["output"] = env.Data?.DeepClone() });
        return env;
    }

    private async Task<ResultEnvelope<JsonNode?>> RunAgentAsync(
        PlanStep step, JsonObject input, List<JsonObject> calls, CancellationToken ct)
    {
        var env = await agentStepRunner.ExecuteAsync(step, input, ct);
        calls.Add(new JsonObject { ["llm"] = step.Llm, ["ok"] = env.Ok, ["output"] = env.Data?.DeepClone() });
        return env;
    }

    // -----------------------------------------------------------------------
    // Input resolution + auto-map detection

    private (JsonObject resolved, string? autoMapKey, JsonArray? fanOutArray) ResolveInputs(
        PlanStep step, ExecutionStore store)
    {
        var resolved = new JsonObject();
        string? autoMapKey = null;
        JsonArray? fanOutArray = null;

        bool isTool = !string.IsNullOrEmpty(step.Tool);

        foreach (var kvp in step.In)
        {
            if (kvp.Value is JsonValue jv && jv.TryGetValue<string>(out var s) && s.StartsWith('$'))
            {
                var (sourceVal, isExplicit) = ParseStepRef(s[1..], step.Id, store);

                // Auto-fan-out only for plain $stepId refs, not $stepId[n] or $stepId.field
                // (explicit accessors mean the planner already selected a specific value).
                if (!isExplicit && sourceVal is JsonArray arr)
                {
                    // Tools: fan-out based on schema (scalar param + array input).
                    // LLM steps: fan-out only when the plan explicitly marks each:true.
                    bool expectsScalar = isTool
                        ? ToolExpectsScalar(step.Tool!, kvp.Key)
                        : step.Each;

                    if (expectsScalar && autoMapKey is null)
                    {
                        autoMapKey = kvp.Key;
                        fanOutArray = arr;
                        resolved[kvp.Key] = arr.DeepClone(); // placeholder, replaced per-call
                        continue;
                    }
                }

                resolved[kvp.Key] = sourceVal?.DeepClone();
            }
            else
            {
                resolved[kvp.Key] = kvp.Value?.DeepClone();
            }
        }

        return (resolved, autoMapKey, fanOutArray);
    }

    /// <summary>
    /// Resolves a step-reference expression (the part after '$') to a JsonNode.
    /// Supported forms:
    ///   stepId           – full output of a previous step
    ///   stepId[n]        – element n of an array output (0-based)
    ///   stepId.field     – named field of an object output
    ///   stepId[n].field  – field of the n-th array element
    /// Returns (value, isExplicit) where isExplicit=true when an index or field accessor
    /// is present, suppressing auto-fan-out in the caller.
    /// </summary>
    private static (JsonNode? value, bool isExplicit) ParseStepRef(
        string expr, string currentStepId, ExecutionStore store)
    {
        var stepId = expr;
        int? arrayIndex = null;
        string? fieldName = null;

        var bracketStart = expr.IndexOf('[');
        if (bracketStart >= 0)
        {
            stepId = expr[..bracketStart];
            var bracketEnd = expr.IndexOf(']', bracketStart);
            if (bracketEnd < 0)
                throw new InvalidOperationException(
                    $"Step '{currentStepId}': invalid ref '${expr}' — unclosed '['.");
            if (!int.TryParse(expr[(bracketStart + 1)..bracketEnd], out var idx))
                throw new InvalidOperationException(
                    $"Step '{currentStepId}': invalid array index in '${expr}'.");
            arrayIndex = idx;
            if (bracketEnd + 1 < expr.Length && expr[bracketEnd + 1] == '.')
                fieldName = expr[(bracketEnd + 2)..];
        }
        else
        {
            var dot = expr.IndexOf('.');
            if (dot >= 0)
            {
                stepId = expr[..dot];
                fieldName = expr[(dot + 1)..];
            }
        }

        if (!store.TryGet(stepId, out var node))
            throw new InvalidOperationException(
                $"Step '{currentStepId}': ref '${expr}' — step '{stepId}' not found in store.");

        bool isExplicit = arrayIndex.HasValue || fieldName != null;

        if (arrayIndex.HasValue)
        {
            if (node is not JsonArray arr)
                throw new InvalidOperationException(
                    $"Step '{currentStepId}': ref '${expr}' — step '{stepId}' output is not a JsonArray.");
            if (arrayIndex.Value >= arr.Count)
                throw new InvalidOperationException(
                    $"Step '{currentStepId}': ref '${expr}' — index {arrayIndex} out of range (count={arr.Count}).");
            node = arr[arrayIndex.Value];
        }

        if (fieldName != null)
        {
            if (node is not JsonObject obj)
                throw new InvalidOperationException(
                    $"Step '{currentStepId}': ref '${expr}' — cannot access field '{fieldName}' on non-object.");
            obj.TryGetPropertyValue(fieldName, out node);
        }

        return (node, isExplicit);
    }

    private bool ToolExpectsScalar(string toolName, string paramName)
    {
        var meta = toolRegistry.GetRequired(toolName).PlannerMetadata;
        if (!meta.InputSchema.TryGetPropertyValue("properties", out var props) || props is not JsonObject propsObj)
            return true; // assume scalar when no schema
        if (!propsObj.TryGetPropertyValue(paramName, out var pDef) || pDef is not JsonObject pDefObj)
            return true;
        if (!pDefObj.TryGetPropertyValue("type", out var typeProp))
            return true;
        return typeProp?.ToString() != "array";
    }

    private static JsonObject SubstituteScalar(JsonObject resolved, string key, JsonNode? value)
    {
        var copy = (JsonObject)resolved.DeepClone();
        copy[key] = value?.DeepClone();
        return copy;
    }

    private static string SerializeNode(JsonNode? node)
    {
        return node?.ToJsonString(new JsonSerializerOptions { WriteIndented = false }) ?? "null";
    }

    private static string Shorten(string? value, int maxLength)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "<none>";

        var normalized = value.ReplaceLineEndings(" ").Trim();
        return normalized.Length <= maxLength
            ? normalized
            : $"{normalized[..maxLength]}...";
    }

    // -----------------------------------------------------------------------

    private static List<string> GetMissingRefs(PlanStep step, ExecutionStore store)
    {
        var missing = new List<string>();
        foreach (var val in step.In.Values)
        {
            if (val is JsonValue jv && jv.TryGetValue<string>(out var s) && s.StartsWith('$'))
            {
                var expr = s[1..];
                // Extract the base step ID before any [n] or .field accessor
                var baseId = expr;
                var bracketPos = expr.IndexOf('[');
                var dotPos = expr.IndexOf('.');
                if (bracketPos >= 0) baseId = expr[..bracketPos];
                else if (dotPos >= 0) baseId = expr[..dotPos];

                if (!store.TryGet(baseId, out _)) missing.Add(expr);
            }
        }
        return missing;
    }
}
