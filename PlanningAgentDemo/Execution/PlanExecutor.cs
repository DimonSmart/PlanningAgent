using System.Text.Json;
using System.Text.Json.Nodes;
using PlanningAgentDemo.Agents;
using PlanningAgentDemo.Common;
using PlanningAgentDemo.Planning;
using PlanningAgentDemo.Tools;
using PlanningAgentDemo.Verification;

namespace PlanningAgentDemo.Execution;

public sealed class PlanExecutor(
    IToolRegistry toolRegistry,
    IAgentStepRunner agentStepRunner,
    IExecutionLogger? executionLogger = null)
{
    private readonly IExecutionLogger _log = executionLogger ?? NullExecutionLogger.Instance;

    public async Task<ExecutionResult> ExecuteAsync(
        PlanDefinition plan,
        CancellationToken cancellationToken = default)
    {
        var traces = new List<StepExecutionTrace>();
        var stepMap = plan.Steps.ToDictionary(step => step.Id, StringComparer.Ordinal);
        ResultEnvelope<JsonNode?>? lastEnvelope = null;

        foreach (var step in plan.Steps)
        {
            if (IsReusable(step))
            {
                _log.Log($"[exec] step:reuse id={step.Id}");
                traces.Add(new StepExecutionTrace
                {
                    StepId = step.Id,
                    Success = true,
                    Reused = true
                });
                continue;
            }

            var missingRefs = GetMissingRefs(step, stepMap);
            if (missingRefs.Count > 0)
                throw new InvalidOperationException($"Step '{step.Id}' is not ready. Missing resolved refs: {string.Join(", ", missingRefs)}");

            PlanExecutionState.ResetStep(step);

            var (trace, envelope) = await ExecuteStepAsync(step, stepMap, cancellationToken);
            traces.Add(trace);
            lastEnvelope = envelope;

            if (!trace.Success)
                return new ExecutionResult { StepTraces = traces, LastEnvelope = lastEnvelope };
        }

        return new ExecutionResult { StepTraces = traces, LastEnvelope = lastEnvelope };
    }

    private async Task<(StepExecutionTrace trace, ResultEnvelope<JsonNode?> envelope)> ExecuteStepAsync(
        PlanStep step,
        IReadOnlyDictionary<string, PlanStep> stepMap,
        CancellationToken cancellationToken)
    {
        var calls = new List<JsonObject>();
        var outputs = new List<JsonNode?>();

        var (resolved, fanOutInputs) = ResolveInputs(step, stepMap);
        var isTool = !string.IsNullOrWhiteSpace(step.Tool);
        var fanOutCount = fanOutInputs?.Values.FirstOrDefault()?.Count ?? 0;

        _log.Log($"[exec] step:start id={step.Id} kind={(isTool ? "tool" : "llm")} name={(isTool ? step.Tool : step.Llm)} fanOut={(fanOutInputs is null ? "no" : fanOutCount.ToString())} resolvedInputs={SerializeNode(resolved)}");

        ResultEnvelope<JsonNode?> envelope;
        if (fanOutInputs is null)
        {
            _log.Log($"[exec] call:start step={step.Id} callIndex=0 input={SerializeNode(resolved)}");
            envelope = isTool
                ? await RunToolAsync(step.Tool!, resolved, calls, cancellationToken)
                : await RunAgentAsync(step, resolved, calls, cancellationToken);
            _log.Log($"[exec] call:end step={step.Id} callIndex=0 ok={envelope.Ok} output={SerializeNode(envelope.Data)} error={Shorten(envelope.Error?.Message, 240)} details={SerializeNode(envelope.Error?.Details)}");

            if (envelope.Ok)
                outputs.Add(envelope.Data?.DeepClone());
        }
        else
        {
            envelope = ResultEnvelope<JsonNode?>.Success(null);
            for (var callIndex = 0; callIndex < fanOutCount; callIndex++)
            {
                var singleInput = SubstituteScalars(resolved, fanOutInputs, callIndex);
                _log.Log($"[exec] call:start step={step.Id} callIndex={callIndex} input={SerializeNode(singleInput)}");
                envelope = isTool
                    ? await RunToolAsync(step.Tool!, singleInput, calls, cancellationToken)
                    : await RunAgentAsync(step, singleInput, calls, cancellationToken);
                _log.Log($"[exec] call:end step={step.Id} callIndex={callIndex} ok={envelope.Ok} output={SerializeNode(envelope.Data)} error={Shorten(envelope.Error?.Message, 240)} details={SerializeNode(envelope.Error?.Details)}");

                if (!envelope.Ok)
                    break;

                outputs.Add(envelope.Data?.DeepClone());
            }
        }

        if (outputs.Count > 0)
        {
            step.Result = fanOutInputs is null
                ? outputs[0]?.DeepClone()
                : new JsonArray(outputs.Select(output => output?.DeepClone()).ToArray());
        }

        if (!envelope.Ok)
        {
            step.Status = PlanStepStatuses.Fail;
            step.Error = CreatePlanStepError(envelope.Error);
            _log.Log($"[exec] step:end id={step.Id} success=False calls={calls.Count} error={Shorten(step.Error?.Message, 240)} details={SerializeNode(step.Error?.Details)}");
            return (CreateTrace(step, success: false, reused: false, calls, []), envelope);
        }

        var verificationIssues = StepOutputVerifier.Verify(step, step.Result);
        if (verificationIssues.Count > 0)
        {
            step.Status = PlanStepStatuses.Fail;
            step.Error = CreateVerificationError(step.Id, verificationIssues);
            foreach (var issue in verificationIssues)
                _log.Log($"[verify] step={step.Id} code={issue.Code} message={issue.Message}");

            envelope = ResultEnvelope<JsonNode?>.Failure(step.Error.Code, step.Error.Message, step.Error.Details?.DeepClone() as JsonObject);
            _log.Log($"[exec] step:end id={step.Id} success=False calls={calls.Count} error={Shorten(step.Error.Message, 240)} details={SerializeNode(step.Error.Details)}");
            return (CreateTrace(step, success: false, reused: false, calls, verificationIssues), envelope);
        }

        step.Status = PlanStepStatuses.Done;
        step.Error = null;
        _log.Log($"[exec] step:stored id={step.Id} output={SerializeNode(step.Result)}");
        _log.Log($"[exec] step:end id={step.Id} success=True calls={calls.Count} error=<none> details=null");
        return (CreateTrace(step, success: true, reused: false, calls, []), envelope);
    }

    private async Task<ResultEnvelope<JsonNode?>> RunToolAsync(
        string toolName,
        JsonObject input,
        List<JsonObject> calls,
        CancellationToken cancellationToken)
    {
        var tool = toolRegistry.GetRequired(toolName);
        var envelope = await tool.ExecuteAsync(input, cancellationToken);
        calls.Add(new JsonObject
        {
            ["tool"] = toolName,
            ["input"] = input.DeepClone(),
            ["ok"] = envelope.Ok,
            ["output"] = envelope.Data?.DeepClone(),
            ["error"] = SerializeError(envelope.Error)
        });
        return envelope;
    }

    private async Task<ResultEnvelope<JsonNode?>> RunAgentAsync(
        PlanStep step,
        JsonObject input,
        List<JsonObject> calls,
        CancellationToken cancellationToken)
    {
        var envelope = await agentStepRunner.ExecuteAsync(step, input, cancellationToken);
        calls.Add(new JsonObject
        {
            ["llm"] = step.Llm,
            ["ok"] = envelope.Ok,
            ["output"] = envelope.Data?.DeepClone(),
            ["error"] = SerializeError(envelope.Error)
        });
        return envelope;
    }

    private (JsonObject resolved, Dictionary<string, JsonArray>? fanOutInputs) ResolveInputs(
        PlanStep step,
        IReadOnlyDictionary<string, PlanStep> stepMap)
    {
        var resolved = new JsonObject();
        Dictionary<string, JsonArray>? fanOutInputs = null;
        var isTool = !string.IsNullOrWhiteSpace(step.Tool);

        foreach (var input in step.In)
        {
            if (input.Value is JsonValue value
                && value.TryGetValue<string>(out var text)
                && text.StartsWith("$", StringComparison.Ordinal))
            {
                var resolution = ParseStepRef(text[1..], step.Id, stepMap);
                if (!resolution.SuppressAutoMap && resolution.Value is JsonArray array)
                {
                    var expectsScalar = isTool
                        ? ToolExpectsScalar(step.Tool!, input.Key)
                        : step.Each;

                    if (expectsScalar)
                    {
                        fanOutInputs ??= new Dictionary<string, JsonArray>(StringComparer.Ordinal);
                        fanOutInputs[input.Key] = array;
                        resolved[input.Key] = array.DeepClone();
                        continue;
                    }
                }

                resolved[input.Key] = resolution.Value?.DeepClone();
                continue;
            }

            resolved[input.Key] = input.Value?.DeepClone();
        }

        if (fanOutInputs is not null)
            ValidateFanOutInputs(step, fanOutInputs);

        return (resolved, fanOutInputs);
    }

    private static bool IsReusable(PlanStep step) =>
        PlanExecutionState.IsDone(step)
        || string.Equals(step.Status, PlanStepStatuses.Skip, StringComparison.Ordinal);

    private sealed record RefResolution(JsonNode? Value, bool SuppressAutoMap);

    private static RefResolution ParseStepRef(
        string expression,
        string currentStepId,
        IReadOnlyDictionary<string, PlanStep> stepMap)
    {
        var stepId = expression;
        int? arrayIndex = null;
        var projectArray = false;
        string? fieldName = null;

        var bracketStart = expression.IndexOf('[');
        if (bracketStart >= 0)
        {
            stepId = expression[..bracketStart];
            var bracketEnd = expression.IndexOf(']', bracketStart);
            if (bracketEnd < 0)
                throw new InvalidOperationException($"Step '{currentStepId}': invalid ref '${expression}' - unclosed '['.");

            var indexToken = expression[(bracketStart + 1)..bracketEnd];
            if (indexToken.Length == 0)
            {
                projectArray = true;
            }
            else if (int.TryParse(indexToken, out var parsedIndex))
            {
                arrayIndex = parsedIndex;
            }
            else
            {
                throw new InvalidOperationException($"Step '{currentStepId}': invalid array index in '${expression}'.");
            }

            if (bracketEnd + 1 < expression.Length && expression[bracketEnd + 1] == '.')
                fieldName = expression[(bracketEnd + 2)..];
        }
        else
        {
            var dotIndex = expression.IndexOf('.');
            if (dotIndex >= 0)
            {
                stepId = expression[..dotIndex];
                fieldName = expression[(dotIndex + 1)..];
            }
        }

        if (!stepMap.TryGetValue(stepId, out var referencedStep))
            throw new InvalidOperationException($"Step '{currentStepId}': ref '${expression}' - step '{stepId}' not found.");

        if (!PlanExecutionState.IsDone(referencedStep) || referencedStep.Result is null)
            throw new InvalidOperationException($"Step '{currentStepId}': ref '${expression}' - step '{stepId}' has no completed result.");

        var node = referencedStep.Result;
        if (projectArray)
        {
            if (node is not JsonArray projectedArray)
                throw new InvalidOperationException($"Step '{currentStepId}': ref '${expression}' - step '{stepId}' output is not a JsonArray.");

            if (fieldName is null)
                return new RefResolution(projectedArray.DeepClone(), SuppressAutoMap: false);

            return new RefResolution(ProjectArrayField(projectedArray, fieldName, currentStepId, expression), SuppressAutoMap: false);
        }

        if (arrayIndex.HasValue)
        {
            if (node is not JsonArray indexedArray)
                throw new InvalidOperationException($"Step '{currentStepId}': ref '${expression}' - step '{stepId}' output is not a JsonArray.");
            if (arrayIndex.Value < 0 || arrayIndex.Value >= indexedArray.Count)
                throw new InvalidOperationException($"Step '{currentStepId}': ref '${expression}' - index {arrayIndex} out of range (count={indexedArray.Count}).");

            node = indexedArray[arrayIndex.Value];
        }

        if (fieldName is null)
            return new RefResolution(node?.DeepClone(), SuppressAutoMap: arrayIndex.HasValue);

        return node switch
        {
            JsonObject obj => new RefResolution(obj[fieldName]?.DeepClone(), SuppressAutoMap: arrayIndex.HasValue),
            JsonArray array => new RefResolution(ProjectArrayField(array, fieldName, currentStepId, expression), SuppressAutoMap: false),
            _ => throw new InvalidOperationException($"Step '{currentStepId}': ref '${expression}' - cannot access field '{fieldName}' on non-object.")
        };
    }

    private bool ToolExpectsScalar(string toolName, string paramName)
    {
        var metadata = toolRegistry.GetRequired(toolName).PlannerMetadata;
        if (!metadata.InputSchema.TryGetPropertyValue("properties", out var propertiesNode) || propertiesNode is not JsonObject properties)
            return true;
        if (!properties.TryGetPropertyValue(paramName, out var parameterNode) || parameterNode is not JsonObject parameter)
            return true;
        if (!parameter.TryGetPropertyValue("type", out var typeNode))
            return true;

        return !string.Equals(typeNode?.ToString(), "array", StringComparison.OrdinalIgnoreCase);
    }

    private static JsonObject SubstituteScalars(
        JsonObject resolved,
        IReadOnlyDictionary<string, JsonArray> fanOutInputs,
        int index)
    {
        var copy = (JsonObject)resolved.DeepClone();
        foreach (var fanOutInput in fanOutInputs)
            copy[fanOutInput.Key] = fanOutInput.Value[index]?.DeepClone();

        return copy;
    }

    private static void ValidateFanOutInputs(PlanStep step, IReadOnlyDictionary<string, JsonArray> fanOutInputs)
    {
        var expectedCount = fanOutInputs.First().Value.Count;
        foreach (var fanOutInput in fanOutInputs)
        {
            if (fanOutInput.Value.Count != expectedCount)
            {
                throw new InvalidOperationException(
                    $"Step '{step.Id}' resolves multiple array inputs with different lengths. Cannot zip fan-out '{fanOutInputs.First().Key}' ({expectedCount}) and '{fanOutInput.Key}' ({fanOutInput.Value.Count}).");
            }
        }
    }

    private static JsonArray ProjectArrayField(JsonArray array, string fieldName, string currentStepId, string expression)
    {
        var projected = new JsonArray();
        foreach (var item in array)
        {
            if (item is not JsonObject obj)
            {
                throw new InvalidOperationException(
                    $"Step '{currentStepId}': ref '${expression}' - cannot access field '{fieldName}' on non-object array item.");
            }

            projected.Add(obj[fieldName]?.DeepClone());
        }

        return projected;
    }

    private static List<string> GetMissingRefs(PlanStep step, IReadOnlyDictionary<string, PlanStep> stepMap)
    {
        var missing = new List<string>();
        foreach (var value in step.In.Values)
        {
            if (value is not JsonValue jsonValue
                || !jsonValue.TryGetValue<string>(out var text)
                || !text.StartsWith("$", StringComparison.Ordinal))
            {
                continue;
            }

            var stepId = ExtractBaseStepId(text[1..]);
            if (!stepMap.TryGetValue(stepId, out var dependency)
                || !PlanExecutionState.IsDone(dependency)
                || dependency.Result is null)
            {
                missing.Add(text);
            }
        }

        return missing;
    }

    private static string ExtractBaseStepId(string expression)
    {
        var bracketIndex = expression.IndexOf('[');
        var dotIndex = expression.IndexOf('.');
        if (bracketIndex >= 0 && dotIndex >= 0)
            return expression[..Math.Min(bracketIndex, dotIndex)];
        if (bracketIndex >= 0)
            return expression[..bracketIndex];
        if (dotIndex >= 0)
            return expression[..dotIndex];
        return expression;
    }

    private static StepExecutionTrace CreateTrace(
        PlanStep step,
        bool success,
        bool reused,
        IEnumerable<JsonObject> calls,
        IReadOnlyCollection<StepVerificationIssue> verificationIssues) => new()
    {
        StepId = step.Id,
        Success = success,
        Reused = reused,
        ErrorCode = success ? null : step.Error?.Code,
        ErrorMessage = success ? null : step.Error?.Message,
        ErrorDetails = success ? null : step.Error?.Details?.DeepClone() as JsonObject,
        Calls = calls.Select(call => (JsonObject)call.DeepClone()).ToList(),
        VerificationIssues = verificationIssues
            .Select(issue => new StepVerificationIssue
            {
                Code = issue.Code,
                Message = issue.Message
            })
            .ToList()
    };

    private static PlanStepError? CreatePlanStepError(ErrorInfo? error)
    {
        if (error is null)
            return null;

        return new PlanStepError
        {
            Code = error.Code,
            Message = error.Message,
            Details = error.Details?.DeepClone() as JsonObject
        };
    }

    private static PlanStepError CreateVerificationError(string stepId, IReadOnlyCollection<StepVerificationIssue> verificationIssues)
    {
        var issues = new JsonArray();
        foreach (var issue in verificationIssues)
        {
            issues.Add(new JsonObject
            {
                ["code"] = issue.Code,
                ["message"] = issue.Message
            });
        }

        return new PlanStepError
        {
            Code = "verification_failed",
            Message = $"Step '{stepId}' produced output that failed verification.",
            Details = new JsonObject
            {
                ["issues"] = issues
            }
        };
    }

    private static JsonObject? SerializeError(ErrorInfo? error)
    {
        if (error is null)
            return null;

        return new JsonObject
        {
            ["code"] = error.Code,
            ["message"] = error.Message,
            ["details"] = error.Details?.DeepClone()
        };
    }

    private static string SerializeNode(JsonNode? node) =>
        node?.ToJsonString(new JsonSerializerOptions { WriteIndented = false }) ?? "null";

    private static string Shorten(string? value, int maxLength)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "<none>";

        var normalized = value.ReplaceLineEndings(" ").Trim();
        return normalized.Length <= maxLength
            ? normalized
            : $"{normalized[..maxLength]}...";
    }
}
