using System.Text.Json.Nodes;
using PlanningAgentDemo.Common;

namespace PlanningAgentDemo.Execution;

public sealed record StepExecutionTrace
{
    public string StepId { get; init; } = string.Empty;

    public bool Success { get; init; }

    public string? ErrorCode { get; init; }

    public string? ErrorMessage { get; init; }

    public List<JsonObject> Calls { get; init; } = new();
}

public sealed class ExecutionResult
{
    public List<StepExecutionTrace> StepTraces { get; init; } = new();

    public bool HasErrors => StepTraces.Any(x => !x.Success);

    public ResultEnvelope<JsonNode?>? LastEnvelope { get; init; }
}
