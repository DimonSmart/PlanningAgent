using System.Text.Json.Nodes;
using PlanningAgentDemo.Common;

namespace PlanningAgentDemo.Execution;

public sealed record StepExecutionTrace
{
    public string StepId { get; init; } = string.Empty;

    public bool Success { get; init; }

    public bool Reused { get; init; }

    public string? ErrorCode { get; init; }

    public string? ErrorMessage { get; init; }

    public JsonObject? ErrorDetails { get; init; }

    public List<JsonObject> Calls { get; init; } = new();

    public List<StepVerificationIssue> VerificationIssues { get; init; } = [];
}

public sealed record StepVerificationIssue
{
    public string Code { get; init; } = string.Empty;

    public string Message { get; init; } = string.Empty;
}

public sealed class ExecutionResult
{
    public List<StepExecutionTrace> StepTraces { get; init; } = new();

    public bool HasErrors => StepTraces.Any(x => !x.Success);

    public bool HasVerificationIssues => StepTraces.Any(x => x.VerificationIssues.Count > 0);

    public ResultEnvelope<JsonNode?>? LastEnvelope { get; init; }
}
