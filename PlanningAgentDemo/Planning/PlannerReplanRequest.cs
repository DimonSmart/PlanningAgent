using System.Text.Json.Nodes;
using PlanningAgentDemo.Execution;
using PlanningAgentDemo.Verification;

namespace PlanningAgentDemo.Planning;

public sealed class PlannerReplanRequest
{
    public string UserQuery { get; init; } = string.Empty;

    public int AttemptNumber { get; init; }

    public PlanDefinition PreviousPlan { get; init; } = new();

    public ExecutionResult ExecutionResult { get; init; } = new();

    public GoalVerdict GoalVerdict { get; init; } = new();

    public JsonObject StoreSnapshot { get; init; } = [];
}