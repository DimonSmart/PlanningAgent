using System.Text.Json.Nodes;
using PlanningAgentDemo.Common;
using PlanningAgentDemo.Execution;
using PlanningAgentDemo.Planning;

namespace PlanningAgentDemo.Orchestration;

public sealed class PlanningOrchestrator(IPlanner planner, PlanExecutor executor)
{
    public async Task<ResultEnvelope<JsonNode?>> RunAsync(
        string userQuery, CancellationToken cancellationToken = default)
    {
        var store = new ExecutionStore();
        var plan = await planner.CreatePlanAsync(userQuery, cancellationToken);
        var result = await executor.ExecuteAsync(plan, store, cancellationToken);

        if (result.HasErrors)
            return ResultEnvelope<JsonNode?>.Failure(
                result.LastEnvelope?.Error?.Code ?? "execution_failed",
                result.LastEnvelope?.Error?.Message ?? "Plan execution failed.");

        // Return the last step output as the answer
        var lastStepId = plan.Steps[^1].Id;
        store.TryGet(lastStepId, out var final);
        return ResultEnvelope<JsonNode?>.Success(final?.DeepClone());
    }
}
