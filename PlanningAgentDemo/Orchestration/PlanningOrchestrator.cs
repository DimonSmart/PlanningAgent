using System.Text.Json.Nodes;
using PlanningAgentDemo.Common;
using PlanningAgentDemo.Execution;
using PlanningAgentDemo.Planning;
using PlanningAgentDemo.Verification;

namespace PlanningAgentDemo.Orchestration;

public sealed class PlanningOrchestrator(
    IPlanner planner,
    PlanExecutor executor,
    GoalVerifier? goalVerifier = null,
    IExecutionLogger? executionLogger = null,
    int maxAttempts = 3)
{
    private readonly GoalVerifier _goalVerifier = goalVerifier ?? new GoalVerifier();
    private readonly IExecutionLogger _log = executionLogger ?? NullExecutionLogger.Instance;
    private readonly int _maxAttempts = maxAttempts;

    public async Task<ResultEnvelope<JsonNode?>> RunAsync(
        string userQuery, CancellationToken cancellationToken = default)
    {
        PlannerReplanRequest? replanRequest = null;

        for (var attempt = 1; attempt <= _maxAttempts; attempt++)
        {
            _log.Log($"[orchestrator] attempt={attempt} phase={(replanRequest is null ? "plan" : "replan")}");

            var plan = replanRequest is null
                ? await planner.CreatePlanAsync(userQuery, cancellationToken)
                : await CreateReplanAsync(replanRequest, cancellationToken);

            var store = new ExecutionStore();
            var result = await executor.ExecuteAsync(plan, store, cancellationToken);
            var verdict = _goalVerifier.Check(plan, result, store);

            _log.Log($"[verify] goal:action={verdict.Action} reason={verdict.Reason}");

            if (verdict.Action == GoalAction.Done)
            {
                var lastStepId = plan.Steps[^1].Id;
                store.TryGet(lastStepId, out var final);
                return ResultEnvelope<JsonNode?>.Success(final?.DeepClone());
            }

            if (verdict.Action == GoalAction.AskUser)
                return ResultEnvelope<JsonNode?>.Failure(
                    "ask_user",
                    verdict.Reason,
                    new JsonObject { ["question"] = verdict.UserQuestion });

            if (attempt == _maxAttempts || planner is not IReplanCapablePlanner)
                return ResultEnvelope<JsonNode?>.Failure(
                    result.LastEnvelope?.Error?.Code ?? "goal_not_achieved",
                    verdict.Reason,
                    new JsonObject
                    {
                        ["missing"] = new JsonArray(verdict.Missing.Select(missingItem => JsonValue.Create(missingItem)).ToArray()),
                        ["attempt"] = attempt
                    });

            replanRequest = new PlannerReplanRequest
            {
                UserQuery = userQuery,
                AttemptNumber = attempt,
                PreviousPlan = plan,
                ExecutionResult = result,
                GoalVerdict = verdict,
                StoreSnapshot = store.CreateSnapshot()
            };
        }

        return ResultEnvelope<JsonNode?>.Failure("goal_not_achieved", "Plan execution exceeded max attempts.");
    }

    private async Task<PlanDefinition> CreateReplanAsync(PlannerReplanRequest request, CancellationToken cancellationToken)
    {
        if (planner is not IReplanCapablePlanner replanCapablePlanner)
            return await planner.CreatePlanAsync(request.UserQuery, cancellationToken);

        return await replanCapablePlanner.ReplanAsync(request, cancellationToken);
    }
}
