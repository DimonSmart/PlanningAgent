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
    int maxAttempts = 3,
    IReplanner? replanner = null)
{
    private readonly GoalVerifier _goalVerifier = goalVerifier ?? new GoalVerifier();
    private readonly IExecutionLogger _log = executionLogger ?? NullExecutionLogger.Instance;
    private readonly int _maxAttempts = maxAttempts;
    private readonly IReplanner? _replanner = replanner;

    public async Task<ResultEnvelope<JsonNode?>> RunAsync(
        string userQuery,
        CancellationToken cancellationToken = default)
    {
        PlanDefinition? plan = null;
        PlannerReplanRequest? replanRequest = null;

        for (var attempt = 1; attempt <= _maxAttempts; attempt++)
        {
            _log.Log($"[orchestrator] attempt={attempt} phase={(plan is null ? "plan" : "replan")}");

            plan = plan is null
                ? await planner.CreatePlanAsync(userQuery, cancellationToken)
                : await CreateReplanAsync(replanRequest!, cancellationToken);

            var result = await executor.ExecuteAsync(plan, cancellationToken);
            var verdict = _goalVerifier.Check(plan, result);

            _log.Log($"[verify] goal:action={verdict.Action} reason={verdict.Reason}");

            if (verdict.Action == GoalAction.Done)
                return ResultEnvelope<JsonNode?>.Success(plan.Steps[^1].Result?.DeepClone());

            if (verdict.Action == GoalAction.AskUser)
            {
                return ResultEnvelope<JsonNode?>.Failure(
                    "ask_user",
                    verdict.Reason,
                    new JsonObject { ["question"] = verdict.UserQuestion });
            }

            if (attempt == _maxAttempts || _replanner is null)
            {
                return ResultEnvelope<JsonNode?>.Failure(
                    result.LastEnvelope?.Error?.Code ?? "goal_not_achieved",
                    verdict.Reason,
                    new JsonObject
                    {
                        ["missing"] = new JsonArray(verdict.Missing.Select(missingItem => JsonValue.Create(missingItem)).ToArray()),
                        ["attempt"] = attempt
                    });
            }

            replanRequest = new PlannerReplanRequest
            {
                UserQuery = userQuery,
                AttemptNumber = attempt,
                Plan = plan,
                ExecutionResult = result,
                GoalVerdict = verdict
            };
        }

        return ResultEnvelope<JsonNode?>.Failure("goal_not_achieved", "Plan execution exceeded max attempts.");
    }

    private async Task<PlanDefinition> CreateReplanAsync(PlannerReplanRequest request, CancellationToken cancellationToken)
    {
        if (_replanner is null)
            return request.Plan;

        return await _replanner.ReplanAsync(request, cancellationToken);
    }
}
