using System.Text.Json;
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

    public async Task<ResultEnvelope<JsonElement?>> RunAsync(
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
                return ResultEnvelope<JsonElement?>.Success(plan.Steps[^1].Result?.Clone());

            if (verdict.Action == GoalAction.AskUser)
            {
                return ResultEnvelope<JsonElement?>.Failure(
                    "ask_user",
                    verdict.Reason,
                    JsonSerializer.SerializeToElement(new { question = verdict.UserQuestion }));
            }

            if (attempt == _maxAttempts || _replanner is null)
            {
                return ResultEnvelope<JsonElement?>.Failure(
                    result.LastEnvelope?.Error?.Code ?? "goal_not_achieved",
                    verdict.Reason,
                    JsonSerializer.SerializeToElement(new
                    {
                        missing = verdict.Missing,
                        attempt
                    }));
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

        return ResultEnvelope<JsonElement?>.Failure("goal_not_achieved", "Plan execution exceeded max attempts.");
    }

    private async Task<PlanDefinition> CreateReplanAsync(PlannerReplanRequest request, CancellationToken cancellationToken)
    {
        if (_replanner is null)
            return request.Plan;

        return await _replanner.ReplanAsync(request, cancellationToken);
    }
}
