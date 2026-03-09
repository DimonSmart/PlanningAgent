using System.Text.Json.Nodes;
using PlanningAgentDemo.Execution;
using PlanningAgentDemo.Planning;

namespace PlanningAgentDemo.Verification;

public sealed class GoalVerifier(bool askUserEnabled = false)
{
    public GoalVerdict Check(PlanDefinition plan, ExecutionResult executionResult, ExecutionStore store)
    {
        if (executionResult.HasErrors)
        {
            return new GoalVerdict
            {
                Action = GoalAction.Replan,
                Reason = "Execution has failed steps.",
                Missing = ["successful_execution"]
            };
        }

        if (!store.TryGet("final", out var finalNode) || finalNode is null)
        {
            return new GoalVerdict
            {
                Action = GoalAction.Replan,
                Reason = "Store does not contain final result.",
                Missing = ["final"]
            };
        }

        if (finalNode is not JsonObject finalObj || finalObj.Count == 0)
        {
            return new GoalVerdict
            {
                Action = GoalAction.Replan,
                Reason = "Final output does not match minimal expected structure.",
                Missing = ["final_structure"]
            };
        }

        if (askUserEnabled && finalObj["needUserInput"]?.GetValue<bool>() == true)
        {
            return new GoalVerdict
            {
                Action = GoalAction.AskUser,
                Reason = "Need clarification from user.",
                UserQuestion = finalObj["question"]?.GetValue<string>()
            };
        }

        return new GoalVerdict
        {
            Action = GoalAction.Done,
            Reason = "Goal achieved."
        };
    }
}
