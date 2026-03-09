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

        var verificationIssues = executionResult.StepTraces
            .Where(trace => trace.VerificationIssues.Count > 0)
            .SelectMany(trace => trace.VerificationIssues.Select(issue => $"{trace.StepId}:{issue.Code}"))
            .ToList();

        if (verificationIssues.Count > 0)
        {
            return new GoalVerdict
            {
                Action = GoalAction.Replan,
                Reason = "Execution completed, but one or more step outputs look incomplete.",
                Missing = verificationIssues
            };
        }

        var lastStepId = plan.Steps[^1].Id;
        if (!store.TryGet(lastStepId, out var finalNode) || finalNode is null)
        {
            return new GoalVerdict
            {
                Action = GoalAction.Replan,
                Reason = $"Store does not contain final result for step '{lastStepId}'.",
                Missing = [lastStepId]
            };
        }

        if (finalNode is JsonObject finalObj && finalObj.Count == 0)
        {
            return new GoalVerdict
            {
                Action = GoalAction.Replan,
                Reason = "Final output is an empty object.",
                Missing = ["final_content"]
            };
        }

        if (finalNode is JsonArray finalArray && finalArray.Count == 0)
        {
            return new GoalVerdict
            {
                Action = GoalAction.Replan,
                Reason = "Final output is an empty array.",
                Missing = ["final_content"]
            };
        }

        if (finalNode is JsonValue finalValue
            && finalValue.TryGetValue<string>(out var finalText)
            && string.IsNullOrWhiteSpace(finalText))
        {
            return new GoalVerdict
            {
                Action = GoalAction.Replan,
                Reason = "Final output is an empty string.",
                Missing = ["final_content"]
            };
        }

        if (askUserEnabled && finalNode is JsonObject askUserObject && askUserObject["needUserInput"]?.GetValue<bool>() == true)
        {
            return new GoalVerdict
            {
                Action = GoalAction.AskUser,
                Reason = "Need clarification from user.",
                UserQuestion = askUserObject["question"]?.GetValue<string>()
            };
        }

        return new GoalVerdict
        {
            Action = GoalAction.Done,
            Reason = "Goal achieved."
        };
    }
}
