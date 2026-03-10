using System.Text.Json.Nodes;
using PlanningAgentDemo.Execution;
using PlanningAgentDemo.Planning;

namespace PlanningAgentDemo.Verification;

public sealed class GoalVerifier(bool askUserEnabled = false)
{
    public GoalVerdict Check(PlanDefinition plan, ExecutionResult executionResult)
    {
        var verificationIssues = plan.Steps
            .Where(step => string.Equals(step.Error?.Code, "verification_failed", StringComparison.Ordinal))
            .SelectMany(step => ExtractVerificationIssues(step.Id, step.Error?.Details))
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

        var failedSteps = plan.Steps.Where(PlanExecutionState.IsFailed).ToList();
        if (failedSteps.Count > 0)
        {
            return new GoalVerdict
            {
                Action = GoalAction.Replan,
                Reason = "Execution has failed steps.",
                Missing = failedSteps.Select(step => step.Id).ToList()
            };
        }

        var finalStep = plan.Steps[^1];
        if (!PlanExecutionState.IsDone(finalStep) || finalStep.Result is null)
        {
            return new GoalVerdict
            {
                Action = GoalAction.Replan,
                Reason = $"Plan does not contain a completed final result for step '{finalStep.Id}'.",
                Missing = [finalStep.Id]
            };
        }

        var finalNode = finalStep.Result;
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

    private static IReadOnlyCollection<string> ExtractVerificationIssues(string stepId, JsonObject? errorDetails)
    {
        if (errorDetails?["issues"] is not JsonArray issues)
            return [$"{stepId}:verification_failed"];

        return issues
            .Select(issue => issue?["code"]?.GetValue<string>())
            .Where(code => !string.IsNullOrWhiteSpace(code))
            .Select(code => $"{stepId}:{code}")
            .ToList();
    }
}
