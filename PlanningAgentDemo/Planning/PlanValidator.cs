using PlanningAgentDemo.Execution;
using System.Text.RegularExpressions;
using System.Text.Json.Nodes;

namespace PlanningAgentDemo.Planning;

public static class PlanValidator
{
    public static void ValidateOrThrow(PlanDefinition plan)
    {
        if (string.IsNullOrWhiteSpace(plan.Goal)) throw new InvalidOperationException("Plan.goal is required.");
        if (plan.Steps.Count == 0) throw new InvalidOperationException("Plan.steps must contain at least one step.");
        
        foreach (var step in plan.Steps)
        {
            if (string.IsNullOrWhiteSpace(step.Id)) throw new InvalidOperationException("Each step must have an id.");
            if (string.IsNullOrWhiteSpace(step.Tool) && string.IsNullOrWhiteSpace(step.Llm)) throw new InvalidOperationException("Each step must have either 'tool' or 'llm'.");
        }
    }
}
