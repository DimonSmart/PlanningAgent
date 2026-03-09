using System.Text.Json.Nodes;
using PlanningAgentDemo.Execution;
using PlanningAgentDemo.Planning;

namespace PlanningAgentDemo.Verification;

public static class StepOutputVerifier
{
    public static List<StepVerificationIssue> Verify(PlanStep step, JsonNode? output)
    {
        var issues = new List<StepVerificationIssue>();

        if (output is null)
        {
            issues.Add(new StepVerificationIssue
            {
                Code = "null_output",
                Message = $"Step '{step.Id}' completed successfully but produced null output."
            });
            return issues;
        }

        switch (output)
        {
            case JsonValue value when value.TryGetValue<string>(out var text) && string.IsNullOrWhiteSpace(text):
                issues.Add(new StepVerificationIssue
                {
                    Code = "empty_string_output",
                    Message = $"Step '{step.Id}' produced an empty string output."
                });
                return issues;

            case JsonObject jsonObject when jsonObject.Count == 0:
                issues.Add(new StepVerificationIssue
                {
                    Code = "empty_object_output",
                    Message = $"Step '{step.Id}' produced an empty object output."
                });
                return issues;

            case JsonArray jsonArray when jsonArray.Count == 0:
                issues.Add(new StepVerificationIssue
                {
                    Code = "empty_array_output",
                    Message = $"Step '{step.Id}' produced an empty array output."
                });
                return issues;
        }

        if (output is JsonArray array)
        {
            for (var index = 0; index < array.Count; index++)
            {
                if (!IsStructurallyEmpty(array[index]))
                    continue;

                issues.Add(new StepVerificationIssue
                {
                    Code = "structurally_empty_array_item",
                    Message = $"Step '{step.Id}' produced a structurally empty item at index {index}."
                });
            }

            return issues;
        }

        if (IsStructurallyEmpty(output))
        {
            issues.Add(new StepVerificationIssue
            {
                Code = "structurally_empty_output",
                Message = $"Step '{step.Id}' produced a structurally empty output."
            });
        }

        return issues;
    }

    private static bool IsStructurallyEmpty(JsonNode? node)
    {
        if (node is null)
            return true;

        if (node is JsonValue value)
        {
            if (value.TryGetValue<string>(out var text))
                return string.IsNullOrWhiteSpace(text);

            return false;
        }

        if (node is JsonObject jsonObject)
        {
            if (jsonObject.Count == 0)
                return true;

            return jsonObject.All(property => IsStructurallyEmpty(property.Value));
        }

        if (node is JsonArray jsonArray)
        {
            if (jsonArray.Count == 0)
                return true;

            return jsonArray.All(IsStructurallyEmpty);
        }

        return false;
    }
}