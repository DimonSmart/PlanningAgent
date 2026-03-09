using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using PlanningAgentDemo.Agents;
using PlanningAgentDemo.Execution;
using PlanningAgentDemo.Tools;

namespace PlanningAgentDemo.Planning;

/// <summary>
/// Generates a PlanDefinition from a natural-language user query by asking the LLM.
/// The planner lists available workflow building blocks (tool steps) and LLM reasoning
/// steps, making it clear that only tool steps can access external systems.
/// </summary>
public sealed class LlmPlanner(
  ILlmClient llmClient,
  IToolRegistry toolRegistry,
  IExecutionLogger? executionLogger = null) : IPlanner
{
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };
  private readonly IExecutionLogger _log = executionLogger ?? NullExecutionLogger.Instance;

    public async Task<PlanDefinition> CreatePlanAsync(string userQuery, CancellationToken cancellationToken = default)
    {
        var tools = toolRegistry.ListPlannerMetadata();
        var systemPrompt = BuildSystemPrompt(tools);
    _log.Log($"[plan] create:start toolCount={tools.Count} query={Shorten(userQuery, 240)}");
        var raw = await llmClient.GenerateAsync(systemPrompt, userQuery, cancellationToken);
        var plan = ParsePlan(raw);
        PlanValidator.ValidateOrThrow(plan);
    _log.Log($"[plan] create:success steps={plan.Steps.Count} goal={Shorten(plan.Goal, 240)}");
    _log.Log($"[plan] create:json {JsonSerializer.Serialize(plan, new JsonSerializerOptions { WriteIndented = true })}");
        return plan;
    }

  private static string Shorten(string? value, int maxLength)
  {
    if (string.IsNullOrWhiteSpace(value))
      return "<empty>";

    var normalized = value.ReplaceLineEndings(" ").Trim();
    return normalized.Length <= maxLength
      ? normalized
      : $"{normalized[..maxLength]}...";
  }

    private static string BuildSystemPrompt(IReadOnlyCollection<ToolPlannerMetadata> tools)
    {
        var sb = new StringBuilder();
        sb.AppendLine("You are a planning agent. Given a user request, produce a JSON execution plan.");
        sb.AppendLine("A plan is an ordered list of workflow steps. There are exactly two kinds of steps:");
        sb.AppendLine();
        sb.AppendLine("## Kind 1 - Tool steps  (field: \"tool\": \"<name>\")");
        sb.AppendLine("Tool steps are the ONLY way to interact with external systems (web, APIs, databases).");
        sb.AppendLine("They fetch data, download pages, run searches. Without a tool step there is NO external data.");
        sb.AppendLine("Rule: if the task requires content from a URL, you MUST include a download tool step");
        sb.AppendLine("      BEFORE any LLM step that analyses that content.");
        sb.AppendLine();
        sb.AppendLine("Available tools:");
        foreach (var t in tools)
        {
            sb.AppendLine($"- name: {t.Name}");
            sb.AppendLine($"  description: {t.Description}");
            sb.AppendLine($"  inputSchema: {t.InputSchema.ToJsonString()}");
        }
        sb.AppendLine();
        sb.AppendLine("## Kind 2 - LLM steps  (field: \"llm\": \"<free label>\")");
        sb.AppendLine("LLM steps call the language model with prompts YOU define inline in the plan.");
        sb.AppendLine("CRITICAL: an LLM step has NO internet access, NO tools, NO external systems whatsoever.");
        sb.AppendLine("It can ONLY reason over data that earlier tool steps have already placed in the plan.");
        sb.AppendLine("Use LLM steps for: extraction, summarization, comparison, classification, synthesis.");
        sb.AppendLine("There is no registry of LLM step names - you pick the label and write both prompts.");
        sb.AppendLine();
        sb.AppendLine("## Fan-out rules");
        sb.AppendLine("Tool steps: the executor automatically fans out when a scalar input receives a JsonArray");
        sb.AppendLine("from a $stepRef (calls the tool once per element). No explicit loop step needed.");
        sb.AppendLine();
        sb.AppendLine("LLM steps - control fan-out with the \"each\" flag:");
        sb.AppendLine("  \"each\": true  - call LLM once per array element, collect results (map/extract pattern)");
        sb.AppendLine("  \"each\": false - pass the whole array to LLM in one call (reduce/compare pattern, default)");
        sb.AppendLine();
        sb.AppendLine("## Plan JSON schema");
        sb.AppendLine(@"{
  ""goal"": ""<what the plan achieves>"",
  ""steps"": [
    {
      ""id"": ""step1"",
      ""tool"": ""<toolName>"",
      ""in"": { ""param"": ""literal"" }
    },
    {
      ""id"": ""step2"",
      ""tool"": ""<toolName>"",
      ""in"": { ""param"": ""$step1"" }
    },
    {
      ""id"": ""step3"",
      ""llm"": ""<free label>"",
      ""systemPrompt"": ""<LLM system prompt>"",
      ""userPrompt"": ""<instruction per item>"",
      ""each"": true,
      ""in"": { ""item"": ""$step2"" },
      ""out"": ""json""
    },
    {
      ""id"": ""step4"",
      ""llm"": ""<free label>"",
      ""systemPrompt"": ""<LLM system prompt>"",
      ""userPrompt"": ""<instruction for whole batch>"",
      ""in"": { ""items"": ""$step3"" },
      ""out"": ""string""
    }
  ]
}");
        sb.AppendLine();
        sb.AppendLine("## Step references");
        sb.AppendLine("Use these forms as 'in' values to reference outputs of previous steps:");
        sb.AppendLine("  $stepId          - full output of a step (array triggers auto-fan-out for tools / 'each' for LLM)");
        sb.AppendLine("  $stepId[n]       - element n of an array output (0-based); no auto-fan-out");
        sb.AppendLine("  $stepId.field    - named field of an object output");
        sb.AppendLine("  $stepId[n].field - field of the n-th array element");
        sb.AppendLine("Prefer $stepId (full ref) + auto-fan-out over splitting into $stepId[0] / $stepId[1] per step.");
        sb.AppendLine();
        sb.AppendLine("Rules:");
        sb.AppendLine("- out: \"json\" (default) or \"string\".");
        sb.AppendLine("- NEVER use template placeholders ({var} or {{var}}) in systemPrompt or userPrompt.");
        sb.AppendLine("  The executor always appends an 'Input:' section with all resolved 'in' values.");
        sb.AppendLine("  Write prompts as plain instructions — do NOT reference input fields by name in the text.");
        sb.AppendLine("- When out is \"json\", the systemPrompt MUST state that only valid JSON should be returned.");
        sb.AppendLine("- Output ONLY the raw JSON plan. No markdown fences, no explanation.");
        return sb.ToString();
    }

    private static PlanDefinition ParsePlan(string raw)
    {
        var start = raw.IndexOf('{');
        var end = raw.LastIndexOf('}');
        if (start < 0 || end <= start)
            throw new InvalidOperationException($"LLM did not return a JSON object. Raw: {raw[..Math.Min(raw.Length, 300)]}");
        var json = raw[start..(end + 1)];
        try
        {
            return JsonSerializer.Deserialize<PlanDefinition>(json, JsonOptions)
                ?? throw new InvalidOperationException("Deserialized plan was null.");
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to parse plan: {ex.Message}\nRaw JSON: {json[..Math.Min(json.Length, 500)]}");
        }
    }
}
