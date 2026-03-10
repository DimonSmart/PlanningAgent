using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using PlanningAgentDemo.Execution;
using PlanningAgentDemo.Tools;

namespace PlanningAgentDemo.Planning;

/// <summary>
/// Generates a PlanDefinition from a natural-language user query by asking the LLM.
/// The planner lists available workflow building blocks (tool steps) and LLM reasoning
/// steps, making it clear that only tool steps can access external systems.
/// </summary>
public sealed class LlmPlanner(
    IChatClient chatClient,
    IToolRegistry toolRegistry,
    IExecutionLogger? executionLogger = null) : IPlanner
{
    private const int MaxPlanningAttempts = 5;
    private readonly IExecutionLogger _log = executionLogger ?? NullExecutionLogger.Instance;
    private static readonly HashSet<string> ReservedStepProperties = new(StringComparer.OrdinalIgnoreCase)
    {
        "id",
        "tool",
        "llm",
        "agent",
        "systemPrompt",
        "system",
        "systemInstruction",
        "systemMessage",
        "userPrompt",
        "prompt",
        "instruction",
        "task",
        "message",
        "in",
        "input",
        "inputs",
        "args",
        "params",
        "out",
        "output",
        "outputType",
        "responseFormat",
        "each",
        "s",
        "res",
        "err"
    };

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        TypeInfoResolver = new DefaultJsonTypeInfoResolver()
    };

    public async Task<PlanDefinition> CreatePlanAsync(string userQuery, CancellationToken cancellationToken = default)
    {
        _log.Log($"[plan] create:start toolCount={toolRegistry.ListPlannerMetadata().Count} query={Shorten(userQuery, 240)}");
        return await GeneratePlanCoreAsync(BuildPlanningUserPrompt(userQuery), userQuery, cancellationToken);
    }

    private async Task<PlanDefinition> GeneratePlanCoreAsync(
        string userPrompt,
        string defaultGoal,
        CancellationToken cancellationToken)
    {
        var tools = toolRegistry.ListPlannerMetadata();
        var systemPrompt = BuildSystemPrompt(tools);
        var agent = new ChatClientAgent(chatClient, systemPrompt, "planner", null, null, null, null);
        var planningPrompt = userPrompt;
        Exception? lastError = null;

        for (var attempt = 1; attempt <= MaxPlanningAttempts; attempt++)
        {
            var response = await agent.RunAsync<JsonNode?>(planningPrompt, null, JsonOptions, null, cancellationToken);

            try
            {
                var plan = DeserializePlan(response.Result?.DeepClone(), defaultGoal, tools);
                PlanValidator.ValidateOrThrow(plan, tools);

                _log.Log($"[plan] create:success steps={plan.Steps.Count} goal={Shorten(plan.Goal, 240)}");
                _log.Log($"[plan] create:json {JsonSerializer.Serialize(plan, new JsonSerializerOptions { WriteIndented = true })}");
                return plan;
            }
            catch (Exception ex) when (attempt < MaxPlanningAttempts)
            {
                lastError = ex;
                var invalidPlanJson = response.Result?.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) ?? "null";
                _log.Log($"[plan] create:retry attempt={attempt} error={Shorten(ex.Message, 240)}");
                planningPrompt = BuildRepairPrompt(userPrompt, invalidPlanJson, ex.Message);
            }
            catch (Exception ex)
            {
                lastError = ex;
                break;
            }
        }

        throw new InvalidOperationException(
            $"Planner could not produce a valid plan after {MaxPlanningAttempts} attempts. Last error: {lastError?.Message}",
            lastError);
    }

    private static PlanDefinition DeserializePlan(
        JsonNode? planNode,
        string defaultGoal,
        IReadOnlyCollection<ToolPlannerMetadata> tools)
    {
        if (planNode is not JsonObject root)
            throw new InvalidOperationException("Planner did not return a JSON object.");

        root = UnwrapPlanRoot(root);
        NormalizePlanObject(root, defaultGoal, tools);

        return JsonSerializer.Deserialize<PlanDefinition>(root, JsonOptions)
            ?? throw new InvalidOperationException("Planner returned an empty plan payload.");
    }

    private static JsonObject UnwrapPlanRoot(JsonObject root)
    {
        if (HasPlanShape(root))
            return root;

        foreach (var key in new[] { "plan", "result", "response", "data" })
        {
            if (root[key] is JsonObject nestedRoot && HasPlanShape(nestedRoot))
                return nestedRoot;
        }

        foreach (var property in root)
        {
            if (property.Value is JsonObject nestedRoot && HasPlanShape(nestedRoot))
                return nestedRoot;
        }

        return root;
    }

    private static bool HasPlanShape(JsonObject node) =>
        node["goal"] is not null || node["steps"] is JsonArray;

    private static void NormalizePlanObject(
        JsonObject root,
        string defaultGoal,
        IReadOnlyCollection<ToolPlannerMetadata> tools)
    {
        var toolMap = tools.ToDictionary(tool => tool.Name, StringComparer.OrdinalIgnoreCase);

        var goal = root["goal"]?.GetValue<string>()?.Trim();
        root["goal"] = string.IsNullOrWhiteSpace(goal) ? defaultGoal : goal;

        if (root["steps"] is not JsonArray steps)
            return;

        var usedGeneratedIds = new HashSet<string>(StringComparer.Ordinal);
        for (var stepIndex = 0; stepIndex < steps.Count; stepIndex++)
        {
            if (steps[stepIndex] is not JsonObject step)
                continue;

            NormalizeStepAliases(step, stepIndex, usedGeneratedIds);
            TrimStringProperty(step, "id");
            TrimStringProperty(step, "tool");
            TrimStringProperty(step, "llm");
            TrimStringProperty(step, "systemPrompt");
            TrimStringProperty(step, "userPrompt");
            TrimStringProperty(step, "out");

            CanonicalizeToolName(step, toolMap);
            NormalizeStepInputs(step, toolMap);
            NormalizeLlmShape(step);
            NormalizeExecutionState(step);
        }
    }

    private static void NormalizeStepAliases(JsonObject step, int stepIndex, ISet<string> usedGeneratedIds)
    {
        CopyStringAlias(step, "id", "stepId", "name");
        CopyStringAlias(step, "llm", "agent", "agentName");
        CopyStringAlias(step, "systemPrompt", "system", "systemInstruction", "systemMessage");
        CopyStringAlias(step, "userPrompt", "prompt", "instruction", "task", "message");
        CopyStringAlias(step, "out", "output", "outputType", "responseFormat");

        if (step["llm"] is null && step["agent"] is JsonValue agentFlag && agentFlag.TryGetValue<bool>(out var isAgent) && isAgent)
            step["llm"] = $"llm_step_{stepIndex + 1}";

        if (HasMeaningfulString(step["id"]))
        {
            usedGeneratedIds.Add(step["id"]!.GetValue<string>().Trim());
            return;
        }

        if (step["id"] is null || string.IsNullOrWhiteSpace(step["id"]?.GetValue<string>()))
            step["id"] = BuildGeneratedStepId(step, stepIndex, usedGeneratedIds);
    }

    private static void TrimStringProperty(JsonObject obj, string propertyName)
    {
        if (obj[propertyName] is JsonValue value && value.TryGetValue<string>(out var text))
            obj[propertyName] = text.Trim();
    }

    private static void CanonicalizeToolName(
        JsonObject step,
        IReadOnlyDictionary<string, ToolPlannerMetadata> toolMap)
    {
        var toolName = step["tool"]?.GetValue<string>();
        if (string.IsNullOrWhiteSpace(toolName))
            return;

        if (toolMap.TryGetValue(toolName, out var metadata))
            step["tool"] = metadata.Name;
    }

    private static void NormalizeStepInputs(
        JsonObject step,
        IReadOnlyDictionary<string, ToolPlannerMetadata> toolMap)
    {
        var inputNode = ResolveInputNode(step);
        if (inputNode is not JsonObject inputObject)
        {
            var normalizedInputs = new JsonObject();
            var originalInput = inputNode?.DeepClone();
            if (originalInput is not null)
                normalizedInputs[ResolveDefaultInputKey(step, toolMap)] = NormalizeInputNode(originalInput);

            step["in"] = normalizedInputs;
            return;
        }

        foreach (var input in inputObject.ToList())
            inputObject[input.Key] = NormalizeInputNode(input.Value);

        step["in"] = inputObject;
    }

    private static JsonNode? NormalizeInputNode(JsonNode? node)
    {
        if (node is JsonValue value && value.TryGetValue<string>(out var text))
            return JsonValue.Create(text.Trim());

        return node?.DeepClone();
    }

    private static void NormalizeExecutionState(JsonObject step)
    {
        var status = step["s"]?.GetValue<string>()?.Trim();
        step["s"] = string.IsNullOrWhiteSpace(status) ? PlanStepStatuses.Todo : status;
        step["res"] ??= null;
        step["err"] ??= null;
    }

    private static void NormalizeLlmShape(JsonObject step)
    {
        var llmName = step["llm"]?.GetValue<string>()?.Trim();
        if (string.IsNullOrWhiteSpace(llmName))
            return;

        if (string.IsNullOrWhiteSpace(step["out"]?.GetValue<string>()))
            step["out"] = "json";
    }

    private static string ResolveDefaultInputKey(
        JsonObject step,
        IReadOnlyDictionary<string, ToolPlannerMetadata> toolMap)
    {
        var toolName = step["tool"]?.GetValue<string>();
        if (!string.IsNullOrWhiteSpace(toolName) && toolMap.TryGetValue(toolName, out var tool))
            return ResolveToolInputKey(tool.InputSchema);

        return "input";
    }

    private static JsonNode? ResolveInputNode(JsonObject step)
    {
        foreach (var propertyName in new[] { "in", "input", "inputs", "args", "params" })
        {
            if (step[propertyName] is not null)
                return step[propertyName];
        }

        var inferredInputs = new JsonObject();
        foreach (var property in step)
        {
            if (ReservedStepProperties.Contains(property.Key))
                continue;

            inferredInputs[property.Key] = property.Value?.DeepClone();
        }

        return inferredInputs.Count > 0 ? inferredInputs : null;
    }

    private static string ResolveToolInputKey(JsonObject inputSchema)
    {
        if (inputSchema["required"] is JsonArray requiredProperties)
        {
            foreach (var requiredProperty in requiredProperties)
            {
                if (requiredProperty is JsonValue value && value.TryGetValue<string>(out var propertyName))
                    return propertyName;
            }
        }

        if (inputSchema["properties"] is JsonObject properties)
        {
            foreach (var property in properties)
                return property.Key;
        }

        return "input";
    }

    private static string BuildSystemPrompt(IReadOnlyCollection<ToolPlannerMetadata> tools)
    {
        var sb = new StringBuilder();
        sb.AppendLine("You are a planning agent. Given a user request, produce an execution plan.");
        sb.AppendLine("Return a COMPLETE and VALID plan object on the first try.");
        sb.AppendLine("A plan is an ordered list of workflow steps. There are exactly two kinds of steps:");
        sb.AppendLine();
        sb.AppendLine("1. Tool steps: use field \"tool\": \"<name>\".");
        sb.AppendLine("   Tool steps are the ONLY way to access external data.");
        sb.AppendLine("2. LLM steps: use field \"llm\": \"<free label>\".");
        sb.AppendLine("   LLM steps have NO tool access and NO internet access.");
        sb.AppendLine("   They can only reason over outputs produced by earlier tool steps.");
        sb.AppendLine();
        sb.AppendLine("Required JSON shape:");
        sb.AppendLine("{");
        sb.AppendLine("  \"goal\": \"string\",");
        sb.AppendLine("  \"steps\": [");
        sb.AppendLine("    {");
        sb.AppendLine("      \"id\": \"string\",");
        sb.AppendLine("      \"tool\": \"tool-name\",");
        sb.AppendLine("      \"in\": { \"param\": \"literal or $ref\" },");
        sb.AppendLine("      \"s\": \"todo\",");
        sb.AppendLine("      \"res\": null,");
        sb.AppendLine("      \"err\": null");
        sb.AppendLine("    },");
        sb.AppendLine("    {");
        sb.AppendLine("      \"id\": \"string\",");
        sb.AppendLine("      \"llm\": \"step-label\",");
        sb.AppendLine("      \"systemPrompt\": \"string\",");
        sb.AppendLine("      \"userPrompt\": \"string\",");
        sb.AppendLine("      \"in\": { \"param\": \"literal or $ref\" },");
        sb.AppendLine("      \"out\": \"json|string\",");
        sb.AppendLine("      \"each\": true|false,");
        sb.AppendLine("      \"s\": \"todo\",");
        sb.AppendLine("      \"res\": null,");
        sb.AppendLine("      \"err\": null");
        sb.AppendLine("    }");
        sb.AppendLine("  ]");
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("General planning rules:");
        sb.AppendLine("- Prefer the SHORTEST plan that can succeed.");
        sb.AppendLine("- When the user asks to compare, rank, summarize, or compute over a small number of discovered items, prefer this shape:");
        sb.AppendLine("  search candidate pages -> download candidate pages -> extract structured facts from each page -> synthesize final answer.");
        sb.AppendLine("- For final comparison or recommendation steps, instruct the LLM to explicitly mention the compared item names and the reason for the choice.");
        sb.AppendLine("- Avoid a second search unless the first downloaded pages clearly cannot contain the requested facts.");
        sb.AppendLine("- If the user asks for two items, prefer to search for two candidate pages or a very small over-fetch, not many pages.");
        sb.AppendLine("- If a tool returns array data and the next tool expects a scalar parameter, pass the full array ref and let the executor fan out.");
        sb.AppendLine("- For extraction steps, if the source does not contain the requested entity or the critical facts are absent, rely on the structured execution error contract instead of guessing.");
        sb.AppendLine();
        sb.AppendLine("Available tools:");
        foreach (var tool in tools)
        {
            sb.AppendLine($"- name: {tool.Name}");
            sb.AppendLine($"  description: {tool.Description}");
            sb.AppendLine($"  inputSchema: {tool.InputSchema.ToJsonString()}");
            sb.AppendLine($"  outputSchema: {tool.OutputSchema.ToJsonString()}");
        }

        sb.AppendLine();
        sb.AppendLine("Fan-out rules:");
        sb.AppendLine("- Tool steps: the executor automatically fans out when a scalar input receives an array.");
        sb.AppendLine("- LLM steps: set \"each\": true to call the LLM once per array element. Omit it or set false to pass the whole array in one call.");
        sb.AppendLine();
        sb.AppendLine("Reference syntax allowed in step inputs:");
        sb.AppendLine("- $stepId");
        sb.AppendLine("- $stepId.field");
        sb.AppendLine("- $stepId[]");
        sb.AppendLine("- $stepId[].field");
        sb.AppendLine("- $stepId[n]");
        sb.AppendLine("- $stepId[n].field");
        sb.AppendLine("Use $stepId.field only when the referenced step output is an object.");
        sb.AppendLine("Use $stepId[].field when the referenced step output is an array of objects.");
        sb.AppendLine();
        sb.AppendLine("LLM step rules:");
        sb.AppendLine("- LLM steps MUST provide both systemPrompt and userPrompt.");
        sb.AppendLine("- The executor appends an Input section with all resolved 'in' values.");
        sb.AppendLine("- Do NOT use template placeholders such as {var} or {{var}} in prompts.");
        sb.AppendLine("- out is \"json\" or \"string\". Prefer \"json\" unless the final answer is intentionally plain text.");
        sb.AppendLine("- When out is \"json\", the systemPrompt MUST explicitly require valid JSON only.");
        sb.AppendLine("- Use the exact field name \"userPrompt\". Do not use aliases like \"prompt\" or \"instruction\".");
        sb.AppendLine("- Put step parameters under \"in\". Do not place tool args as top-level step properties.");
        sb.AppendLine("- Every step must include id, in, s, res, and err.");
        sb.AppendLine();
        sb.AppendLine("Return only the plan object. No markdown fences. No prose outside the JSON.");
        return sb.ToString();
    }

    private static string BuildPlanningUserPrompt(string userQuery) => userQuery;

    private static string BuildRepairPrompt(string originalUserPrompt, string invalidPlanJson, string errorMessage) =>
        $"{originalUserPrompt}\n\nYour previous plan was invalid.\nValidation error: {errorMessage}\nInvalid plan:\n{invalidPlanJson}\n\nReturn a corrected FULL plan as JSON only.\nNon-negotiable requirements:\n- Each step must include id, in, s, res, and err.\n- A step must have exactly one of tool or llm.\n- Each llm step must include systemPrompt, userPrompt, and out.\n- Use the exact field name userPrompt, not prompt.\n- Put all step inputs under in.\n- Use only refs to earlier steps.\nDo not repeat the same mistake.";

    private static void CopyStringAlias(JsonObject step, string targetPropertyName, params string[] aliasPropertyNames)
    {
        if (HasMeaningfulString(step[targetPropertyName]))
            return;

        foreach (var aliasPropertyName in aliasPropertyNames)
        {
            if (!HasMeaningfulString(step[aliasPropertyName]))
                continue;

            step[targetPropertyName] = step[aliasPropertyName]!.DeepClone();
            return;
        }
    }

    private static bool HasMeaningfulString(JsonNode? node) =>
        node is JsonValue value
        && value.TryGetValue<string>(out var text)
        && !string.IsNullOrWhiteSpace(text);

    private static string BuildGeneratedStepId(JsonObject step, int stepIndex, ISet<string> usedGeneratedIds)
    {
        var rawBase = step["tool"]?.GetValue<string>()
            ?? step["llm"]?.GetValue<string>()
            ?? "step";
        var baseId = Slugify(rawBase);
        if (string.IsNullOrWhiteSpace(baseId))
            baseId = "step";

        var candidate = baseId;
        var suffix = 2;
        while (!usedGeneratedIds.Add(candidate))
            candidate = $"{baseId}{suffix++}";

        return candidate;
    }

    private static string Slugify(string value)
    {
        var sb = new StringBuilder(value.Length);
        var previousWasSeparator = false;

        foreach (var ch in value.Trim())
        {
            if (char.IsLetterOrDigit(ch))
            {
                sb.Append(char.ToLowerInvariant(ch));
                previousWasSeparator = false;
                continue;
            }

            if (previousWasSeparator)
                continue;

            sb.Append('_');
            previousWasSeparator = true;
        }

        return sb.ToString().Trim('_');
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
}
