using System.Text.Json;
using System.Text.Json.Nodes;
using PlanningAgentDemo.Agents;
using PlanningAgentDemo.Common;
using PlanningAgentDemo.Execution;
using PlanningAgentDemo.Planning;
using PlanningAgentDemo.Tools;
using Xunit;
using Xunit.Abstractions;

namespace PlanningAgentDemo.Tests;

// --- Test infrastructure -----------------------------------------------------

public class TestLogger(ITestOutputHelper out_) : IExecutionLogger
{
    public void Log(string message) => out_.WriteLine(message);
}

/// <summary>Returns a hard-coded list of two URLs (simulates a web search).</summary>
public class MockSearchTool : ITool
{
    public string Name => "search";
    public ToolPlannerMetadata PlannerMetadata => new(
        "search",
        "Search the web. Returns an array of URLs matching the query.",
        JsonNode.Parse(@"{""type"":""object"",""properties"":{""query"":{""type"":""string""},""limit"":{""type"":""number""}},""required"":[""query""]}")!.AsObject(),
        new JsonObject(), [], []);

    public Task<ResultEnvelope<JsonNode?>> ExecuteAsync(JsonObject input, CancellationToken ct = default) =>
        Task.FromResult(ResultEnvelope<JsonNode?>.Success(new JsonArray(
            JsonValue.Create("https://example.com/item-a"),
            JsonValue.Create("https://example.com/item-b"))));
}

/// <summary>Returns fake page content for a URL (simulates an HTTP download).</summary>
public class MockDownloadTool : ITool
{
    public string Name => "download";
    public ToolPlannerMetadata PlannerMetadata => new(
        "download",
        "Download a web page. Takes a single URL and returns {title, body}.",
        JsonNode.Parse(@"{""type"":""object"",""properties"":{""url"":{""type"":""string""}},""required"":[""url""]}")!.AsObject(),
        new JsonObject(), [], []);

    public Task<ResultEnvelope<JsonNode?>> ExecuteAsync(JsonObject input, CancellationToken ct = default)
    {
        var url = input["url"]?.GetValue<string>() ?? "";
        return url.Contains("item-a")
            ? Task.FromResult(ResultEnvelope<JsonNode?>.Success(new JsonObject
                { ["url"] = url, ["title"] = "Item A", ["body"] = "Item A weighs 1.2 kg and runs for 20 min." }))
            : Task.FromResult(ResultEnvelope<JsonNode?>.Success(new JsonObject
                { ["url"] = url, ["title"] = "Item B", ["body"] = "Item B weighs 3.5 kg and runs for 45 min." }));
    }
}

// --- Tests --------------------------------------------------------------------

public class PipelineTests(ITestOutputHelper output)
{
    private static IToolRegistry BuildTools() =>
        new ToolRegistry(new ITool[] { new MockSearchTool(), new MockDownloadTool() });

    private static ILlmClient BuildLlm(bool longTimeout = false)
    {
        var http = new HttpClient { BaseAddress = new Uri("http://localhost:11434") };
        if (longTimeout) http.Timeout = TimeSpan.FromMinutes(5);
        return new OllamaLlmClient(http, OllamaLlmClient.DevModel);
    }

    // -- Test 1: Executor with a plan whose agent prompts are written by us --

    /// <summary>
    /// The plan is constructed manually here (as the LLM planner would produce it).
    /// Agent steps carry their own system/user prompts � no registry needed.
    /// Auto-map: $search returns string[], download expects a scalar url > fan-out.
    ///           $download returns object[], extract expects a single page   > fan-out.
    /// </summary>
    [Fact]
    public async Task Executor_RunsPlanWithEmbeddedAgentPrompts()
    {
        var llm = BuildLlm();
        var runner = new AgentStepRunner(llm);
        var executor = new PlanExecutor(BuildTools(), runner, new TestLogger(output));

        var plan = new PlanDefinition
        {
            Goal = "Search two items, extract key data from each page, then compare.",
            Steps =
            [
                new PlanStep
                {
                    Id = "search",
                    Tool = "search",
                    In = new() { ["query"] = JsonValue.Create("item comparison"), ["limit"] = JsonValue.Create(2) }
                },
                new PlanStep
                {
                    Id = "download",
                    Tool = "download",
                    In = new() { ["url"] = JsonValue.Create("$search") }
                    // auto-map: search returns string[] but download expects string > fan-out
                },
                new PlanStep
                {
                    Id = "extract",
                    Llm = "extract_data",
                    SystemPrompt = "You extract structured data from a page. Output ONLY valid JSON, no explanation.",
                    UserPrompt = "Extract the item name, weight_kg (number or null), runtime_min (number or null). Return {\"name\":\"...\",\"weight_kg\":...,\"runtime_min\":...}.",
                    In = new() { ["page"] = JsonValue.Create("$download") },
                    Each = true,   // fan-out: call once per downloaded page, collect results into array
                    Out = "json"
                },
                new PlanStep
                {
                    Id = "compare",
                    Llm = "compare_items",
                    SystemPrompt = "You compare items concisely. Output plain text only.",
                    UserPrompt = "Given these two items, state which is lighter (winner for portability) and which has longer runtime. Be brief.",
                    In = new() { ["items"] = JsonValue.Create("$extract") },
                    // extract returns an array here � agent compares the full array in one call
                    Out = "string"
                }
            ]
        };

        var store = new ExecutionStore();
        var result = await executor.ExecuteAsync(plan, store);

        foreach (var t in result.StepTraces)
            output.WriteLine($"  step={t.StepId} success={t.Success} err={t.ErrorMessage}");

        Assert.True(result.StepTraces.All(t => t.Success),
            $"Failed: {result.LastEnvelope?.Error?.Message}");

        store.TryGet("compare", out var answer);
        output.WriteLine($"\n=== FINAL ANSWER ===\n{answer?.ToString()}");
        Assert.NotNull(answer);
    }

    // -- Test 2: Full pipeline  userQuery > LlmPlanner > Executor ----------

    /// <summary>
    /// The LLM planner generates the full plan (including agent prompts) from the user query.
    /// The executor then runs it. This validates the whole chain end-to-end.
    /// </summary>
    [Fact]
    public async Task FullPipeline_PlannerGeneratesPlan_ExecutorRunsIt()
    {
        var llm = BuildLlm(longTimeout: true);
        var planner = new LlmPlanner(llm, BuildTools(), new TestLogger(output));
        var runner = new AgentStepRunner(llm);
        var executor = new PlanExecutor(BuildTools(), runner, new TestLogger(output));

        const string userQuery = "I'm looking for a good robot vacuum cleaner. Can you find two popular models, check their specs, and tell me which one is better?";

        output.WriteLine($"User query: {userQuery}");

        PlanDefinition plan;
        try
        {
            plan = await planner.CreatePlanAsync(userQuery);
        }
        catch (Exception ex)
        {
            output.WriteLine($"Planner error: {ex.Message}");
            throw;
        }

        output.WriteLine($"\n=== GENERATED PLAN ===");
        output.WriteLine(JsonSerializer.Serialize(plan, new JsonSerializerOptions { WriteIndented = true }));

        var store = new ExecutionStore();
        var result = await executor.ExecuteAsync(plan, store);

        foreach (var t in result.StepTraces)
            output.WriteLine($"  step={t.StepId} success={t.Success} err={t.ErrorMessage}");

        Assert.True(result.StepTraces.All(t => t.Success),
            $"Failed step: {result.LastEnvelope?.Error?.Message}");

        var lastId = plan.Steps[^1].Id;
        store.TryGet(lastId, out var final);
        output.WriteLine($"\n=== FINAL ANSWER ({lastId}) ===\n{final?.ToString()}");
        Assert.NotNull(final);
    }
}
