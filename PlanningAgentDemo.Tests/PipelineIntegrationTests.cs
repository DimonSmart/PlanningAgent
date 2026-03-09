using System.Text.Json;
using System.Text.Json.Nodes;
using PlanningAgentDemo.Agents;
using PlanningAgentDemo.Common;
using PlanningAgentDemo.Execution;
using PlanningAgentDemo.Orchestration;
using PlanningAgentDemo.Planning;
using PlanningAgentDemo.Tools;
using PlanningAgentDemo.Verification;
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

    // -- Test 2: Full pipeline  userQuery > Planner > Orchestrator ----------

    /// <summary>
    /// The LLM planner generates the full plan from the user query and the orchestrator
    /// drives planning, execution, verification, and replanning. This validates the
    /// system-level outcome instead of raw step success.
    /// </summary>
    [Fact]
    public async Task FullPipeline_PlannerAndOrchestrator_ReturnsSystemOutcome()
    {
        var llm = BuildLlm(longTimeout: true);
        var planner = new LlmPlanner(llm, BuildTools(), new TestLogger(output));
        var runner = new AgentStepRunner(llm);
        var executor = new PlanExecutor(BuildTools(), runner, new TestLogger(output));
        var orchestrator = new PlanningOrchestrator(
            planner,
            executor,
            new GoalVerifier(askUserEnabled: true),
            new TestLogger(output),
            maxAttempts: 3);

        const string userQuery = "I'm looking for a good robot vacuum cleaner. Can you find two popular models, check their specs, and tell me which one is better?";

        output.WriteLine($"User query: {userQuery}");

        PlanDefinition initialPlan;
        try
        {
            initialPlan = await planner.CreatePlanAsync(userQuery);
        }
        catch (Exception ex)
        {
            output.WriteLine($"Planner error: {ex.Message}");
            throw;
        }

        output.WriteLine($"\n=== GENERATED PLAN ===");
        output.WriteLine(JsonSerializer.Serialize(initialPlan, new JsonSerializerOptions { WriteIndented = true }));

        var result = await orchestrator.RunAsync(userQuery);
        var outcome = ClassifyOutcome(result);

        output.WriteLine($"\n=== SYSTEM OUTCOME ===\n{outcome}");
        if (result.Ok)
            output.WriteLine($"\n=== FINAL ANSWER ===\n{result.Data?.ToJsonString(new JsonSerializerOptions { WriteIndented = true })}");
        else
            output.WriteLine($"\n=== OUTCOME DETAILS ===\ncode={result.Error?.Code}\nmessage={result.Error?.Message}\ndetails={result.Error?.Details?.ToJsonString(new JsonSerializerOptions { WriteIndented = true })}");

        switch (outcome)
        {
            case GoalAction.Done:
                Assert.True(result.Ok);
                Assert.NotNull(result.Data);
                break;

            case GoalAction.AskUser:
                Assert.False(result.Ok);
                Assert.Equal("ask_user", result.Error?.Code);
                Assert.NotNull(result.Error?.Details);
                Assert.True(result.Error!.Details!.ContainsKey("question"));
                break;

            case GoalAction.Replan:
                Assert.False(result.Ok);
                Assert.NotNull(result.Error);
                Assert.False(string.IsNullOrWhiteSpace(result.Error!.Message));
                break;

            default:
                throw new InvalidOperationException($"Unexpected outcome: {outcome}");
        }
    }

    [Fact]
    public void GoalVerifier_UsesLastStepOutputInsteadOfHardcodedFinalKey()
    {
        var plan = new PlanDefinition
        {
            Goal = "Produce an answer.",
            Steps =
            [
                new PlanStep { Id = "search", Tool = "search", In = new() { ["query"] = JsonValue.Create("x") } },
                new PlanStep { Id = "compare", Llm = "compare", UserPrompt = "Compare.", In = new() }
            ]
        };

        var executionResult = new ExecutionResult
        {
            StepTraces =
            [
                new StepExecutionTrace { StepId = "search", Success = true },
                new StepExecutionTrace { StepId = "compare", Success = true }
            ]
        };

        var store = new ExecutionStore();
        store.Set("compare", JsonValue.Create("final answer"));

        var verdict = new GoalVerifier().Check(plan, executionResult, store);

        Assert.Equal(GoalAction.Done, verdict.Action);
    }

    [Fact]
    public async Task Orchestrator_FailsWhenIntermediateStepLooksIncomplete()
    {
        var plan = new PlanDefinition
        {
            Goal = "Extract and compare data.",
            Steps =
            [
                new PlanStep
                {
                    Id = "extract",
                    Llm = "extract",
                    SystemPrompt = "Return valid JSON only.",
                    UserPrompt = "Extract fields.",
                    In = new() { ["item"] = JsonValue.Create("stub") },
                    Out = "json"
                },
                new PlanStep
                {
                    Id = "compare",
                    Llm = "compare",
                    SystemPrompt = "Return plain text.",
                    UserPrompt = "Compare items.",
                    In = new() { ["items"] = JsonValue.Create("$extract") },
                    Out = "string"
                }
            ]
        };

        var planner = new StubPlanner(plan);
        var runner = new StubAgentRunner(
            new JsonObject
            {
                ["model"] = "",
                ["summary"] = ""
            },
            JsonValue.Create("comparison"));
        var executor = new PlanExecutor(BuildTools(), runner, new TestLogger(output));
        var orchestrator = new PlanningOrchestrator(planner, executor, new GoalVerifier(), new TestLogger(output));

        var result = await orchestrator.RunAsync("compare items");

        Assert.False(result.Ok);
        Assert.Equal("goal_not_achieved", result.Error?.Code);
        Assert.Contains("incomplete", result.Error?.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Orchestrator_ReplansWithExecutionContext_AndSucceeds()
    {
        var firstPlan = new PlanDefinition
        {
            Goal = "Produce an answer.",
            Steps =
            [
                new PlanStep
                {
                    Id = "extract",
                    Llm = "extract",
                    SystemPrompt = "Return valid JSON only.",
                    UserPrompt = "Extract data.",
                    In = new() { ["input"] = JsonValue.Create("x") },
                    Out = "json"
                }
            ]
        };

        var secondPlan = new PlanDefinition
        {
            Goal = "Produce an answer.",
            Steps =
            [
                new PlanStep
                {
                    Id = "final_answer",
                    Llm = "answer",
                    SystemPrompt = "Return valid JSON only.",
                    UserPrompt = "Answer.",
                    In = new() { ["input"] = JsonValue.Create("x") },
                    Out = "json"
                }
            ]
        };

        var planner = new StubReplanPlanner(firstPlan, secondPlan);
        var runner = new StubAgentRunner(
            new JsonObject { ["field"] = "" },
            new JsonObject { ["answer"] = "done" });
        var executor = new PlanExecutor(BuildTools(), runner, new TestLogger(output));
        var orchestrator = new PlanningOrchestrator(planner, executor, new GoalVerifier(), new TestLogger(output), maxAttempts: 2);

        var result = await orchestrator.RunAsync("answer the user query");

        Assert.True(result.Ok);
        Assert.NotNull(result.Data);
        Assert.Single(planner.ReplanRequests);
        Assert.Contains(planner.ReplanRequests[0].ExecutionResult.StepTraces[0].VerificationIssues,
            issue => issue.Code == "structurally_empty_output");
    }

    [Fact]
    public async Task AgentStepRunner_ReturnsFailure_WhenAgentReportsStructuredExecutionIssue()
    {
        var llm = new StubLlmClient("""
        {
          "_execution": {
            "status": "blocked",
            "needsReplan": true,
            "errors": [
              { "code": "insufficient_input", "message": "The provided data is not enough." }
            ]
          },
          "result": null
        }
        """);
        var runner = new AgentStepRunner(llm);

        var step = new PlanStep
        {
            Id = "extract",
            Llm = "extract",
            SystemPrompt = "Return valid JSON only.",
            UserPrompt = "Extract data.",
            In = new() { ["input"] = JsonValue.Create("x") },
            Out = "json"
        };

        var result = await runner.ExecuteAsync(step, new JsonObject { ["input"] = "x" });

        Assert.False(result.Ok);
        Assert.Equal("llm_reported_issue", result.Error?.Code);
        Assert.Equal("blocked", result.Error?.Details?["status"]?.GetValue<string>());
    }

    [Fact]
    public async Task Orchestrator_PassesStructuredAgentErrors_ToReplanner()
    {
        var firstPlan = new PlanDefinition
        {
            Goal = "Produce an answer.",
            Steps =
            [
                new PlanStep
                {
                    Id = "extract",
                    Llm = "extract",
                    SystemPrompt = "Return valid JSON only.",
                    UserPrompt = "Extract data.",
                    In = new() { ["input"] = JsonValue.Create("x") },
                    Out = "json"
                }
            ]
        };

        var secondPlan = new PlanDefinition
        {
            Goal = "Produce an answer.",
            Steps =
            [
                new PlanStep
                {
                    Id = "final_answer",
                    Llm = "answer",
                    SystemPrompt = "Return valid JSON only.",
                    UserPrompt = "Answer.",
                    In = new() { ["input"] = JsonValue.Create("x") },
                    Out = "json"
                }
            ]
        };

        var planner = new StubReplanPlanner(firstPlan, secondPlan);
        var llm = new SequenceLlmClient(
            """
            {
              "_execution": {
                "status": "blocked",
                "needsReplan": true,
                "errors": [
                  { "code": "insufficient_input", "message": "The provided data is not enough." }
                ]
              },
              "result": null
            }
            """,
            """
            {
              "answer": "done"
            }
            """);
        var runner = new AgentStepRunner(llm);
        var executor = new PlanExecutor(BuildTools(), runner, new TestLogger(output));
        var orchestrator = new PlanningOrchestrator(planner, executor, new GoalVerifier(), new TestLogger(output), maxAttempts: 2);

        var result = await orchestrator.RunAsync("answer the user query");

        Assert.True(result.Ok);
        Assert.Single(planner.ReplanRequests);
        var trace = Assert.Single(planner.ReplanRequests[0].ExecutionResult.StepTraces);
        Assert.Equal("llm_reported_issue", trace.ErrorCode);
        Assert.Equal("blocked", trace.ErrorDetails?["status"]?.GetValue<string>());
    }

    private sealed class StubPlanner(PlanDefinition plan) : IPlanner
    {
        public Task<PlanDefinition> CreatePlanAsync(string userQuery, CancellationToken cancellationToken = default) =>
            Task.FromResult(plan);
    }

    private sealed class StubReplanPlanner(PlanDefinition initialPlan, PlanDefinition replannedPlan) : IReplanCapablePlanner
    {
        public List<PlannerReplanRequest> ReplanRequests { get; } = [];

        public Task<PlanDefinition> CreatePlanAsync(string userQuery, CancellationToken cancellationToken = default) =>
            Task.FromResult(initialPlan);

        public Task<PlanDefinition> ReplanAsync(PlannerReplanRequest request, CancellationToken cancellationToken = default)
        {
            ReplanRequests.Add(request);
            return Task.FromResult(replannedPlan);
        }
    }

    private sealed class StubAgentRunner(params JsonNode?[] outputs) : IAgentStepRunner
    {
        private readonly Queue<JsonNode?> _outputs = new(outputs.Select(output => output?.DeepClone()));

        public Task<ResultEnvelope<JsonNode?>> ExecuteAsync(PlanStep step, JsonObject resolvedInputs, CancellationToken cancellationToken = default)
        {
            var output = _outputs.Dequeue();
            return Task.FromResult(ResultEnvelope<JsonNode?>.Success(output?.DeepClone()));
        }
    }

    private sealed class StubLlmClient(string response) : ILlmClient
    {
        public Task<string> GenerateAsync(string systemPrompt, string userPrompt, CancellationToken cancellationToken = default) =>
            Task.FromResult(response);
    }

    private sealed class SequenceLlmClient(params string[] responses) : ILlmClient
    {
        private readonly Queue<string> _responses = new(responses);

        public Task<string> GenerateAsync(string systemPrompt, string userPrompt, CancellationToken cancellationToken = default) =>
            Task.FromResult(_responses.Dequeue());
    }

    private static GoalAction ClassifyOutcome(ResultEnvelope<JsonNode?> result)
    {
        if (result.Ok)
            return GoalAction.Done;

        return string.Equals(result.Error?.Code, "ask_user", StringComparison.Ordinal)
            ? GoalAction.AskUser
            : GoalAction.Replan;
    }
}
