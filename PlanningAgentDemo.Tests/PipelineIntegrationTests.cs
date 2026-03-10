using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using System.ClientModel;
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
        "Search the web. Returns candidate page URLs. In this test environment each returned URL points to a page that already contains details about one candidate item.",
        JsonNode.Parse(@"{""type"":""object"",""properties"":{""query"":{""type"":""string""},""limit"":{""type"":""number""}},""required"":[""query""]}")!.AsObject(),
        JsonNode.Parse(@"{""type"":""array"",""items"":{""type"":""string""}}")!.AsObject(), [], []);

    public Task<ResultEnvelope<JsonElement?>> ExecuteAsync(JsonElement input, CancellationToken ct = default) =>
        Task.FromResult(ResultEnvelope<JsonElement?>.Success(JsonSerializer.SerializeToElement(new[]
        {
            "https://example.com/item-a",
            "https://example.com/item-b"
        })));
}

/// <summary>Returns fake page content for a URL (simulates an HTTP download).</summary>
public class MockDownloadTool : ITool
{
    public string Name => "download";
    public ToolPlannerMetadata PlannerMetadata => new(
        "download",
        "Download a single page by URL. Returns an object with the page title and body text.",
        JsonNode.Parse(@"{""type"":""object"",""properties"":{""url"":{""type"":""string""}},""required"":[""url""]}")!.AsObject(),
        JsonNode.Parse(@"{""type"":""object"",""properties"":{""url"":{""type"":""string""},""title"":{""type"":""string""},""body"":{""type"":""string""}},""required"":[""url"",""title"",""body""]}")!.AsObject(), [], []);

    public Task<ResultEnvelope<JsonElement?>> ExecuteAsync(JsonElement input, CancellationToken ct = default)
    {
        var url = input.ValueKind == JsonValueKind.Object
            && input.TryGetProperty("url", out var urlElement)
            && urlElement.ValueKind == JsonValueKind.String
                ? urlElement.GetString() ?? string.Empty
                : string.Empty;

        return url.Contains("item-a")
            ? Task.FromResult(ResultEnvelope<JsonElement?>.Success(JsonSerializer.SerializeToElement(new
                {
                    url,
                    title = "RoboClean A1 Max review",
                    body = "RoboClean A1 Max is a popular robot vacuum cleaner with 7000 Pa suction power, up to 180 minutes of battery runtime, a 0.5 L dustbin, LiDAR navigation, and a list price of $799."
                })))
            : Task.FromResult(ResultEnvelope<JsonElement?>.Success(JsonSerializer.SerializeToElement(new
                {
                    url,
                    title = "HomeSweep S5 review",
                    body = "HomeSweep S5 is a popular robot vacuum cleaner with 5000 Pa suction power, up to 140 minutes of battery runtime, a 0.4 L dustbin, vSLAM navigation, and a list price of $649."
                })));
    }
}

// --- Tests --------------------------------------------------------------------

public class PipelineTests(ITestOutputHelper output)
{
    private const string DevModel = "gpt-oss:120b-cloud";

    private static IToolRegistry BuildTools() =>
        new ToolRegistry(new ITool[] { new MockSearchTool(), new MockDownloadTool() });

    private static IChatClient BuildChatClient(bool longTimeout = false)
    {
        _ = longTimeout;

        var clientOptions = new OpenAIClientOptions
        {
            Endpoint = new Uri("http://localhost:11434/v1/")
        };

        var chatClient = new OpenAIClient(new ApiKeyCredential("ollama"), clientOptions)
            .GetChatClient(DevModel);

        return chatClient.AsIChatClient();
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
        var chatClient = BuildChatClient();
        var runner = new AgentStepRunner(chatClient);
        var executor = new PlanExecutor(BuildTools(), runner, new TestLogger(output));

        var plan = new PlanDefinition
        {
            Goal = "Find two robot vacuum models, extract key specs from each page, then compare them.",
            Steps =
            [
                new PlanStep
                {
                    Id = "search",
                    Tool = "search",
                    In = new() { ["query"] = JsonValue.Create("popular robot vacuum comparison"), ["limit"] = JsonValue.Create(2) }
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
                    Id = "extract_specs",
                    Llm = "extract_specs",
                    SystemPrompt = "You extract structured robot vacuum specifications from a page. Return ONLY valid JSON.",
                    UserPrompt = "Extract the model name, suction_power_pa, battery_runtime_min, dustbin_capacity_l, navigation, and price_usd. Return a JSON object with those fields.",
                    In = new() { ["page"] = JsonValue.Create("$download") },
                    Each = true,   // fan-out: call once per downloaded page, collect results into array
                    Out = "json"
                },
                new PlanStep
                {
                    Id = "compare",
                    Llm = "compare_items",
                    SystemPrompt = "You compare two robot vacuum models and return ONLY valid JSON.",
                    UserPrompt = "Compare the models and return {\"better_model\":\"...\",\"reason\":\"...\"}. Prefer better suction, longer battery runtime, larger dustbin, stronger navigation, and lower price when the trade-offs are close.",
                    In = new() { ["items"] = JsonValue.Create("$extract_specs") },
                    // extract returns an array here � agent compares the full array in one call
                    Out = "json"
                }
            ]
        };

        var result = await executor.ExecuteAsync(plan);

        foreach (var t in result.StepTraces)
            output.WriteLine($"  step={t.StepId} success={t.Success} err={t.ErrorMessage}");

        Assert.True(result.StepTraces.All(t => t.Success),
            $"Failed: {result.LastEnvelope?.Error?.Message}");

        var answer = plan.Steps.Single(step => step.Id == "compare").Result;
        output.WriteLine($"\n=== FINAL ANSWER ===\n{SerializeJson(answer)}");
        Assert.NotNull(answer);
        Assert.Equal("RoboClean A1 Max", GetStringProperty(answer, "better_model"));
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
        var chatClient = BuildChatClient(longTimeout: true);
        var answerAsserter = new CachedLlmAnswerAsserter(chatClient, DevModel);
        var planner = new LlmPlanner(chatClient, BuildTools(), new TestLogger(output));
        var replanner = new LlmReplanner(chatClient, BuildTools(), new TestLogger(output));
        var runner = new AgentStepRunner(chatClient);
        var executor = new PlanExecutor(BuildTools(), runner, new TestLogger(output));
        var orchestrator = new PlanningOrchestrator(
            planner,
            executor,
            new GoalVerifier(askUserEnabled: true),
            new TestLogger(output),
            maxAttempts: 3,
            replanner: replanner);

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
        AssertReasonableComparisonPlan(initialPlan);

        var result = await orchestrator.RunAsync(userQuery);
        var outcome = ClassifyOutcome(result);

        output.WriteLine($"\n=== SYSTEM OUTCOME ===\n{outcome}");
        if (result.Ok)
            output.WriteLine($"\n=== FINAL ANSWER ===\n{SerializeJson(result.Data)}");
        else
            output.WriteLine($"\n=== OUTCOME DETAILS ===\ncode={result.Error?.Code}\nmessage={result.Error?.Message}\ndetails={SerializeJson(result.Error?.Details)}");

        switch (outcome)
        {
            case GoalAction.Done:
                Assert.True(result.Ok);
                Assert.NotNull(result.Data);
                var verdict = await answerAsserter.EvaluateAsync(userQuery, result.Data);
                output.WriteLine($"LLM asserter verdict: isAnswer={verdict.IsAnswer} cache={verdict.FromCache} comment={verdict.Comment}");
                Assert.True(verdict.IsAnswer, verdict.Comment);
                break;

            case GoalAction.AskUser:
                Assert.False(result.Ok);
                Assert.Equal("ask_user", result.Error?.Code);
                Assert.NotNull(result.Error?.Details);
                Assert.True(HasProperty(result.Error?.Details, "question"));
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
    public async Task LlmPlanner_RejectsNonCanonicalPlanShape()
    {
        var planner = new LlmPlanner(
            new StubChatClient(
                """
                {
                  "ok": true,
                  "data": {
                    "goal": "Compare two robot vacuum cleaners.",
                    "steps": [
                      {
                        "tool": "search",
                        "query": "popular robot vacuum cleaner",
                        "limit": 2
                      },
                      {
                        "id": "download_pages",
                        "tool": "download",
                        "url": "$search"
                      },
                      {
                        "id": "compare1",
                        "agent": "compare_models",
                        "system": "Return ONLY valid JSON.",
                        "prompt": "Compare the provided items and recommend one model.",
                        "items": "$download_pages"
                      }
                    ]
                  },
                  "error": null
                }
                """),
            BuildTools(),
            new TestLogger(output));

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => planner.CreatePlanAsync("Compare two popular robot vacuums."));
    }

    [Fact]
    public async Task AgentStepRunner_WithRealLlm_ReturnsStructuredFailure_WhenFactsAreMissing()
    {
        var runner = new AgentStepRunner(BuildChatClient(longTimeout: true));
        var step = new PlanStep
        {
            Id = "extract_specs",
            Llm = "extract_specs",
            SystemPrompt = "You extract robot vacuum specifications from a page. Return ONLY valid JSON. If the model name or the requested facts are not present, do not guess and do not return a successful payload.",
            UserPrompt = "Extract model_name, suction_power_pa, battery_runtime_min, dustbin_capacity_l, navigation, price_usd, and docking_station_wattage.",
            In = new() { ["page"] = JsonValue.Create("stub") },
            Out = "json"
        };

        var result = await runner.ExecuteAsync(
            step,
            JsonSerializer.SerializeToElement(new
            {
                page = new
                {
                    title = "Robot vacuum buying tips",
                    body = "This page only says that robot vacuums are convenient for daily cleaning. It does not name a model and does not list technical specifications."
                }
            }));

        Assert.False(result.Ok);
        Assert.False(string.IsNullOrWhiteSpace(result.Error?.Code));
        Assert.Equal("blocked", GetStringProperty(result.Error?.Details, "status"));
        Assert.True(PropertyIsArray(result.Error?.Details, "missingFacts"));
    }

    [Fact]
    public async Task Orchestrator_ReplansAfterStructuredAgentFailure_WithRealLlm()
    {
        var chatClient = BuildChatClient(longTimeout: true);
        var answerAsserter = new CachedLlmAnswerAsserter(chatClient, DevModel);
        var planner = new StubPlanner(BuildBadInitialVacuumPlan());
        var replanner = new CapturingReplanner(new LlmReplanner(chatClient, BuildTools(), new TestLogger(output)));
        var runner = new AgentStepRunner(chatClient);
        var executor = new PlanExecutor(BuildTools(), runner, new TestLogger(output));
        var orchestrator = new PlanningOrchestrator(planner, executor, new GoalVerifier(), new TestLogger(output), maxAttempts: 2, replanner: replanner);

        const string userQuery = "I'm looking for a good robot vacuum cleaner. Can you find two popular models, check their specs, and tell me which one is better?";

        var result = await orchestrator.RunAsync(userQuery);

        Assert.True(result.Ok);
        Assert.Single(replanner.ReplanRequests);
        var trace = Assert.Single(replanner.ReplanRequests[0].ExecutionResult.StepTraces, step => !step.Success);
        Assert.False(string.IsNullOrWhiteSpace(trace.ErrorCode));
        Assert.Equal("blocked", GetStringProperty(trace.ErrorDetails, "status"));
        Assert.True(PropertyIsArray(trace.ErrorDetails, "missingFacts"));
        var verdict = await answerAsserter.EvaluateAsync(userQuery, result.Data);
        output.WriteLine($"LLM asserter verdict: isAnswer={verdict.IsAnswer} cache={verdict.FromCache} comment={verdict.Comment}");
        Assert.True(verdict.IsAnswer, verdict.Comment);
    }

    [Fact]
    public async Task Orchestrator_Replan_ReusesSuccessfulPrefixOutputs()
    {
        var searchTool = new CountingTool(new MockSearchTool());
        var downloadTool = new CountingTool(new MockDownloadTool());
        var tools = new ToolRegistry(new ITool[] { searchTool, downloadTool });

        var initialPlan = BuildBadInitialVacuumPlan();
        var replannedPlan = new PlanDefinition
        {
            Goal = initialPlan.Goal,
            Steps =
            [
                initialPlan.Steps[0],
                initialPlan.Steps[1],
                new PlanStep
                {
                    Id = "extract_specs",
                    Llm = "extract_specs",
                    SystemPrompt = "Return ONLY valid JSON.",
                    UserPrompt = "Extract model_name, suction_power_pa, battery_runtime_min, dustbin_capacity_l, navigation, and price_usd. Use null for absent fields.",
                    In = new() { ["page"] = JsonValue.Create("$download") },
                    Each = true,
                    Out = "json"
                },
                initialPlan.Steps[3]
            ]
        };

        var planner = new StubPlanner(initialPlan);
        var replanner = new StubReplanner(replannedPlan);
        var runner = new SequenceEnvelopeAgentRunner(
            ResultEnvelope<JsonNode?>.Failure(
                "missing_field",
                "missing docking station wattage",
                ToNullableElement(new JsonObject
                {
                    ["status"] = "blocked",
                    ["needsReplan"] = true,
                    ["missingFacts"] = new JsonArray("docking_station_wattage"),
                    ["observedEvidence"] = new JsonArray("Page contains suction, battery, dustbin, navigation, and price.")
                })),
            ResultEnvelope<JsonNode?>.Success(new JsonObject
            {
                ["model_name"] = "RoboClean A1 Max",
                ["suction_power_pa"] = 7000,
                ["battery_runtime_min"] = 180,
                ["dustbin_capacity_l"] = 0.5,
                ["navigation"] = "LiDAR",
                ["price_usd"] = 799
            }),
            ResultEnvelope<JsonNode?>.Success(new JsonObject
            {
                ["model_name"] = "HomeSweep S5",
                ["suction_power_pa"] = 5000,
                ["battery_runtime_min"] = 140,
                ["dustbin_capacity_l"] = 0.4,
                ["navigation"] = "vSLAM",
                ["price_usd"] = 649
            }),
            ResultEnvelope<JsonNode?>.Success(new JsonObject
            {
                ["better_model"] = "RoboClean A1 Max",
                ["reason"] = "Higher suction and battery runtime."
            }));
        var executor = new PlanExecutor(tools, runner, new TestLogger(output));
        var orchestrator = new PlanningOrchestrator(planner, executor, new GoalVerifier(), new TestLogger(output), maxAttempts: 2, replanner: replanner);

        var result = await orchestrator.RunAsync("compare two robot vacuums");

        Assert.True(result.Ok);
        Assert.Equal(1, searchTool.CallCount);
        Assert.Equal(2, downloadTool.CallCount);
    }

    [Fact]
    public async Task LlmReplanner_CanReadCompactFailedTrace_BeforeEditingPlan()
    {
        var replanner = new LlmReplanner(
            new SequenceChatClient(
                """
                {
                  "ok": true,
                  "data": {
                    "done": false,
                    "reason": "Need the failed trace details.",
                    "actions": [
                      {
                        "tool": "runtime.readFailedTrace",
                        "in": { "stepId": "extract" }
                      }
                    ]
                  },
                  "error": null
                }
                """,
                """
                {
                  "ok": true,
                  "data": {
                    "done": true,
                    "reason": "Replace the failed extraction step.",
                    "actions": [
                      {
                        "tool": "plan.replaceStep",
                        "in": {
                          "stepId": "extract",
                          "step": {
                            "id": "extract",
                            "llm": "extract",
                            "systemPrompt": "Return ONLY valid JSON. Use null for missing facts.",
                            "userPrompt": "Extract model_name and suction_power_pa.",
                            "in": { "page": "$download" },
                            "out": "json",
                            "each": true
                          }
                        }
                      }
                    ]
                  },
                  "error": null
                }
                """),
            BuildTools(),
            new TestLogger(output));

        var previousPlan = new PlanDefinition
        {
            Goal = "Extract and compare product data.",
            Steps =
            [
                new PlanStep
                {
                    Id = "search",
                    Tool = "search",
                    In = new() { ["query"] = JsonValue.Create("robot vacuum"), ["limit"] = JsonValue.Create(2) }
                },
                new PlanStep
                {
                    Id = "download",
                    Tool = "download",
                    In = new() { ["url"] = JsonValue.Create("$search") }
                },
                new PlanStep
                {
                    Id = "extract",
                    Llm = "extract",
                    SystemPrompt = "Return ONLY valid JSON.",
                    UserPrompt = "Extract model_name and docking_station_wattage.",
                    In = new() { ["page"] = JsonValue.Create("$download") },
                    Each = true,
                    Out = "json"
                }
            ]
        };

        var replanned = await replanner.ReplanAsync(new PlannerReplanRequest
        {
            UserQuery = "compare two robot vacuums",
            AttemptNumber = 1,
            Plan = previousPlan,
            ExecutionResult = new ExecutionResult
            {
                StepTraces =
                [
                    new StepExecutionTrace
                    {
                        StepId = "download",
                        Success = true
                    },
                    new StepExecutionTrace
                    {
                        StepId = "extract",
                        Success = false,
                        ErrorCode = "missing_field",
                        ErrorMessage = "missing docking station wattage",
                        ErrorDetails = ToNullableElement(new JsonObject
                        {
                            ["status"] = "blocked",
                            ["needsReplan"] = true,
                            ["missingFacts"] = new JsonArray("docking_station_wattage"),
                            ["observedEvidence"] = new JsonArray("The downloaded page contains suction power and price only.")
                        })
                    }
                ]
            },
            GoalVerdict = new GoalVerdict
            {
                Action = GoalAction.Replan,
                Reason = "Execution has failed steps.",
                Missing = ["successful_execution"]
            }
        });

        Assert.Same(previousPlan, replanned);
        var extract = Assert.Single(replanned.Steps, step => step.Id == "extract");
        Assert.Equal("Return ONLY valid JSON. Use null for missing facts.", extract.SystemPrompt);
    }

    [Fact]
    public void PlanEditingSession_ReplaceStep_ResetsChangedStepAndDownstreamState()
    {
        var plan = new PlanDefinition
        {
            Goal = "Compare two models.",
            Steps =
            [
                new PlanStep
                {
                    Id = "search",
                    Tool = "search",
                    In = new() { ["query"] = JsonValue.Create("robot vacuum") },
                    Status = PlanStepStatuses.Done,
                    Result = JsonSerializer.SerializeToElement(new[]
                    {
                        "https://example.com/a",
                        "https://example.com/b"
                    })
                },
                new PlanStep
                {
                    Id = "extract",
                    Llm = "extract_specs",
                    SystemPrompt = "Return ONLY valid JSON.",
                    UserPrompt = "Extract model_name.",
                    In = new() { ["page"] = JsonValue.Create("$search") },
                    Out = "json",
                    Each = true,
                    Status = PlanStepStatuses.Fail,
                    Result = JsonSerializer.SerializeToElement(Array.Empty<object>()),
                    Error = new PlanStepError { Code = "missing_field", Message = "missing facts" }
                },
                new PlanStep
                {
                    Id = "compare",
                    Llm = "compare_models",
                    SystemPrompt = "Return ONLY valid JSON.",
                    UserPrompt = "Compare the models.",
                    In = new() { ["items"] = JsonValue.Create("$extract") },
                    Out = "json",
                    Status = PlanStepStatuses.Done,
                    Result = JsonSerializer.SerializeToElement(new { better_model = "stale" })
                }
            ]
        };

        var session = new PlanEditingSession(plan);
        var action = JsonNode.Parse(
            """
            {
              "tool": "plan.replaceStep",
              "in": {
                "stepId": "extract",
                "step": {
                  "id": "extract",
                  "llm": "extract_specs",
                  "systemPrompt": "Return ONLY valid JSON. Use null for missing facts.",
                  "userPrompt": "Extract model_name and suction_power_pa.",
                  "in": { "page": "$search" },
                  "out": "json",
                  "each": true
                }
              }
            }
            """)!.AsObject();

        var result = session.ExecuteAction(
            action["tool"]?.GetValue<string>() ?? string.Empty,
            action["in"]?.AsObject() ?? []);

        Assert.True(result["ok"]?.GetValue<bool>());
        Assert.Equal(PlanStepStatuses.Done, plan.Steps[0].Status);
        Assert.NotNull(plan.Steps[0].Result);

        Assert.Equal(PlanStepStatuses.Todo, plan.Steps[1].Status);
        Assert.Null(plan.Steps[1].Result);
        Assert.Null(plan.Steps[1].Error);
        Assert.Equal("Extract model_name and suction_power_pa.", plan.Steps[1].UserPrompt);

        Assert.Equal(PlanStepStatuses.Todo, plan.Steps[2].Status);
        Assert.Null(plan.Steps[2].Result);
        Assert.Null(plan.Steps[2].Error);
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

        plan.Steps[1].Status = PlanStepStatuses.Done;
        plan.Steps[1].Result = JsonSerializer.SerializeToElement("final answer");

        var verdict = new GoalVerifier().Check(plan, executionResult);

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
        Assert.Equal("verification_failed", result.Error?.Code);
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

        var planner = new StubPlanner(firstPlan);
        var replanner = new StubReplanner(secondPlan);
        var runner = new StubAgentRunner(
            new JsonObject { ["field"] = "" },
            new JsonObject { ["answer"] = "done" });
        var executor = new PlanExecutor(BuildTools(), runner, new TestLogger(output));
        var orchestrator = new PlanningOrchestrator(planner, executor, new GoalVerifier(), new TestLogger(output), maxAttempts: 2, replanner: replanner);

        var result = await orchestrator.RunAsync("answer the user query");

        Assert.True(result.Ok);
        Assert.NotNull(result.Data);
        Assert.Single(replanner.ReplanRequests);
        Assert.Contains(replanner.ReplanRequests[0].ExecutionResult.StepTraces[0].VerificationIssues,
            issue => issue.Code == "structurally_empty_output");
    }

    [Fact]
    public async Task AgentStepRunner_ReturnsFailure_WhenAgentReportsStructuredExecutionIssue()
    {
                var chatClient = new StubChatClient("""
        {
          "ok": false,
          "data": null,
          "error": {
            "code": "insufficient_input",
            "message": "The provided data is not enough.",
            "details": {
              "status": "blocked",
              "needsReplan": true,
              "missingFacts": ["required_input"],
              "observedEvidence": ["The provided data is not enough."]
            }
          }
        }
        """);
        var runner = new AgentStepRunner(chatClient);

        var step = new PlanStep
        {
            Id = "extract",
            Llm = "extract",
            SystemPrompt = "Return valid JSON only.",
            UserPrompt = "Extract data.",
            In = new() { ["input"] = JsonValue.Create("x") },
            Out = "json"
        };

        var result = await runner.ExecuteAsync(step, JsonSerializer.SerializeToElement(new { input = "x" }));

        Assert.False(result.Ok);
        Assert.Equal("insufficient_input", result.Error?.Code);
        Assert.Equal("blocked", GetStringProperty(result.Error?.Details, "status"));
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

        var planner = new StubPlanner(firstPlan);
        var replanner = new StubReplanner(secondPlan);
                var chatClient = new SequenceChatClient(
            """
            {
              "ok": false,
              "data": null,
              "error": {
                "code": "insufficient_input",
                "message": "The provided data is not enough.",
                "details": {
                  "status": "blocked",
                  "needsReplan": true,
                  "missingFacts": ["required_input"],
                  "observedEvidence": ["The provided data is not enough."]
                }
              }
            }
            """,
            """
            {
              "ok": true,
              "data": {
                "answer": "done"
              },
              "error": null
            }
            """);
                var runner = new AgentStepRunner(chatClient);
        var executor = new PlanExecutor(BuildTools(), runner, new TestLogger(output));
        var orchestrator = new PlanningOrchestrator(planner, executor, new GoalVerifier(), new TestLogger(output), maxAttempts: 2, replanner: replanner);

        var result = await orchestrator.RunAsync("answer the user query");

        Assert.True(result.Ok);
        Assert.Single(replanner.ReplanRequests);
        var trace = Assert.Single(replanner.ReplanRequests[0].ExecutionResult.StepTraces);
        Assert.Equal("insufficient_input", trace.ErrorCode);
        Assert.Equal("blocked", GetStringProperty(trace.ErrorDetails, "status"));
    }

    private static void AssertReasonableComparisonPlan(PlanDefinition plan)
    {
        Assert.True(plan.Steps.Count >= 4, "Plan should contain at least search, download, extract, and compare phases.");

        var search = plan.Steps[0];
        Assert.Equal("search", search.Tool);
        Assert.True(string.IsNullOrWhiteSpace(search.Llm));

        var downloadSteps = plan.Steps
            .Where(step => string.Equals(step.Tool, "download", StringComparison.Ordinal))
            .ToList();
        Assert.NotEmpty(downloadSteps);
        Assert.All(downloadSteps, step => Assert.True(ReferencesAnyStep(step, [search.Id])));

        var extractSteps = plan.Steps
            .Where(step => string.IsNullOrWhiteSpace(step.Tool) && !string.IsNullOrWhiteSpace(step.Llm))
            .Where(step => step.Id != plan.Steps[^1].Id)
            .Where(step => ReferencesAnyStep(step, downloadSteps.Select(downloadStep => downloadStep.Id)))
            .ToList();
        Assert.NotEmpty(extractSteps);

        var compare = plan.Steps[^1];
        Assert.True(string.IsNullOrWhiteSpace(compare.Tool));
        Assert.False(string.IsNullOrWhiteSpace(compare.Llm));
        Assert.True(ReferencesAnyStep(compare, extractSteps.Select(extractStep => extractStep.Id)));
    }

    private static bool ReferencesAnyStep(PlanStep step, IEnumerable<string> stepIds)
    {
        var stepIdSet = stepIds.ToHashSet(StringComparer.Ordinal);
        foreach (var value in step.In.Values)
        {
            if (value is not JsonValue jsonValue || !jsonValue.TryGetValue<string>(out var text) || !text.StartsWith("$", StringComparison.Ordinal))
                continue;

            var candidate = text[1..];
            var bracketIndex = candidate.IndexOf('[');
            var dotIndex = candidate.IndexOf('.');
            var endIndex = bracketIndex >= 0 && dotIndex >= 0
                ? Math.Min(bracketIndex, dotIndex)
                : bracketIndex >= 0
                    ? bracketIndex
                    : dotIndex;
            var stepId = endIndex >= 0 ? candidate[..endIndex] : candidate;
            if (stepIdSet.Contains(stepId))
                return true;
        }

        return false;
    }

    private static void AssertMeaningfulVacuumAnswer(JsonElement? answer)
    {
        Assert.NotNull(answer);

        if (answer is not JsonElement answerElement)
            throw new Xunit.Sdk.XunitException("Answer should not be null.");

        switch (answerElement.ValueKind)
        {
            case JsonValueKind.Object:
                var betterModel =
                    GetStringProperty(answerElement, "betterModel")
                    ?? GetStringProperty(answerElement, "better_model")
                    ?? GetStringProperty(answerElement, "recommended_model")
                    ?? GetStringProperty(answerElement, "recommendedModel")
                    ?? GetStringProperty(answerElement, "bestModel")
                    ?? GetStringProperty(answerElement, "preferredModel");
                Assert.False(string.IsNullOrWhiteSpace(betterModel));
                Assert.True(
                    betterModel.Contains("RoboClean", StringComparison.OrdinalIgnoreCase)
                    || betterModel.Contains("A1", StringComparison.OrdinalIgnoreCase));
                break;

            case JsonValueKind.String:
                var text = answerElement.GetString();
                Assert.False(string.IsNullOrWhiteSpace(text));
                Assert.True(
                    text.Contains("RoboClean", StringComparison.OrdinalIgnoreCase)
                    || text.Contains("A1", StringComparison.OrdinalIgnoreCase));
                break;

            default:
                throw new Xunit.Sdk.XunitException($"Unexpected final answer node: {SerializeJson(answerElement)}");
        }
    }

    private static PlanDefinition BuildBadInitialVacuumPlan() =>
        new()
        {
            Goal = "Find two robot vacuum models, inspect their specifications, and decide which one is better.",
            Steps =
            [
                new PlanStep
                {
                    Id = "search",
                    Tool = "search",
                    In = new() { ["query"] = JsonValue.Create("popular robot vacuum cleaner"), ["limit"] = JsonValue.Create(2) }
                },
                new PlanStep
                {
                    Id = "download",
                    Tool = "download",
                    In = new() { ["url"] = JsonValue.Create("$search") }
                },
                new PlanStep
                {
                    Id = "extract_specs",
                    Llm = "extract_specs",
                    SystemPrompt = "You extract robot vacuum specifications from a page. Return ONLY valid JSON. If any requested field is absent, return a blocked execution error instead of a successful payload.",
                    UserPrompt = "Extract model_name, suction_power_pa, battery_runtime_min, dustbin_capacity_l, navigation, price_usd, and docking_station_wattage.",
                    In = new() { ["page"] = JsonValue.Create("$download") },
                    Each = true,
                    Out = "json"
                },
                new PlanStep
                {
                    Id = "compare",
                    Llm = "compare_models",
                    SystemPrompt = "You compare robot vacuum models and return ONLY valid JSON.",
                    UserPrompt = "Compare the models and return {\"better_model\":\"...\",\"reason\":\"...\"}.",
                    In = new() { ["items"] = JsonValue.Create("$extract_specs") },
                    Out = "json"
                }
            ]
        };

    private sealed class StubPlanner(PlanDefinition plan) : IPlanner
    {
        public Task<PlanDefinition> CreatePlanAsync(string userQuery, CancellationToken cancellationToken = default) =>
            Task.FromResult(plan);
    }

    private sealed class StubReplanner(PlanDefinition replannedPlan) : IReplanner
    {
        public List<PlannerReplanRequest> ReplanRequests { get; } = [];

        public Task<PlanDefinition> ReplanAsync(PlannerReplanRequest request, CancellationToken cancellationToken = default)
        {
            ReplanRequests.Add(request);
            request.Plan.Steps.Clear();
            foreach (var step in replannedPlan.Steps.Select(CloneStep))
                request.Plan.Steps.Add(step);

            return Task.FromResult(request.Plan);
        }

        private static PlanStep CloneStep(PlanStep step) =>
            JsonSerializer.Deserialize<PlanStep>(JsonSerializer.Serialize(step))
            ?? throw new InvalidOperationException($"Failed to clone step '{step.Id}'.");
    }

    private sealed class CapturingReplanner(IReplanner replanner) : IReplanner
    {
        public List<PlannerReplanRequest> ReplanRequests { get; } = [];

        public async Task<PlanDefinition> ReplanAsync(PlannerReplanRequest request, CancellationToken cancellationToken = default)
        {
            ReplanRequests.Add(request);
            return await replanner.ReplanAsync(request, cancellationToken);
        }
    }

    private sealed class CountingTool(ITool innerTool) : ITool
    {
        public int CallCount { get; private set; }

        public string Name => innerTool.Name;

        public ToolPlannerMetadata PlannerMetadata => innerTool.PlannerMetadata;

        public async Task<ResultEnvelope<JsonElement?>> ExecuteAsync(JsonElement input, CancellationToken ct = default)
        {
            CallCount++;
            return await innerTool.ExecuteAsync(input, ct);
        }
    }

    private sealed class StubAgentRunner(params JsonNode?[] outputs) : IAgentStepRunner
    {
        private readonly Queue<JsonElement?> _outputs = new(outputs.Select(ToNullableElement));

        public Task<ResultEnvelope<JsonElement?>> ExecuteAsync(PlanStep step, JsonElement resolvedInputs, CancellationToken cancellationToken = default)
        {
            var output = _outputs.Dequeue();
            return Task.FromResult(ResultEnvelope<JsonElement?>.Success(CloneElement(output)));
        }
    }

    private sealed class SequenceEnvelopeAgentRunner(params ResultEnvelope<JsonNode?>[] outputs) : IAgentStepRunner
    {
        private readonly Queue<ResultEnvelope<JsonElement?>> _outputs = new(outputs.Select(ConvertEnvelope));

        public Task<ResultEnvelope<JsonElement?>> ExecuteAsync(PlanStep step, JsonElement resolvedInputs, CancellationToken cancellationToken = default)
        {
            var next = _outputs.Dequeue();
            return Task.FromResult(CloneEnvelope(next));
        }

        private static ResultEnvelope<JsonElement?> ConvertEnvelope(ResultEnvelope<JsonNode?> envelope)
        {
            if (envelope.Ok)
                return ResultEnvelope<JsonElement?>.Success(ToNullableElement(envelope.Data));

            return ResultEnvelope<JsonElement?>.Failure(
                envelope.Error?.Code ?? "unknown_error",
                envelope.Error?.Message ?? "unknown error",
                CloneElement(envelope.Error?.Details));
        }

        private static ResultEnvelope<JsonElement?> CloneEnvelope(ResultEnvelope<JsonElement?> envelope)
        {
            if (envelope.Ok)
                return ResultEnvelope<JsonElement?>.Success(CloneElement(envelope.Data));

            return ResultEnvelope<JsonElement?>.Failure(
                envelope.Error?.Code ?? "unknown_error",
                envelope.Error?.Message ?? "unknown error",
                CloneElement(envelope.Error?.Details));
        }
    }

    private sealed class StubChatClient(string response) : IChatClient
    {
        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
            Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, response)));

        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
            AsyncEnumerable.Empty<ChatResponseUpdate>();

        public object? GetService(Type serviceType, object? serviceKey = null) => null;

        public void Dispose()
        {
        }
    }

    private sealed class SequenceChatClient(params string[] responses) : IChatClient
    {
        private readonly Queue<string> _responses = new(responses);

        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
            Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, _responses.Dequeue())));

        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
            AsyncEnumerable.Empty<ChatResponseUpdate>();

        public object? GetService(Type serviceType, object? serviceKey = null) => null;

        public void Dispose()
        {
        }
    }

    private static GoalAction ClassifyOutcome(ResultEnvelope<JsonElement?> result)
    {
        if (result.Ok)
            return GoalAction.Done;

        return string.Equals(result.Error?.Code, "ask_user", StringComparison.Ordinal)
            ? GoalAction.AskUser
            : GoalAction.Replan;
    }

    private static string SerializeJson(JsonElement? element) =>
        element is null ? "null" : JsonSerializer.Serialize(element.Value, new JsonSerializerOptions { WriteIndented = true });

    private static string? GetStringProperty(JsonElement? element, string propertyName) =>
        TryGetProperty(element, propertyName, out var property) && property.ValueKind == JsonValueKind.String
            ? property.GetString()
            : null;

    private static bool HasProperty(JsonElement? element, string propertyName) =>
        TryGetProperty(element, propertyName, out _);

    private static bool PropertyIsArray(JsonElement? element, string propertyName) =>
        TryGetProperty(element, propertyName, out var property) && property.ValueKind == JsonValueKind.Array;

    private static bool TryGetProperty(JsonElement? element, string propertyName, out JsonElement property)
    {
        if (element is { ValueKind: JsonValueKind.Object } objectElement && objectElement.TryGetProperty(propertyName, out property))
            return true;

        property = default;
        return false;
    }

    private static JsonElement? ToNullableElement(JsonNode? node) =>
        node is null ? null : JsonSerializer.SerializeToElement(node);

    private static JsonElement? CloneElement(JsonElement? element) =>
        element?.Clone();
}
