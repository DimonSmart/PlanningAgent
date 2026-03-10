using System.Net;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;
using OpenAI;
using PlanningAgentDemo.Agents;
using PlanningAgentDemo.Common;
using PlanningAgentDemo.Execution;
using PlanningAgentDemo.Orchestration;
using PlanningAgentDemo.Planning;
using PlanningAgentDemo.Tools;
using PlanningAgentDemo.Verification;
using System.ClientModel;
using Xunit.Abstractions;

namespace PlanningAgentDemo.Tests;

public sealed class TestLogger(ITestOutputHelper output) : IExecutionLogger
{
    public void Log(string message) => output.WriteLine(message);
}

public sealed class MockSearchTool : ITool
{
    public string Name => "search";

    public ToolPlannerMetadata PlannerMetadata => new(
        "search",
        "Search the web and return candidate page URLs.",
        JsonNode.Parse(@"{""type"":""object"",""properties"":{""query"":{""type"":""string""},""limit"":{""type"":""number""}},""required"":[""query""]}")!.AsObject(),
        JsonNode.Parse(@"{""type"":""array"",""items"":{""type"":""string""}}")!.AsObject(),
        [],
        []);

    public Task<ResultEnvelope<JsonElement?>> ExecuteAsync(JsonElement input, CancellationToken cancellationToken = default) =>
        Task.FromResult(ResultEnvelope<JsonElement?>.Success(JsonSerializer.SerializeToElement(new[]
        {
            "https://example.com/item-a",
            "https://example.com/item-b"
        })));
}

public sealed class MockDownloadTool : ITool
{
    public string Name => "download";

    public ToolPlannerMetadata PlannerMetadata => new(
        "download",
        "Download a single page by URL and return its title and body text.",
        JsonNode.Parse(@"{""type"":""object"",""properties"":{""url"":{""type"":""string""}},""required"":[""url""]}")!.AsObject(),
        JsonNode.Parse(@"{""type"":""object"",""properties"":{""url"":{""type"":""string""},""title"":{""type"":""string""},""body"":{""type"":""string""}},""required"":[""url"",""title"",""body""]}")!.AsObject(),
        [],
        []);

    public Task<ResultEnvelope<JsonElement?>> ExecuteAsync(JsonElement input, CancellationToken cancellationToken = default)
    {
        var url = TryGetStringProperty(input, "url");
        if (string.IsNullOrWhiteSpace(url))
            return Task.FromResult(ResultEnvelope<JsonElement?>.Failure("invalid_input", "Download URL is required."));

        var payload = url.Contains("item-a", StringComparison.OrdinalIgnoreCase)
            ? new
            {
                url,
                title = "RoboClean A1 Max review",
                body = "RoboClean A1 Max is a popular robot vacuum cleaner with 7000 Pa suction power, up to 180 minutes of battery runtime, a 0.5 L dustbin, LiDAR navigation, and a list price of $799."
            }
            : new
            {
                url,
                title = "HomeSweep S5 review",
                body = "HomeSweep S5 is a popular robot vacuum cleaner with 5000 Pa suction power, up to 140 minutes of battery runtime, a 0.4 L dustbin, vSLAM navigation, and a list price of $649."
            };

        return Task.FromResult(ResultEnvelope<JsonElement?>.Success(JsonSerializer.SerializeToElement(payload)));
    }

    private static string? TryGetStringProperty(JsonElement element, string propertyName) =>
        element.ValueKind == JsonValueKind.Object
        && element.TryGetProperty(propertyName, out var property)
        && property.ValueKind == JsonValueKind.String
            ? property.GetString()
            : null;
}

public sealed class RealWebSearchTool : ITool
{
    private const int DefaultLimit = 4;
    private const int MaxLimit = 6;
    private static readonly Regex AbsoluteLinkRegex = new("href=\"(https://[^\"#]+)\"", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly string[] IgnoredHosts =
    [
        "search.brave.com",
        "cdn.search.brave.com",
        "imgs.search.brave.com",
        "tiles.search.brave.com"
    ];

    public string Name => "search";

    public ToolPlannerMetadata PlannerMetadata => new(
        "search",
        "Search the web and return candidate page URLs.",
        JsonNode.Parse(@"{""type"":""object"",""properties"":{""query"":{""type"":""string""},""limit"":{""type"":""number""}},""required"":[""query""]}")!.AsObject(),
        JsonNode.Parse(@"{""type"":""array"",""items"":{""type"":""string""}}")!.AsObject(),
        ["web", "search"],
        ["auto"]);

    public async Task<ResultEnvelope<JsonElement?>> ExecuteAsync(JsonElement input, CancellationToken cancellationToken = default)
    {
        var query = TryGetStringProperty(input, "query");
        if (string.IsNullOrWhiteSpace(query))
            return ResultEnvelope<JsonElement?>.Failure("invalid_input", "Search query is required.");

        var limit = input.TryGetProperty("limit", out var limitElement) && limitElement.TryGetInt32(out var parsedLimit)
            ? Math.Clamp(parsedLimit, 1, MaxLimit)
            : DefaultLimit;

        using var client = CreateClient();
        var url = $"https://search.brave.com/search?q={UrlEncoder.Default.Encode(query)}";
        var html = await client.GetStringAsync(url, cancellationToken);

        var queryTerms = query
            .Split([' ', '.', ',', ':', ';', '-', '_', '/', '\\', '(', ')'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(term => term.Length >= 3)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        var links = AbsoluteLinkRegex
            .Matches(html)
            .Select(match => match.Groups[1].Value)
            .Where(link => Uri.TryCreate(link, UriKind.Absolute, out var uri)
                && (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps)
                && !IgnoredHosts.Contains(uri.Host, StringComparer.OrdinalIgnoreCase))
            .OrderByDescending(link => ScoreCandidate(link, queryTerms))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(limit)
            .ToArray();

        if (links.Length == 0)
            return ResultEnvelope<JsonElement?>.Failure("search_failed", "Search returned no candidate URLs.");

        return ResultEnvelope<JsonElement?>.Success(JsonSerializer.SerializeToElement(links));
    }

    private static HttpClient CreateClient()
    {
        var client = new HttpClient();
        client.DefaultRequestHeaders.UserAgent.ParseAdd("Mozilla/5.0 (compatible; PlanningAgentDemoTests/1.0)");
        return client;
    }

    private static int ScoreCandidate(string link, IReadOnlyCollection<string> queryTerms)
    {
        var score = 0;
        foreach (var term in queryTerms)
        {
            if (link.Contains(term, StringComparison.OrdinalIgnoreCase))
                score++;
        }

        if (link.Contains("github.com", StringComparison.OrdinalIgnoreCase))
            score += 2;

        return score;
    }

    private static string? TryGetStringProperty(JsonElement element, string propertyName) =>
        element.ValueKind == JsonValueKind.Object
        && element.TryGetProperty(propertyName, out var property)
        && property.ValueKind == JsonValueKind.String
            ? property.GetString()
            : null;
}

public sealed class RealWebDownloadTool : ITool
{
    private const int MaxBodyLength = 12000;
    private static readonly Regex TitleRegex = new("<title[^>]*>(.*?)</title>", RegexOptions.IgnoreCase | RegexOptions.Singleline | RegexOptions.Compiled);
    private static readonly Regex ScriptRegex = new("<script\\b[^<]*(?:(?!</script>)<[^<]*)*</script>", RegexOptions.IgnoreCase | RegexOptions.Singleline | RegexOptions.Compiled);
    private static readonly Regex StyleRegex = new("<style\\b[^<]*(?:(?!</style>)<[^<]*)*</style>", RegexOptions.IgnoreCase | RegexOptions.Singleline | RegexOptions.Compiled);
    private static readonly Regex TagRegex = new("<[^>]+>", RegexOptions.Singleline | RegexOptions.Compiled);
    private static readonly Regex WhitespaceRegex = new(@"\s+", RegexOptions.Compiled);

    public string Name => "download";

    public ToolPlannerMetadata PlannerMetadata => new(
        "download",
        "Download a single page by URL and return its title and body text.",
        JsonNode.Parse(@"{""type"":""object"",""properties"":{""url"":{""type"":""string""}},""required"":[""url""]}")!.AsObject(),
        JsonNode.Parse(@"{""type"":""object"",""properties"":{""url"":{""type"":""string""},""title"":{""type"":""string""},""body"":{""type"":""string""}},""required"":[""url"",""title"",""body""]}")!.AsObject(),
        ["web", "download"],
        ["auto", "each"]);

    public async Task<ResultEnvelope<JsonElement?>> ExecuteAsync(JsonElement input, CancellationToken cancellationToken = default)
    {
        var url = TryGetStringProperty(input, "url");
        if (string.IsNullOrWhiteSpace(url))
            return ResultEnvelope<JsonElement?>.Failure("invalid_input", "Download URL is required.");
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) || (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps))
            return ResultEnvelope<JsonElement?>.Failure("invalid_input", "Download URL must be an absolute HTTP or HTTPS URL.");

        using var client = CreateClient();
        var html = await client.GetStringAsync(uri, cancellationToken);
        var title = ExtractTitle(html, uri);
        var body = ExtractBody(html);

        return ResultEnvelope<JsonElement?>.Success(JsonSerializer.SerializeToElement(new
        {
            url = uri.ToString(),
            title,
            body
        }));
    }

    private static HttpClient CreateClient()
    {
        var client = new HttpClient();
        client.DefaultRequestHeaders.UserAgent.ParseAdd("Mozilla/5.0 (compatible; PlanningAgentDemoTests/1.0)");
        return client;
    }

    private static string ExtractTitle(string html, Uri uri)
    {
        var match = TitleRegex.Match(html);
        if (!match.Success)
            return uri.Host;

        return WebUtility.HtmlDecode(match.Groups[1].Value).Trim();
    }

    private static string ExtractBody(string html)
    {
        var noScripts = ScriptRegex.Replace(html, " ");
        var noStyles = StyleRegex.Replace(noScripts, " ");
        var noTags = TagRegex.Replace(noStyles, " ");
        var decoded = WebUtility.HtmlDecode(noTags);
        var text = WhitespaceRegex.Replace(decoded, " ").Trim();

        return text.Length <= MaxBodyLength
            ? text
            : text[..MaxBodyLength];
    }

    private static string? TryGetStringProperty(JsonElement element, string propertyName) =>
        element.ValueKind == JsonValueKind.Object
        && element.TryGetProperty(propertyName, out var property)
        && property.ValueKind == JsonValueKind.String
            ? property.GetString()
            : null;
}

public class PipelineTests(ITestOutputHelper output)
{
    private const string DevModel = "gpt-oss:120b-cloud";

    [Fact]
    public async Task FullPipeline_PlannerAndOrchestrator_ReturnsSystemOutcome()
    {
        const string userQuery = "I'm looking for a good robot vacuum cleaner. Can you find two popular models, check their specs, and tell me which one is better?";
        await RunFullPipelineAsync(userQuery, new ToolRegistry([new MockSearchTool(), new MockDownloadTool()]));
    }

    [Fact]
    public async Task FullPipeline_PlannerAndOrchestrator_WithRealWebSearchAndDownload_ReturnsSystemOutcome()
    {
        const string userQuery = "Compare Markdig and CommonMark.NET using their GitHub or documentation pages, and tell me which one is better for a small .NET app.";
        await RunFullPipelineAsync(userQuery, new ToolRegistry([new RealWebSearchTool(), new RealWebDownloadTool()]));
    }

    private async Task RunFullPipelineAsync(string userQuery, IToolRegistry tools)
    {
        var chatClient = BuildChatClient();
        var logger = new TestLogger(output);
        var answerAsserter = new CachedLlmAnswerAsserter(chatClient, DevModel);
        var planner = new LlmPlanner(chatClient, tools, logger);
        var replanner = new LlmReplanner(chatClient, tools, logger);
        var runner = new AgentStepRunner(chatClient);
        var executor = new PlanExecutor(tools, runner, logger);
        var orchestrator = new PlanningOrchestrator(
            planner,
            executor,
            new GoalVerifier(askUserEnabled: true),
            logger,
            maxAttempts: 3,
            replanner: replanner,
            finalAnswerVerifier: new LlmFinalAnswerVerifier(chatClient));

        output.WriteLine($"User query: {userQuery}");

        var result = await orchestrator.RunAsync(userQuery);

        if (result.Ok)
            output.WriteLine($"\n=== FINAL ANSWER ===\n{SerializeJson(result.Data)}");
        else
            output.WriteLine($"\n=== OUTCOME DETAILS ===\ncode={result.Error?.Code}\nmessage={result.Error?.Message}\ndetails={SerializeJson(result.Error?.Details)}");

        Assert.True(
            result.Ok,
            $"Expected orchestrator to return a final answer, but got code={result.Error?.Code}, message={result.Error?.Message}, details={SerializeJson(result.Error?.Details)}");

        await AssertAnswersQuestionAsync(answerAsserter, userQuery, result.Data);
    }

    private static IChatClient BuildChatClient()
    {
        var clientOptions = new OpenAIClientOptions
        {
            Endpoint = new Uri("http://localhost:11434/v1/")
        };

        return new OpenAIClient(new ApiKeyCredential("ollama"), clientOptions)
            .GetChatClient(DevModel)
            .AsIChatClient();
    }

    private async Task AssertAnswersQuestionAsync(
        CachedLlmAnswerAsserter answerAsserter,
        string userQuery,
        JsonElement? answer)
    {
        Assert.NotNull(answer);

        var verdict = await answerAsserter.EvaluateAsync(userQuery, answer);
        output.WriteLine($"LLM asserter verdict: isAnswer={verdict.IsAnswer} cache={verdict.FromCache} comment={verdict.Comment}");
        Assert.True(verdict.IsAnswer, verdict.Comment);
    }

    private static string SerializeJson(JsonElement? element) =>
        element is null
            ? "null"
            : JsonSerializer.Serialize(element.Value, new JsonSerializerOptions { WriteIndented = true });
}
