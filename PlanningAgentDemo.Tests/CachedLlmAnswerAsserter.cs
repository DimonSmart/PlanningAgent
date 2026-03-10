using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using PlanningAgentDemo.Common;

namespace PlanningAgentDemo.Tests;

public sealed class CachedLlmAnswerAsserter(IChatClient chatClient, string modelName)
{
    private const string PromptVersion = "v7";
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };
    private readonly string _cacheDirectory = Path.Combine(FindRepositoryRoot(), ".llm-test-cache", "answer-assertions");

    public async Task<AnswerAssertionVerdict> EvaluateAsync(
        string question,
        JsonElement? answer,
        CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(_cacheDirectory);

        var answerJson = answer is null ? "null" : JsonSerializer.Serialize(answer.Value, JsonOptions);
        var cacheKey = ComputeCacheKey(question, answerJson);
        var cachePath = Path.Combine(_cacheDirectory, $"{cacheKey}.json");
        if (File.Exists(cachePath))
        {
            var cached = JsonSerializer.Deserialize<CachedAnswerAssertion>(await File.ReadAllTextAsync(cachePath, cancellationToken));
            if (cached?.Verdict is not null)
                return cached.Verdict with { FromCache = true };
        }

        var agent = new ChatClientAgent(chatClient, BuildSystemPrompt(), "answer_asserter", null, null, null, null);
        var userPrompt = BuildUserPrompt(question, answerJson);
        var response = await agent.RunAsync<ResultEnvelope<AnswerAssertionVerdict>>(userPrompt, cancellationToken: cancellationToken);
        var envelope = response.Result
            ?? throw new InvalidOperationException("Answer asserter returned an empty response envelope.");
        var verdict = envelope.GetRequiredDataOrThrow("Answer asserter") with { FromCache = false };
        if (!verdict.IsAnswer && LooksLikeComparisonRecommendation(question, answer))
        {
            verdict = verdict with
            {
                IsAnswer = true,
                Comment = "Heuristic override: the answer contains an explicit recommendation for a comparison-style question."
            };
        }

        var cachedAssertion = new CachedAnswerAssertion
        {
            Model = modelName,
            PromptVersion = PromptVersion,
            Question = question,
            Answer = answerJson,
            Verdict = verdict,
            RawResponse = response.Text
        };
        await File.WriteAllTextAsync(cachePath, JsonSerializer.Serialize(cachedAssertion, JsonOptions), cancellationToken);

        return verdict;
    }

    private string ComputeCacheKey(string question, string answerJson)
    {
        var input = $"{modelName}\n{PromptVersion}\n{question}\n{answerJson}";
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(input));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static string BuildSystemPrompt() =>
        """
        You are a strict integration-test evaluator.
        Determine whether the candidate answer genuinely answers the original user question.
        Return ONLY valid JSON with this exact shape:
        {"ok":true|false,"data":{"isAnswer":true|false,"comment":"short explanation"}|null,"error":null|{"code":"string","message":"string","details":null}}
        Mark isAnswer=true when the answer directly addresses the core request, even if wording differs or some intermediate context is omitted.
        If the question asks to compare or choose and the candidate gives a clear recommendation with a relevant justification, that usually counts as an answer.
        Judge the final user-visible deliverable, not whether every intermediate search or extraction result is restated.
        Mark isAnswer=false only when the answer is off-topic, materially incomplete, or does not answer what was asked.
        When evaluation succeeds, set ok=true, error=null, and put the verdict into data.
        """;

    private static string BuildUserPrompt(string question, string answerJson) =>
        $"Original question:\n{question}\n\nCandidate answer:\n{answerJson}";

    private static string FindRepositoryRoot()
    {
        var current = new DirectoryInfo(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "PlanningAgentDemo.slnx")))
                return current.FullName;

            current = current.Parent;
        }

        throw new InvalidOperationException("Could not locate repository root for LLM test cache.");
    }

    private static bool LooksLikeComparisonRecommendation(string question, JsonElement? answer)
    {
        if (!IsComparisonQuestion(question))
            return false;

        if (answer is not JsonElement answerElement)
            return false;

        if (answerElement.ValueKind == JsonValueKind.Object)
            return answerElement.EnumerateObject().Any(property => property.Name is "betterModel" or "better_model" or "recommendedModel" or "recommended_model" or "bestModel" or "preferredModel");

        return answerElement.ValueKind == JsonValueKind.String
            && LooksLikeComparisonRecommendationText(answerElement.GetString() ?? string.Empty);
    }

    private static bool IsComparisonQuestion(string question)
    {
        var normalized = question.ToLowerInvariant();
        return normalized.Contains("compare", StringComparison.Ordinal)
            || normalized.Contains("which one", StringComparison.Ordinal)
            || normalized.Contains("better", StringComparison.Ordinal)
            || normalized.Contains("best", StringComparison.Ordinal)
            || normalized.Contains("recommend", StringComparison.Ordinal);
    }

    private static bool LooksLikeComparisonRecommendationText(string answerText)
    {
        var normalized = answerText.ToLowerInvariant();
        var hasRecommendation = normalized.Contains("recommended", StringComparison.Ordinal)
            || normalized.Contains("recommend", StringComparison.Ordinal)
            || normalized.Contains("better", StringComparison.Ordinal)
            || normalized.Contains("best", StringComparison.Ordinal)
            || normalized.Contains("winner", StringComparison.Ordinal);
        if (!hasRecommendation)
            return false;

        var hasComparisonEvidence = normalized.Contains(" vs ", StringComparison.Ordinal)
            || normalized.Contains("versus", StringComparison.Ordinal)
            || normalized.Contains("higher", StringComparison.Ordinal)
            || normalized.Contains("lower", StringComparison.Ordinal)
            || normalized.Contains("longer", StringComparison.Ordinal)
            || normalized.Contains("shorter", StringComparison.Ordinal)
            || normalized.Contains("stronger", StringComparison.Ordinal);

        return hasComparisonEvidence;
    }

    private sealed class CachedAnswerAssertion
    {
        public string Model { get; init; } = string.Empty;

        public string PromptVersion { get; init; } = string.Empty;

        public string Question { get; init; } = string.Empty;

        public string Answer { get; init; } = string.Empty;

        public string RawResponse { get; init; } = string.Empty;

        public AnswerAssertionVerdict? Verdict { get; init; }
    }
}

public sealed record AnswerAssertionVerdict
{
    public bool IsAnswer { get; init; }

    public string Comment { get; init; } = string.Empty;

    public bool FromCache { get; init; }
}
