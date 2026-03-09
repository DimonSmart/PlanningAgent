namespace PlanningAgentDemo.Agents;

public interface ILlmClient
{
    Task<string> GenerateAsync(string systemPrompt, string userPrompt, CancellationToken cancellationToken = default);
}
