using System.Text.Json.Nodes;
using PlanningAgentDemo.Tools;

namespace PlanningAgentDemo.Planning;

public interface IPlannerToolMetadataSource
{
    IReadOnlyCollection<ToolPlannerMetadata> GetMetadata();
}

public sealed class ToolRegistryMetadataSource(IToolRegistry toolRegistry) : IPlannerToolMetadataSource
{
    public IReadOnlyCollection<ToolPlannerMetadata> GetMetadata() => toolRegistry.ListPlannerMetadata();
}

public sealed class StaticPlannerToolMetadataSource(IEnumerable<ToolPlannerMetadata> metadata) : IPlannerToolMetadataSource
{
    private readonly IReadOnlyCollection<ToolPlannerMetadata> _metadata = metadata.ToList();

    public IReadOnlyCollection<ToolPlannerMetadata> GetMetadata() => _metadata;
}

public sealed class CompositePlannerToolMetadataSource(IEnumerable<IPlannerToolMetadataSource> sources) : IPlannerToolMetadataSource
{
    private readonly IReadOnlyCollection<IPlannerToolMetadataSource> _sources = sources.ToList();

    public IReadOnlyCollection<ToolPlannerMetadata> GetMetadata()
    {
        return _sources
            .SelectMany(s => s.GetMetadata())
            .GroupBy(m => m.Name, StringComparer.Ordinal)
            .Select(g => g.Last())
            .OrderBy(x => x.Name, StringComparer.Ordinal)
            .ToList();
    }
}

public static class PlannerBuiltInToolMetadata
{
    public static ToolPlannerMetadata LlmChat() => new(
        Name: "llm.chat",
        Description: "Agent step for reasoning and synthesis over collected context. Returns planner-defined JSON shape in envelope.data.",
        InputSchema: new JsonObject
        {
            ["inputs"] = new JsonObject
            {
                ["*"] = "literal|StoreRef"
            },
            ["agent"] = new JsonObject
            {
                ["systemPrompt"] = "string",
                ["userPrompt"] = "string",
                ["context"] = new JsonObject
                {
                    ["*"] = "StoreRef"
                },
                ["allowedTools"] = new JsonArray("string"),
                ["responseSchemaHint"] = "string|null"
            }
        },
        OutputSchema: new JsonObject
        {
            ["$"] = "json"
        },
        Tags: ["llm", "synthesis", "agent"],
        SupportedConsume: ["auto", "batch"]);
}
