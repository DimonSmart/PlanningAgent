using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace PlanningAgentDemo.Common;

public sealed record ErrorInfo(
    [property: JsonPropertyName("code")] string Code,
    [property: JsonPropertyName("message")] string Message,
    [property: JsonPropertyName("details")] JsonObject? Details = null);

public sealed record ResultEnvelope<T>(
    [property: JsonPropertyName("ok")] bool Ok,
    [property: JsonPropertyName("data")] T? Data,
    [property: JsonPropertyName("error")] ErrorInfo? Error)
{
    public static ResultEnvelope<T> Success(T data) => new(true, data, null);

    public static ResultEnvelope<T> Failure(string code, string message, JsonObject? details = null) =>
        new(false, default, new ErrorInfo(code, message, details));
}
