using System.Text.Json.Nodes;

namespace PlanningAgentDemo.Execution;

public enum InputBindingMode
{
    all,
    each
}

public sealed record InputBindingRef(string From, string Path, InputBindingMode Mode)
{
    public bool IsEach => Mode == InputBindingMode.each;
}

public static class InputBindingRefParser
{
    public static bool TryParse(JsonNode? node, out InputBindingRef inputRef)
    {
        inputRef = default!;

        if (node is not JsonObject obj)
        {
            return false;
        }

        if (obj["from"] is not JsonValue fromValue ||
            !fromValue.TryGetValue<string>(out var fromText) ||
            string.IsNullOrWhiteSpace(fromText))
        {
            return false;
        }

        var path = "$";
        if (obj["path"] is JsonValue pathValue &&
            pathValue.TryGetValue<string>(out var pathText) &&
            !string.IsNullOrWhiteSpace(pathText))
        {
            path = pathText.Trim();
        }

        if (!path.StartsWith("$", StringComparison.Ordinal))
        {
            return false;
        }

        var mode = InputBindingMode.all;
        if (obj["mode"] is JsonValue modeValue &&
            modeValue.TryGetValue<string>(out var modeText) &&
            !string.IsNullOrWhiteSpace(modeText))
        {
            if (string.Equals(modeText, "each", StringComparison.OrdinalIgnoreCase))
            {
                mode = InputBindingMode.each;
            }
            else if (string.Equals(modeText, "all", StringComparison.OrdinalIgnoreCase))
            {
                mode = InputBindingMode.all;
            }
            else
            {
                return false;
            }
        }

        inputRef = new InputBindingRef(fromText.Trim(), path, mode);
        return true;
    }
}
