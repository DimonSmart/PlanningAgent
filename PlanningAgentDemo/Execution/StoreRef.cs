using System.Text.RegularExpressions;

namespace PlanningAgentDemo.Execution;

public sealed record StoreRef(string Key, string? JsonPath, bool IsMulti)
{
    private static readonly Regex KeyRegex = new("^[A-Za-z0-9_.\\[\\]-]+$", RegexOptions.Compiled);

    public static bool TryParse(string value, out StoreRef storeRef)
    {
        storeRef = default!;
        if (string.IsNullOrWhiteSpace(value) || value.Contains(' '))
        {
            return false;
        }

        var trimmed = value.Trim();
        var isMulti = trimmed.EndsWith("[]", StringComparison.Ordinal);
        if (isMulti)
        {
            trimmed = trimmed[..^2];
        }

        var separatorIndex = trimmed.IndexOf(':');
        var key = separatorIndex >= 0 ? trimmed[..separatorIndex] : trimmed;
        var jsonPath = separatorIndex >= 0 ? trimmed[(separatorIndex + 1)..] : null;

        if (string.IsNullOrWhiteSpace(key) || !KeyRegex.IsMatch(key))
        {
            return false;
        }

        if (jsonPath is not null && !jsonPath.StartsWith("$", StringComparison.Ordinal))
        {
            return false;
        }

        storeRef = new StoreRef(key, jsonPath, isMulti);
        return true;
    }
}
