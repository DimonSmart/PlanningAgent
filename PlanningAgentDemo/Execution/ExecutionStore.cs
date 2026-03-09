using System.Text.Json.Nodes;

namespace PlanningAgentDemo.Execution;

public sealed class ExecutionStore
{
    private readonly Dictionary<string, JsonNode?> _values = new(StringComparer.Ordinal);

    public IReadOnlyCollection<string> Keys => _values.Keys;

    public bool TryGet(string key, out JsonNode? value) => _values.TryGetValue(key, out value);

    public JsonNode? Get(string key) => _values[key];

    public void Set(string key, JsonNode? value)
    {
        _values[key] = value?.DeepClone();
    }

    public void Append(string key, JsonNode? item)
    {
        if (!_values.TryGetValue(key, out var existing) || existing is not JsonArray array)
        {
            array = new JsonArray();
            _values[key] = array;
        }

        if (item is JsonArray arrayInput)
        {
            foreach (var element in arrayInput)
            {
                AppendSingle(key, array, element);
            }

            return;
        }

        AppendSingle(key, array, item);
    }

    public JsonObject ResolveStoreRefOrThrow(StoreRef storeRef)
    {
        if (!TryGet(storeRef.Key, out var value))
        {
            throw new InvalidOperationException($"Store key '{storeRef.Key}' was not found.");
        }

        var selected = storeRef.JsonPath is null ? value : JsonPathEvaluator.Select(value, storeRef.JsonPath);
        var payload = new JsonObject
        {
            ["value"] = selected?.DeepClone()
        };
        return payload;
    }

    private void AppendSingle(string key, JsonArray array, JsonNode? item)
    {
        var clone = item?.DeepClone();
        array.Add(clone);
        var index = array.Count - 1;
        _values[$"{key}[{index}]"] = clone?.DeepClone();
    }
}
