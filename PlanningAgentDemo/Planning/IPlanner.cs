namespace PlanningAgentDemo.Planning;

public interface IPlanner
{
    Task<PlanDefinition> CreatePlanAsync(string userQuery, CancellationToken cancellationToken = default);
}

public interface IReplanCapablePlanner : IPlanner
{
    Task<PlanDefinition> ReplanAsync(PlannerReplanRequest request, CancellationToken cancellationToken = default);
}
