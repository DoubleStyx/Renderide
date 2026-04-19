namespace SharedTypeGenerator.Analysis;

/// <summary>Shared helpers for IL walk stacks that track the last field name loaded before a pack call.</summary>
internal static class FieldNameStackHelpers
{
    /// <summary>Pops the most recently pushed field name and clears the stack (one field per pack operation or <c>brfalse</c> condition).</summary>
    public static string PopLastFieldAndClear(Stack<string> stack)
    {
        if (stack.Count == 0)
            return "_unknown";

        string last = stack.Pop();
        stack.Clear();
        return last;
    }
}
