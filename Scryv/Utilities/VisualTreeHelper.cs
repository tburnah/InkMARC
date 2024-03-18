namespace Scryv.Utilities
{
    /// <summary>
    /// Provides helper methods for working with the visual tree.
    /// </summary>
    public static class VisualTreeHelper
    {
        /// <summary>
        /// Finds the parent of the specified element of type T in the visual tree.
        /// </summary>
        /// <typeparam name="T">The type of the parent element to find.</typeparam>
        /// <param name="element">The element whose parent to find.</param>
        /// <returns>The parent element of type T, or null if not found.</returns>
        public static T FindParentOfType<T>(Element element) where T : Element
        {
            if (element == null)
                return null;

            var parent = element.Parent;
            while (parent != null)
            {
                if (parent is T correctlyTypedParent)
                {
                    return correctlyTypedParent;
                }

                parent = parent.Parent;
            }

            return null;
        }
    }
}
