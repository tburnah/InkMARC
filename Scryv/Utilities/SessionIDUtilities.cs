namespace Scryv.Utilities
{
    /// <summary>
    /// Provides utility methods for generating and manipulating session IDs.
    /// </summary>
    public static class SessionIDUtilities
    {
        private static Random random = new Random();

        /// <summary>
        /// Generates a random session ID key of the specified length.
        /// </summary>
        /// <param name="length">The length of the session ID key.</param>
        /// <returns>A randomly generated session ID key.</returns>
        public static string GenerateKey(int length)
        {
            const string chars = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789abcdefghjklmnpqrstuvwxyz";
            var key = new char[length];

            for (int i = 0; i < length; i++)
            {
                key[i] = chars[random.Next(chars.Length)];
            }

            return new String(key);
        }

        /// <summary>
        /// Generates a unique session ID by inserting a space character in the middle of the generated key.
        /// </summary>
        /// <returns>A unique session ID.</returns>
        public static string GetUniqueSessionID()
        {
            return GenerateKey(6).Insert(3, " ");
        }
    }
}
