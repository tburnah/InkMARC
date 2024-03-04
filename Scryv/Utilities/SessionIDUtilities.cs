using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Utilities
{
    public static class SessionIDUtilities
    {
        private static Random random = new Random();

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

        public static string GetUniqueSessionID()
        {
            return GenerateKey(6).Insert(3, " ");
        }
    }
}
