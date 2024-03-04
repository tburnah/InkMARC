using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Utilities
{
    public static class VisualTreeHelper
    {
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
