using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InkMARC.Utilities
{
    public class ColorValueConverter : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value is string text && parameter is string comparison)
            {
                if (text == comparison)
                {
                    if (Application.Current.Resources.TryGetValue("ButtonColor", out var colorResource))
                    {
                        if (colorResource is Color color)
                        {
                            return color;
                        }
                    }
                    return Colors.Blue;
                }
                else
                {
                    if (Application.Current.Resources.TryGetValue("White", out var colorResource))
                    {
                        if (colorResource is Color color)
                        {
                            return color;
                        }
                    }
                    return Colors.White;
                }
            }
            return Colors.White;
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    public class ColorTextValueConverter : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value is string text && parameter is string comparison)
            {
                if (text == comparison)
                {
                    if (Application.Current.Resources.TryGetValue("White", out var colorResource))
                    {
                        if (colorResource is Color color)
                        {
                            return color;
                        }
                    }
                    return Colors.White;
                }
                else
                {
                    if (Application.Current.Resources.TryGetValue("PrimaryDarkText", out var colorResource))
                    {
                        if (colorResource is Color color)
                        {
                            return color;
                        }
                    }
                    return Colors.Black;
                }
            }
            return Colors.White;
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
