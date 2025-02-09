using System.Globalization;
using System.Windows.Controls;
using System.Windows.Data;

namespace InkMARC.Label
{
    public class NumericValidationRule : ValidationRule
    {
        public override ValidationResult Validate(object value, CultureInfo cultureInfo)
        {
            string input = value as string;

            if (string.IsNullOrWhiteSpace(input))
                return new ValidationResult(false, "Value cannot be empty.");

            if (float.TryParse(input, NumberStyles.Float, cultureInfo, out _))
                return ValidationResult.ValidResult;

            return new ValidationResult(false, "Please enter a valid number.");
        }
    }

    public class StringToFloatConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value?.ToString();
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is string input && float.TryParse(input, NumberStyles.Float, culture, out var result))
            {
                return result;
            }

            return 0f; // Default value in case of invalid input
        }
    }
}
