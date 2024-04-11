using OcuInk.Models.Extensions;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;

namespace OcuInk.Models.Primatives
{
    /// <summary>
    /// Represents an ink color with red, green, blue, and alpha channels.
    /// </summary>
    [DebuggerDisplay("Red={Red}, Green={Green}, Blue={Blue}, Alpha={Alpha}")]
    public struct InkColor
    {
        /// <summary>
        /// The red channel value of the ink color.
        /// </summary>
        public readonly float Red;

        /// <summary>
        /// The green channel value of the ink color.
        /// </summary>
        public readonly float Green;

        /// <summary>
        /// The blue channel value of the ink color.
        /// </summary>
        public readonly float Blue;

        /// <summary>
        /// The alpha channel value of the ink color.
        /// </summary>
        public readonly float Alpha = 1;

        /// <summary>
        /// Initializes a new instance of the <see cref="InkColor"/> struct with default black color.
        /// </summary>
        public InkColor()
        {
            Red = Green = Blue = 0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InkColor"/> struct with the specified gray value.
        /// </summary>
        /// <param name="gray">The gray value for all color channels.</param>
        public InkColor(float gray)
        {
            Red = Green = Blue = gray.Clamp(0, 1);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InkColor"/> struct with the specified red, green, and blue values.
        /// </summary>
        /// <param name="red">The red channel value.</param>
        /// <param name="green">The green channel value.</param>
        /// <param name="blue">The blue channel value.</param>
        public InkColor(float red, float green, float blue)
        {
            Red = red.Clamp(0, 1);
            Green = green.Clamp(0, 1);
            Blue = blue.Clamp(0, 1);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InkColor"/> struct with the specified red, green, blue, and alpha values.
        /// </summary>
        /// <param name="red">The red channel value.</param>
        /// <param name="green">The green channel value.</param>
        /// <param name="blue">The blue channel value.</param>
        /// <param name="alpha">The alpha channel value.</param>
        public InkColor(float red, float green, float blue, float alpha)
        {
            Red = red.Clamp(0, 1);
            Green = green.Clamp(0, 1);
            Blue = blue.Clamp(0, 1);
            Alpha = alpha.Clamp(0, 1);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InkColor"/> struct with the specified red, green, and blue values in byte format.
        /// </summary>
        /// <param name="red">The red channel value in byte format.</param>
        /// <param name="green">The green channel value in byte format.</param>
        /// <param name="blue">The blue channel value in byte format.</param>
        public InkColor(byte red, byte green, byte blue)
        {
            Red = (red / 255f).Clamp(0, 1);
            Green = (green / 255f).Clamp(0, 1);
            Blue = (blue / 255f).Clamp(0, 1);
            Alpha = 1.0f;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InkColor"/> struct with the specified red, green, blue, and alpha values in byte format.
        /// </summary>
        /// <param name="red">The red channel value in byte format.</param>
        /// <param name="green">The green channel value in byte format.</param>
        /// <param name="blue">The blue channel value in byte format.</param>
        /// <param name="alpha">The alpha channel value in byte format.</param>
        public InkColor(byte red, byte green, byte blue, byte alpha)
        {
            Red = (red / 255f).Clamp(0, 1);
            Green = (green / 255f).Clamp(0, 1);
            Blue = (blue / 255f).Clamp(0, 1);
            Alpha = (alpha / 255f).Clamp(0, 1);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InkColor"/> struct with the specified red, green, and blue values in integer format.
        /// </summary>
        /// <param name="red">The red channel value in integer format.</param>
        /// <param name="green">The green channel value in integer format.</param>
        /// <param name="blue">The blue channel value in integer format.</param>
        public InkColor(int red, int green, int blue)
        {
            Red = (red / 255f).Clamp(0, 1);
            Green = (green / 255f).Clamp(0, 1);
            Blue = (blue / 255f).Clamp(0, 1);
            Alpha = 1.0f;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InkColor"/> struct with the specified red, green, blue, and alpha values in integer format.
        /// </summary>
        /// <param name="red">The red channel value in integer format.</param>
        /// <param name="green">The green channel value in integer format.</param>
        /// <param name="blue">The blue channel value in integer format.</param>
        /// <param name="alpha">The alpha channel value in integer format.</param>
        public InkColor(int red, int green, int blue, int alpha)
        {
            Red = (red / 255f).Clamp(0, 1);
            Green = (green / 255f).Clamp(0, 1);
            Blue = (blue / 255f).Clamp(0, 1);
            Alpha = (alpha / 255f).Clamp(0, 1);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InkColor"/> struct with the specified color vector.
        /// </summary>
        /// <param name="color">The color vector containing red, green, blue, and alpha values.</param>
        public InkColor(Vector4 color)
        {
            Red = color.X.Clamp(0, 1);
            Green = color.Y.Clamp(0, 1);
            Blue = color.Z.Clamp(0, 1);
            Alpha = color.W.Clamp(0, 1);
        }

        /// <summary>
        /// Returns a string representation of the <see cref="InkColor"/>.
        /// </summary>
        /// <returns>A string representation of the <see cref="InkColor"/>.</returns>
        public override string ToString()
        {
            return $"[InkColor: Red={Red}, Green={Green}, Blue={Blue}, Alpha={Alpha}]";
        }

        /// <summary>
        /// Returns the hash code for the <see cref="InkColor"/>.
        /// </summary>
        /// <returns>The hash code for the <see cref="InkColor"/>.</returns>
        public override int GetHashCode()
        {
            unchecked
            {
                int hashcode = Red.GetHashCode();
                hashcode = (hashcode * 397) ^ Green.GetHashCode();
                hashcode = (hashcode * 397) ^ Blue.GetHashCode();
                hashcode = (hashcode * 397) ^ Alpha.GetHashCode();
                return hashcode;
            }
        }

        /// <summary>
        /// Determines whether the specified object is equal to the current <see cref="InkColor"/>.
        /// </summary>
        /// <param name="obj">The object to compare with the current <see cref="InkColor"/>.</param>
        /// <returns>true if the specified object is equal to the current <see cref="InkColor"/>; otherwise, false.</returns>
        public override bool Equals(object obj)
        {
            if (obj is InkColor other)
                return other.Red == this.Red && other.Green == this.Green && other.Blue == this.Blue && other.Alpha == this.Alpha;

            return base.Equals(obj);
        }

        /// <summary>
        /// Determines whether the specified <see cref="InkColor"/> is equal to the current <see cref="InkColor"/>.
        /// </summary>
        /// <param name="other">The <see cref="InkColor"/> to compare with the current <see cref="InkColor"/>.</param>
        /// <returns>true if the specified <see cref="InkColor"/> is equal to the current <see cref="InkColor"/>; otherwise, false.</returns>
        public bool Equals(InkColor other)
        {
            return other.Red == this.Red && other.Green == this.Green && other.Blue == this.Blue && other.Alpha == this.Alpha;
        }

        /// <summary>
        /// Determines whether two <see cref="InkColor"/> instances are equal.
        /// </summary>
        /// <param name="left">The first <see cref="InkColor"/> to compare.</param>
        /// <param name="right">The second <see cref="InkColor"/> to compare.</param>
        /// <returns>true if the two <see cref="InkColor"/> instances are equal; otherwise, false.</returns>
        public static bool operator ==(InkColor left, InkColor right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Determines whether two <see cref="InkColor"/> instances are not equal.
        /// </summary>
        /// <param name="left">The first <see cref="InkColor"/> to compare.</param>
        /// <param name="right">The second <see cref="InkColor"/> to compare.</param>
        /// <returns>true if the two <see cref="InkColor"/> instances are not equal; otherwise, false.</returns>
        public static bool operator !=(InkColor left, InkColor right)
        {
            return !(left == right);
        }
    }
}
