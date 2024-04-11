namespace OcuInk.Models.Primatives
{
    /// <summary>
    /// Represents the size of an object in the OcuInk system.
    /// </summary>
    public struct OcuInkSize
    {
        /// <summary>
        /// Gets or sets the width of the object.
        /// </summary>
        public float Width { get; set; }

        /// <summary>
        /// Gets or sets the height of the object.
        /// </summary>
        public float Height { get; set; }

        /// <summary>
        /// Represents an empty OcuInkSize instance.
        /// </summary>
        public static readonly OcuInkSize Empty = new OcuInkSize();

        /// <summary>
        /// Initializes a new instance of the OcuInkSize struct with the specified width and height.
        /// </summary>
        /// <param name="width">The width of the object.</param>
        /// <param name="height">The height of the object.</param>
        public OcuInkSize(float width, float height)
        {
            Width = width;
            Height = height;
        }

        /// <summary>
        /// Returns a string representation of the OcuInkSize instance.
        /// </summary>
        /// <returns>A string that represents the OcuInkSize instance.</returns>
        public override string ToString()
        {
            return $"{{Width={Width} Height={Height}}}";
        }

        /// <summary>
        /// Determines whether the specified object is equal to the current OcuInkSize instance.
        /// </summary>
        /// <param name="obj">The object to compare with the current OcuInkSize instance.</param>
        /// <returns>true if the specified object is equal to the current OcuInkSize instance; otherwise, false.</returns>
        public override bool Equals(object obj)
        {
            if (!(obj is OcuInkSize ocuInkSize))
                return false;

            return (ocuInkSize.Width == Width) && (ocuInkSize.Height == Height);
        }

        /// <summary>
        /// Returns the hash code for the current OcuInkSize instance.
        /// </summary>
        /// <returns>A 32-bit signed integer hash code.</returns>
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }
}
