namespace InkMARC.Models.Primatives
{
    /// <summary>
    /// Represents the size of an object in the InkMARC system.
    /// </summary>
    public struct InkMARCSize
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
        /// Represents an empty InkMARCSize instance.
        /// </summary>
        public static readonly InkMARCSize Empty = new InkMARCSize();

        /// <summary>
        /// Initializes a new instance of the InkMARCSize struct with the specified width and height.
        /// </summary>
        /// <param name="width">The width of the object.</param>
        /// <param name="height">The height of the object.</param>
        public InkMARCSize(float width, float height)
        {
            Width = width;
            Height = height;
        }

        /// <summary>
        /// Returns a string representation of the InkMARCSize instance.
        /// </summary>
        /// <returns>A string that represents the InkMARCSize instance.</returns>
        public override string ToString()
        {
            return $"{{Width={Width} Height={Height}}}";
        }

        /// <summary>
        /// Determines whether the specified object is equal to the current InkMARCSize instance.
        /// </summary>
        /// <param name="obj">The object to compare with the current InkMARCSize instance.</param>
        /// <returns>true if the specified object is equal to the current InkMARCSize instance; otherwise, false.</returns>
        public override bool Equals(object obj)
        {
            if (!(obj is InkMARCSize inkMARCSize))
                return false;

            return (inkMARCSize.Width == Width) && (inkMARCSize.Height == Height);
        }

        /// <summary>
        /// Returns the hash code for the current InkMARCSize instance.
        /// </summary>
        /// <returns>A 32-bit signed integer hash code.</returns>
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }
}
