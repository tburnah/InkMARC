using OcuInk.Models.Interfaces;

namespace OcuInkTrain.Interfaces
{
        public interface IAdvancedDrawingLineRenderer
    {
        /// <summary>
        /// Retrieves a System.IO.Stream containing an image of this line, based on the CommunityToolkit.Maui.Core.IDrawingLine.Points data.
        /// </summary>
        /// <param name="imageSizeWidth">Desired width of the image that is returned.</param>
        /// <param name="imageSizeHeight">Desired height of the image that is returned.</param>
        /// <param name="background">Background of the generated image.</param>
        /// <param name="token">System.Threading.CancellationToken.</param>
        /// <returns>System.Threading.Tasks.ValueTask&lt;Stream&gt; containing the data of the requested image with data that's currently on the CommunityToolkit.Maui.Core.IDrawingLine.</returns>
        ValueTask<Stream> GetImageStream(IAdvancedDrawingLine line, double imageSizeWidth, double imageSizeHeight, Paint background, CancellationToken token = default(CancellationToken));
    }
}
