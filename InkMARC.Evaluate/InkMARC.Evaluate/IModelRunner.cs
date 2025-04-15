using Microsoft.Maui.Graphics;
using IImage = Microsoft.Maui.Graphics.IImage;

namespace InkMARC.Evaluate
{
    public interface IModelRunner : IDisposable
    {
        float Predict(IImage bitmap);
    }
}
