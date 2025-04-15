using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InkMARC.Evaluate
{
    public partial class CameraPreview : View
    {
        public Action<float>? OnInferenceResult { get; set; }
    }
}
