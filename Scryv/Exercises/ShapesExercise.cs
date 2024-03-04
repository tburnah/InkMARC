using Scryv.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Exercises
{
    public class ShapesExercise : IExercise
    {        
        public string Prompt => "Trace the given image.";
        public string? TraceImage => "shapes.png";
    }
}
