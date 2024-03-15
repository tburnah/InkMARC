using Scryv.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Exercises
{
    public class ShadedExercise : IExercise
    {
        public string Prompt => "Color the shading, or shadows and lightness of the shapes.";

        public string? TraceImage => "shadedshapes.png";
    }
}
