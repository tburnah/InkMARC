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
        public string Prompt => "Duplicate the shading with lines or dots.";

        public string? TraceImage => "shadedshapes.png";
    }
}
