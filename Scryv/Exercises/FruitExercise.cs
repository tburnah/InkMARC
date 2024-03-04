using Scryv.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Exercises
{
    public class FruitExercise : IExercise
    {
        public string Prompt => "Trace the image shown.";
        public string? TraceImage => "fruit.png";
    }
}
