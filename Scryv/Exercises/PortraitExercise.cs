using Scryv.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Exercises
{
    public class PortraitExercise : IExercise
    {
        public string Prompt => "Trace the portrait shown.";
        public string? TraceImage => "girl.png";
    }
}
