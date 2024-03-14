using Scryv.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Exercises
{
    public class LandscapeExercise : IExercise
    {
        public string Prompt => "Draw a picture of a place outside. You can show things like trees, mountains, or a river.";

        public string? TraceImage => null;
    }
}
