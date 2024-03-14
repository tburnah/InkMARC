using Scryv.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Exercises
{
    public class PetExercise : IExercise
    {
        public string Prompt => "Draw a picture of a pet you would like to have.";

        public string? TraceImage => null;
    }
}
