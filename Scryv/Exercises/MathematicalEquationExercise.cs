using Scryv.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Exercises
{
    internal class MathematicalEquationExercise : IExercise
    {
        public string Prompt => "Write as many mathematical equations as you can fit.  I.e. 1+1=2, e=mc<sup>^2, a^2 + b^2 = c^2, ...";
        public string? TraceImage => null;
    }
}
