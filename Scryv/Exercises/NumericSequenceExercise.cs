using Scryv.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Exercises
{
    public class NumericSequenceExercise : IExercise
    {
        public string Prompt => "Starting at -5, write sequential numbers counting up.  Include as many numbers as you can fit.  I.e. -5 -4 -3 -2 -1 0 1 2 ...";
        public string? TraceImage => null;
    }
}
