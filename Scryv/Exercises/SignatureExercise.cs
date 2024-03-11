using Scryv.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Exercises
{
    public class SignatureExercise : IExercise
    {
        public string Prompt => "Sign your name (with your signature).";

        public string? TraceImage => null;
    }
}
