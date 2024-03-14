using Scryv.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Exercises
{
    internal class FoodExercise : IExercise
    {
        public string Prompt => "Draw a picture of a your favorite food.";

        public string? TraceImage => null;       
    }
}
