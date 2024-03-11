using Scryv.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Exercises
{
    public class MazeExercise : IExercise
    {
        public string Prompt => "Draw a line through the maze.";

        public string? TraceImage => "maze.png";
    }
}
