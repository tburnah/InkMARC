using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Interfaces
{
    public interface IExercise
    {
        public string Prompt { get; }

        public string? TraceImage { get; }
    }
}
