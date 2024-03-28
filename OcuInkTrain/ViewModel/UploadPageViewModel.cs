using CommunityToolkit.Mvvm.ComponentModel;
using OcuInkTrain.Utilities;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OcuInk.ViewModel
{
    public class UploadPageViewModel : ObservableObject
    {

        public ObservableCollection<string> ImagePaths { get; private set; }

        public UploadPageViewModel()
        {
            ImagePaths = new ObservableCollection<string>();
            foreach (string path in SessionContext.ImagePaths)
            {
                ImagePaths.Add(path);
            }
        }
    }
}
