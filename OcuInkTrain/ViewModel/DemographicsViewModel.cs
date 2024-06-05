using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using OcuInkTrain.Views;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OcuInk.ViewModel
{
    /// <summary>
    /// Represents the view model for the demographics view.
    /// </summary>
    public partial class DemographicsViewModel : ObservableObject
    {
        private string age = string.Empty;
        private string gender = string.Empty;
        private string ethnicity = string.Empty;
        private string handedness = string.Empty;

        /// <summary>
        /// Gets or sets the navigation service.
        /// </summary>
        public INavigation? Navigation { get; set; }

        /// <summary>
        /// Gets or sets the age.
        /// </summary>
        public string Age
        {
            get => age;
            set => SetProperty(ref age, value);
        }

        /// <summary>
        /// Gets or sets the gender.
        /// </summary>
        public string Gender
        {
            get => gender;
            set => SetProperty(ref gender, value);
        }

        /// <summary>
        /// Gets or sets the ethnicity.
        /// </summary>
        public string Ethnicity
        {
            get => ethnicity;
            set => SetProperty(ref ethnicity, value);
        }

        /// <summary>
        /// Gets or sets the handedness.
        /// </summary>
        public string Handedness
        {
            get => handedness;
            set => SetProperty(ref handedness, value);
        }

        /// <summary>
        /// Selects the age.
        /// </summary>
        /// <param name="value">The selected age value.</param>
        [RelayCommand]
        public void SelectAge(object value)
        {
            Age = value.ToString();
        }

        /// <summary>
        /// Selects the gender.
        /// </summary>
        /// <param name="value">The selected gender value.</param>
        [RelayCommand]
        public void SelectGender(object value)
        {
            Gender = value.ToString();
        }

        /// <summary>
        /// Selects the ethnicity.
        /// </summary>
        /// <param name="value">The selected ethnicity value.</param>
        [RelayCommand]
        public void SelectEthnicity(object value)
        {
            Ethnicity = value.ToString();
        }

        /// <summary>
        /// Selects the handedness.
        /// </summary>
        /// <param name="value">The selected handedness value.</param>
        [RelayCommand]
        public void SelectHandedness(object value)
        {
            Handedness = value.ToString();
        }

        [RelayCommand]
        public void Continue()
        {
            Navigation?.PushAsync(new CameraSelection());
        }

    }
}
