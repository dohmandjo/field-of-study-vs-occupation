import React, { useState } from 'react';
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
import './App.css'


// Helper function to capitalize the first letter of each word in a string
const capitalizeWords = (str: string) =>
  str.toLowerCase().replace(/\b\w/g, char => char.toUpperCase());

// Initial default values for the form
const initialState = {
  fieldOfStudy: '',
  currentOccupation: '',
  gender: '',
  educationLevel: '',
  industryGrowthRate: '',
  familyInfluence: '',
  age: 25,
  yearsOfExperience: 1,
  jobSatisfaction: 5,
  workLifeBalance: 5,
  jobOpportunities: 10,
  salary: 50000,
  jobSecurity: 5,
  careerChangeInterest: 0,
  skillsGap: 5,
  mentorshipAvailable: 0,
  certifications: 0,
  freelancingExperience: 0,
  geographicMobility: 0,
  professionalNetworks: 5,
  careerChangeEvents: 0,
  technologyAdoption: 5,
};

function App() {
  // React state hook to manage for data
  const [formData, setFormData] = useState(initialState);

  // Handle changes in any form input field
  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target; // Destructure input properties

    let finalValue: any = value;

    // Convert value to number if input type is number
    if (type === 'number') {
      finalValue = Number(value);
    }

    // Capitalize each word for specific string fields
    if (['fieldOfStudy', 'currentOccupation'].includes(name)) {
      finalValue = capitalizeWords(value);
    }

    // Update form state
    setFormData(prev => ({
      ...prev,
      [name]: finalValue,
    }));
  };

  // Handle for submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault(); // Prevent page refresh

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      alert(`Prediction result: ${result.prediction}`);
    } catch (error) {
      console.error("Prediction error:", error);
      alert("There was an error submitting the form. Please try again.");
    }
  };


    return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <form
        onSubmit={handleSubmit}
        className="bg-white shadow-lg rounded-lg p-8 w-full max-w-4xl space-y-6"
      >
        {/* Form title */}
        <h2 className="text-2xl font-bold text-center text-gray-800">
          Career Change Prediction Form
        </h2>

        {/* Grid layout for text/select/number inputs */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <TextInput label="Field of Study" name="fieldOfStudy" value={formData.fieldOfStudy} onChange={handleChange} />
          <TextInput label="Current Occupation" name="currentOccupation" value={formData.currentOccupation} onChange={handleChange} />
          <Select label="Gender" name="gender" value={formData.gender} onChange={handleChange} options={['Male', 'Female']} />
          <Select label="Education Level" name="educationLevel" value={formData.educationLevel} onChange={handleChange} options={["High School", "Bachelor's", "Master's", "PhD"]} />
          <Select label="Industry Growth Rate" name="industryGrowthRate" value={formData.industryGrowthRate} onChange={handleChange} options={['High', 'Medium', 'Low']} />
          <Select label="Family Influence" name="familyInfluence" value={formData.familyInfluence} onChange={handleChange} options={['None', 'Low', 'Medium', 'High']} />
          <NumberInput label="Age" name="age" value={formData.age} onChange={handleChange} min={18} max={150} />
          <NumberInput label="Years of Experience" name="yearsOfExperience" value={formData.yearsOfExperience} onChange={handleChange} min={0} max={100} />
          <Slider label="Job Satisfaction (1–10)" name="jobSatisfaction" value={formData.jobSatisfaction} onChange={handleChange} />
          <Slider label="Work-Life Balance (1–10)" name="workLifeBalance" value={formData.workLifeBalance} onChange={handleChange} />
          <NumberInput label="Job Opportunities" name="jobOpportunities" value={formData.jobOpportunities} onChange={handleChange} min={0} />
          <NumberInput label="Salary (USD)" name="salary" value={formData.salary} onChange={handleChange} min={0} step={1000} />
          <Slider label="Job Security (1–10)" name="jobSecurity" value={formData.jobSecurity} onChange={handleChange} />
          <SliderBin label="Career Change Interest (0-1)" name="careerChangeInterest" value={formData.careerChangeInterest} onChange={handleChange} />  
          <Slider label="Skills Gap (1–10)" name="skillsGap" value={formData.skillsGap} onChange={handleChange} />
          <SliderBin label="Mentorship Available" name="mentorshipAvailable" value={formData.mentorshipAvailable} onChange={handleChange} />
          <SliderBin label="Certifications" name="certifications" value={formData.certifications} onChange={handleChange} />
          <SliderBin label="Freelancing Experience" name="freelancingExperience" value={formData.freelancingExperience} onChange={handleChange} />
          <SliderBin label="Geographic Mobility" name="geographicMobility" value={formData.geographicMobility} onChange={handleChange} />
          <Slider label="Professional Networks (1–10)" name="professionalNetworks" value={formData.professionalNetworks} onChange={handleChange} />
          <NumberInput label="Career Change Events" name="careerChangeEvents" value={formData.careerChangeEvents} onChange={handleChange} min={0} />
          <Slider label="Technology Adoption (1–10)" name="technologyAdoption" value={formData.technologyAdoption} onChange={handleChange} />
        </div>

        {/* Submit button */}
        <button
          type="submit"
          className="w-full bg-blue-600 text-white font-semibold py-2 rounded hover:bg-blue-700 transition"
        >
          Predict
        </button>
      </form>
    </div>
  );
}


export default App;

// ---------------------- Reusable Input Components ----------------------

// Text input for custom strings (Field of Study, etc.)
type TextInputProps = {
  label: string;
  name: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
};

const TextInput = ({ label, name, value, onChange }: TextInputProps) => (
  <div>
    <label htmlFor={name} className="block text-sm font-medium text-gray-700 mb-1">
      {label}
    </label>
    <input
      type="text"
      name={name}
      id={name}
      value={value}
      onChange={onChange}
      className="w-full border border-gray-300 rounded px-3 py-2"
      required
    />
  </div>
);

// Numeric input field

type NumberInputProps = {
  label: string;
  name: string;
  value: number;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  min?: number;
  max?: number;
  step?: number;
};

const NumberInput = ({ label, name, value, onChange, min, max, step = 1 }: NumberInputProps) => (
  <div>
    <label htmlFor={name} className="block text-sm font-medium text-gray-700 mb-1">
      {label}
    </label>
    <input
      type="number"
      name={name}
      id={name}
      value={value}
      onChange={onChange}
      min={min}
      max={max}
      step={step}
      className="w-full border border-gray-300 rounded px-3 py-2"
      required
    />
  </div>
);

// Dropdown select for categorical/binary fields

type SelectProps = {
  label: string;
  name: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  options: string[];
};

const Select = ({ label, name, value, onChange, options }: SelectProps) => (
  <div>
    <label htmlFor={name} className="block text-sm font-medium text-gray-700 mb-1">
      {label}
    </label>
    <select
      name={name}
      id={name}
      value={value}
      onChange={onChange}
      className="w-full border border-gray-300 rounded px-3 py-2"
      required
    >
      <option value="">Select</option>
      {options.map(option => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
  </div>
);

// Slider input for 1–10 scale features

type SliderBinProps = {
  label: string;
  name: string;
  value: number;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
};

const Slider = ({ label, name, value, onChange }: SliderBinProps) => (
  <div className="flex flex-col">
    <label htmlFor={name} className="text-sm font-medium text-gray-700 mb-1">
      {label}: {value}
    </label>
    <input
      type="range"
      name={name}
      id={name}
      min={1}
      max={10}
      value={value}
      onChange={onChange}
      className="w-full"
    />
  </div>
);

// Slider input for 1–10 scale features

type SliderProps = {
  label: string;
  name: string;
  value: number;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
};

const SliderBin = ({ label, name, value, onChange }: SliderProps) => (
  <div className="flex flex-col">
    <label htmlFor={name} className="text-sm font-medium text-gray-700 mb-1">
      {label}: {value}
    </label>
    <input
      type="range"
      name={name}
      id={name}
      min={0}
      max={1}
      value={value}
      onChange={onChange}
      className="w-full"
    />
  </div>
);



// function App() {
//   const [count, setCount] = useState(0)

//   return (
//     <>
//       <div>
//         <a href="https://vite.dev" target="_blank">
//           <img src={viteLogo} className="logo" alt="Vite logo" />
//         </a>
//         <a href="https://react.dev" target="_blank">
//           <img src={reactLogo} className="logo react" alt="React logo" />
//         </a>
//       </div>
//       <h1>Vite + React</h1>
//       <div className="card">
//         <button onClick={() => setCount((count) => count + 1)}>
//           count is {count}
//         </button>
//         <p>
//           Edit <code>src/App.tsx</code> and save to test HMR
//         </p>
//       </div>
//       <p className="read-the-docs">
//         Click on the Vite and React logos to learn more
//       </p>
//     </>
//   )
// }

// export default App


