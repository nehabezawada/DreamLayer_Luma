
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'

// Initialize theme based on system preference or saved preference
const initializeTheme = () => {
  const savedTheme = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  
  if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
    document.documentElement.classList.add('dark');
  }
};

// Initialize theme before rendering to prevent flash
initializeTheme();

createRoot(document.getElementById("root")!).render(<App />);
