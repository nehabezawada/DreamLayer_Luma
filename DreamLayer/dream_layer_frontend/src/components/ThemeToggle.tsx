
import { Moon, Sun } from "lucide-react";
import { useState, useEffect } from "react";

export const ThemeToggle = () => {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    // Check if dark mode is set in localStorage or if the user prefers dark mode
    const savedTheme = localStorage.getItem("theme");
    return savedTheme === "dark" || (!savedTheme && window.matchMedia("(prefers-color-scheme: dark)").matches);
  });

  useEffect(() => {
    // Update DOM and localStorage when theme changes
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [isDarkMode]);

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <div className="flex items-center gap-2">
      <span className="text-sm font-medium text-muted-foreground">
        {isDarkMode ? "Dark" : "Light"}
      </span>
      <button
        onClick={toggleTheme}
        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 ${
          isDarkMode ? "bg-primary" : "bg-gray-200"
        }`}
        aria-label="Toggle theme"
      >
        <span className="sr-only">Toggle theme</span>
        <span
          className={`${
            isDarkMode ? "translate-x-6" : "translate-x-1"
          } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
        />
        <span className="absolute right-1.5 top-1/2 -translate-y-1/2">
          {isDarkMode && (
            <Moon className="h-3 w-3 text-white" />
          )}
        </span>
        <span className="absolute left-1.5 top-1/2 -translate-y-1/2">
          {!isDarkMode && (
            <Sun className="h-3 w-3 text-gray-500" />
          )}
        </span>
      </button>
    </div>
  );
};
