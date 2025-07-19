import { cn } from "@/lib/utils";

interface SubTabNavigationProps {
  tabs: Array<{ id: string; label: string; active?: boolean }>;
  onTabChange?: (tabId: string) => void;
  className?: string;
}

const SubTabNavigation = ({ tabs, onTabChange, className }: SubTabNavigationProps) => {
  return (
    <div className={cn("flex flex-wrap gap-3", className)}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange?.(tab.id)}
          className={cn(
            "px-6 py-2.5 rounded-full text-sm font-medium transition-all duration-200 border-0 focus:outline-none focus:ring-2 focus:ring-offset-2",
            tab.active 
              ? "bg-blue-600 text-white shadow-md hover:bg-blue-700 focus:ring-blue-500" 
              : "bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
};

export default SubTabNavigation;
