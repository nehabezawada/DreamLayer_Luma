
import { ChevronDown } from "lucide-react";
import { useState } from "react";

interface AccordionProps {
  title: string;
  number?: string | number;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

const Accordion = ({ title, number, defaultOpen = false, children }: AccordionProps) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="mb-4 overflow-hidden rounded-md border border-border">
      <button
        className="flex w-full items-center justify-between bg-background p-3 text-left font-medium"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className="flex items-center">
          {number && <span className="mr-2 text-foreground">{number}.</span>}
          {title}
        </span>
        <ChevronDown
          className={`h-5 w-5 text-muted-foreground transition-transform ${isOpen ? "rotate-180" : ""}`}
        />
      </button>
      <div
        className={`bg-card px-4 transition-all ${
          isOpen ? "max-h-[2000px] py-4" : "max-h-0 py-0"
        }`}
      >
        {children}
      </div>
    </div>
  );
};

export default Accordion;
