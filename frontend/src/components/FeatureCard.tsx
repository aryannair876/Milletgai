import { ArrowRight } from "lucide-react";

interface FeatureCardProps {
    title: string;
    description: string;
    helperText: string;
    icon: React.ReactNode;
    onClick: () => void;
    theme: "green" | "orange" | "red" | "amber" | "yellow" | "blue" | "indigo" | "teal" | "gray";
}

const themeStyles = {
    green: { bg: "bg-green-50", border: "hover:border-green-200", icon: "bg-green-600", text: "group-hover:text-green-700", accent: "text-green-600", ring: "focus:ring-green-500" },
    orange: { bg: "bg-orange-50", border: "hover:border-orange-200", icon: "bg-orange-500", text: "group-hover:text-orange-700", accent: "text-orange-600", ring: "focus:ring-orange-500" },
    red: { bg: "bg-red-50", border: "hover:border-red-200", icon: "bg-red-500", text: "group-hover:text-red-700", accent: "text-red-600", ring: "focus:ring-red-500" },
    amber: { bg: "bg-amber-50", border: "hover:border-amber-200", icon: "bg-amber-600", text: "group-hover:text-amber-700", accent: "text-amber-600", ring: "focus:ring-amber-500" },
    yellow: { bg: "bg-yellow-50", border: "hover:border-yellow-200", icon: "bg-yellow-600", text: "group-hover:text-yellow-700", accent: "text-yellow-600", ring: "focus:ring-yellow-500" },
    blue: { bg: "bg-blue-50", border: "hover:border-blue-200", icon: "bg-blue-500", text: "group-hover:text-blue-700", accent: "text-blue-600", ring: "focus:ring-blue-500" },
    indigo: { bg: "bg-indigo-50", border: "hover:border-indigo-200", icon: "bg-indigo-600", text: "group-hover:text-indigo-700", accent: "text-indigo-600", ring: "focus:ring-indigo-500" },
    teal: { bg: "bg-teal-50", border: "hover:border-teal-200", icon: "bg-teal-600", text: "group-hover:text-teal-700", accent: "text-teal-600", ring: "focus:ring-teal-500" },
    gray: { bg: "bg-gray-50", border: "hover:border-gray-200", icon: "bg-gray-600", text: "group-hover:text-gray-700", accent: "text-gray-600", ring: "focus:ring-gray-500" }
};

export function FeatureCard({ title, description, helperText, icon, onClick, theme }: FeatureCardProps) {
    const styles = themeStyles[theme];

    return (
        <button
            onClick={onClick}
            className={`group relative flex flex-col items-start text-left bg-white rounded-2xl p-6 shadow-sm hover:shadow-xl border border-transparent ${styles.border} transition-all duration-300 w-full overflow-hidden hover:-translate-y-1 hover:scale-[1.02] active:scale-95`}
        >
            {/* Ambient Background Tint */}
            <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 ${styles.bg}`} />

            {/* Decorative gradient blob */}
            <div className={`absolute -top-10 -right-10 w-40 h-40 opacity-10 rounded-full blur-3xl transition-transform duration-700 group-hover:scale-150 ${styles.icon.replace('bg-', 'bg-')}`} />

            {/* Icon */}
            <div className={`relative p-3 rounded-xl mb-4 text-white shadow-md transition-all duration-300 group-hover:scale-110 group-hover:rotate-3 ${styles.icon}`}>
                {icon}
            </div>

            {/* Content */}
            <h3 className={`relative text-lg font-bold text-charcoal mb-2 transition-colors ${styles.text}`}>
                {title}
            </h3>
            <p className="relative text-sm text-charcoal/60 mb-6 line-clamp-2 leading-relaxed">
                {description}
            </p>

            {/* Helper Text (Bottom) */}
            <div className={`relative mt-auto w-full pt-4 border-t border-gray-100 flex items-center justify-between text-xs font-bold uppercase tracking-wider ${styles.accent}`}>
                <span>{helperText}</span>
                <div className="bg-white/50 p-1.5 rounded-full border border-gray-100 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-300">
                    <ArrowRight size={12} />
                </div>
            </div>
        </button>
    );
}
