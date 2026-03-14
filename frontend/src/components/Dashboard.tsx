import { useState } from "react";
import { Sprout, Utensils, Activity, Bug, Warehouse, CloudSun, TrendingUp, Cpu, Send, Factory, Zap, Brain } from "lucide-react";
import { FeatureCard } from "./FeatureCard";

export interface Feature {
    title: string;
    description: string;
    helperText: string;
    icon: React.ReactNode;
    id: string;
    color: string;
    theme: "green" | "orange" | "red" | "amber" | "yellow" | "blue" | "indigo" | "teal" | "gray"; // Specific theme keys
}

interface DashboardProps {
    onFeatureSelect: (feature: Feature, initialQuery?: string) => void;
    mode: "fast" | "thinking";
    onModeChange: (mode: "fast" | "thinking") => void;
}

export function Dashboard({ onFeatureSelect, mode, onModeChange }: DashboardProps) {
    const [query, setQuery] = useState("");
    const [isTransitioning, setIsTransitioning] = useState(false);

    const features: Feature[] = [
        {
            title: "Cultivation",
            description: "Expert guidance on sowing, vegetative growth, flowering, and maturity stages.",
            helperText: "Ask about crop growth & yield",
            icon: <Sprout size={24} />,
            id: "Cultivation",
            color: "bg-teal-600",
            theme: "teal"
        },
        {
            title: "Recipe",
            description: "Discover nutritious millet-based recipes and meal planning ideas.",
            helperText: "Ask for recipes & meal plans",
            icon: <Utensils size={24} />,
            id: "Recipe",
            color: "bg-orange-500",
            theme: "orange"
        },
        {
            title: "Nutritional Benefits",
            description: "Detailed analysis of protein, fiber, and mineral content in millets.",
            helperText: "Analyze nutritional values",
            icon: <Activity size={24} />,
            id: "Nutritional Benefits",
            color: "bg-red-500",
            theme: "red"
        },
        {
            title: "Pest Controls",
            description: "Identify pests and diseases with solutions and control methods.",
            helperText: "Identify pests & solutions",
            icon: <Bug size={24} />,
            id: "Pest Controls",
            color: "bg-amber-600",
            theme: "amber"
        },
        {
            title: "Harvesting & Storage",
            description: "Best practices for harvesting timeframes and safe storage techniques.",
            helperText: "Learn storage techniques",
            icon: <Warehouse size={24} />,
            id: "Harvesting & Storage",
            color: "bg-yellow-600",
            theme: "yellow"
        },
        {
            title: "Soil & Climate",
            description: "Check soil suitability, temperature, and rainfall requirements.",
            helperText: "Check climate suitability",
            icon: <CloudSun size={24} />,
            id: "Soil & Climate",
            color: "bg-blue-500",
            theme: "blue"
        }
    ];

    const handleGeneralQuery = () => {
        if (!query.trim()) return;

        setIsTransitioning(true);

        // Wait for animation to finish before switching view
        setTimeout(() => {
            onFeatureSelect({
                title: "General Query",
                description: "MilletGAI Core Assistant",
                helperText: "Ask anything...",
                icon: <Cpu size={24} />,
                id: "General Query",
                color: "bg-gray-700",
                theme: "gray"
            }, query);
        }, 500); // Match this with CSS transition duration
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleGeneralQuery();
        }
    };

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 fade-in pb-24">
            <div className={`transition-all duration-500 transform ${isTransitioning ? '-translate-y-10 opacity-0' : 'translate-y-0 opacity-100'}`}>
                <div className="mb-8 text-center">
                    <h1 className="text-3xl font-bold text-charcoal mb-3">
                        Welcome to <span className="text-deep-green">MilletGAI</span>
                    </h1>
                    <p className="text-charcoal/60 max-w-2xl mx-auto">
                        An advanced AI-powered platform for millet research, cultivation insights, and nutritional analysis.
                    </p>
                </div>

                {/* General Query Section - Moved Up */}
                <div className="max-w-3xl mx-auto mb-10">
                    <div className="bg-white rounded-2xl shadow-soft p-2 flex items-center gap-2 border border-gray-100 transition-all focus-within:ring-2 focus-within:ring-deep-green/20 focus-within:border-deep-green">
                        <input
                            type="text"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask general query..."
                            className="flex-1 bg-transparent border-none focus:ring-0 text-charcoal placeholder:text-gray-400 px-4 py-3 text-lg"
                        />
                        {/* Mode Toggle */}
                        <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
                            <button
                                onClick={() => onModeChange("fast")}
                                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${mode === "fast"
                                        ? "bg-white shadow-sm text-amber-600"
                                        : "text-gray-500 hover:text-gray-700"
                                    }`}
                            >
                                <Zap size={14} />
                                <span>Fast</span>
                            </button>
                            <button
                                onClick={() => onModeChange("thinking")}
                                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${mode === "thinking"
                                        ? "bg-white shadow-sm text-purple-600"
                                        : "text-gray-500 hover:text-gray-700"
                                    }`}
                            >
                                <Brain size={14} />
                                <span>Thinking</span>
                            </button>
                        </div>
                        <button
                            onClick={handleGeneralQuery}
                            disabled={!query.trim()}
                            className="p-3 bg-charcoal/80 hover:bg-deep-green text-white rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <Send size={20} />
                        </button>
                    </div>
                </div>
            </div>

            <div className={`grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-16 transition-all duration-500 delay-100 transform ${isTransitioning ? 'translate-y-20 opacity-0' : 'translate-y-0 opacity-100'}`}>
                {features.map((feature) => (
                    <FeatureCard
                        key={feature.id}
                        title={feature.title}
                        description={feature.description}
                        helperText={feature.helperText}
                        icon={feature.icon}
                        theme={feature.theme}
                        onClick={() => onFeatureSelect(feature)}
                    />
                ))}
            </div>
        </div>
    );
}

