import { useState, useRef, useEffect } from "react";
import { Send, ArrowLeft, Bot, User, Loader2, Sprout, Utensils, Activity, Bug, Warehouse, CloudSun, TrendingUp, Cpu, ChevronDown, ChevronUp, Brain, Zap } from "lucide-react";
import ReactMarkdown from "react-markdown";

// Backend API location. Override via NEXT_PUBLIC_API_URL in .env.local.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface ChatViewProps {
    context: string;
    theme?: string;
    initialQuery?: string;
    onBack: () => void;
    initialMode?: "fast" | "thinking";
    onModeChange?: (mode: "fast" | "thinking") => void;
}

type Message = {
    id: string;
    type: "user" | "ai";
    content: string;
    citations?: string[];
    confidence?: number;
    thinking?: string | null;
    isComplex?: boolean;
};

// Theme Helper
const getThemeColors = (theme: string = "gray") => {
    const maps: Record<string, { bg: string, text: string, accent: string, border: string, userBubble: string }> = {
        teal: { bg: "bg-teal-50", text: "text-teal-800", accent: "text-teal-600", border: "border-teal-200", userBubble: "bg-teal-600" },
        green: { bg: "bg-green-50", text: "text-green-800", accent: "text-green-600", border: "border-green-200", userBubble: "bg-green-700" },
        orange: { bg: "bg-orange-50", text: "text-orange-800", accent: "text-orange-600", border: "border-orange-200", userBubble: "bg-orange-600" },
        red: { bg: "bg-red-50", text: "text-red-800", accent: "text-red-600", border: "border-red-200", userBubble: "bg-red-600" },
        amber: { bg: "bg-amber-50", text: "text-amber-800", accent: "text-amber-600", border: "border-amber-200", userBubble: "bg-amber-600" },
        yellow: { bg: "bg-yellow-50", text: "text-yellow-800", accent: "text-yellow-600", border: "border-yellow-200", userBubble: "bg-yellow-600" },
        blue: { bg: "bg-blue-50", text: "text-blue-800", accent: "text-blue-600", border: "border-blue-200", userBubble: "bg-blue-600" },
        indigo: { bg: "bg-indigo-50", text: "text-indigo-800", accent: "text-indigo-600", border: "border-indigo-200", userBubble: "bg-indigo-600" },
        gray: { bg: "bg-gray-50", text: "text-gray-800", accent: "text-gray-600", border: "border-gray-200", userBubble: "bg-gray-700" },
    };
    return maps[theme] || maps.gray;
};

export function ChatView({ context, theme = "gray", initialQuery, onBack, initialMode = "fast", onModeChange }: ChatViewProps) {
    const [input, setInput] = useState("");
    const [messages, setMessages] = useState<Message[]>([
        {
            id: "welcome",
            type: "ai",
            content: `Hello! I am your **${context}** assistant. How can I help you regarding ${context.toLowerCase()} today?`,
            citations: []
        }
    ]);
    const [isThinking, setIsThinking] = useState(false);
    const [expandedThinking, setExpandedThinking] = useState<Record<string, boolean>>({});
    const [mode, setMode] = useState<"fast" | "thinking">(initialMode);

    // Sync mode changes with parent
    const handleModeChange = (newMode: "fast" | "thinking") => {
        setMode(newMode);
        onModeChange?.(newMode);
    };
    // Use a ref to track if we have already processed the initial query
    const hasProcessedInitialQuery = useRef(false);

    const scrollRef = useRef<HTMLDivElement>(null);
    const colors = getThemeColors(theme);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
        }
    }, [messages, isThinking]);

    const fetchResponse = async (query: string) => {
        try {
            const res = await fetch(`${API_BASE_URL}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: query, context: context, mode: mode })
            });

            if (!res.ok) throw new Error("Backend connection failed");

            const data = await res.json();

            const aiMsg: Message = {
                id: (Date.now() + 1).toString(),
                type: "ai",
                content: data.response,
                citations: data.citations,
                confidence: data.confidence,
                thinking: data.thinking || null,
                isComplex: data.is_complex || false
            };

            setMessages(prev => [...prev, aiMsg]);

        } catch (error) {
            console.error(error);
            const errorMsg: Message = {
                id: (Date.now() + 1).toString(),
                type: "ai",
                content: "⚠️ Error connecting to MilletGAI Core. Please check your connection.",
                citations: [],
                confidence: 0
            };
            setMessages(prev => [...prev, errorMsg]);
        } finally {
            setIsThinking(false);
        }
    };

    // Handle initial query
    useEffect(() => {
        if (initialQuery && !hasProcessedInitialQuery.current) {
            hasProcessedInitialQuery.current = true;
            // Add user message immediately
            const userMsg: Message = {
                id: Date.now().toString(),
                type: "user",
                content: initialQuery
            };
            setMessages(prev => [...prev, userMsg]);
            setIsThinking(true);

            // Trigger fetch
            fetchResponse(initialQuery);
        }
    }, [initialQuery, context]); // Added context to dependency array as fetchResponse uses it.

    const handleSendMessage = () => {
        if (!input.trim()) return;

        const userMsg: Message = {
            id: Date.now().toString(),
            type: "user",
            content: input
        };

        setMessages(prev => [...prev, userMsg]);
        setInput("");
        setIsThinking(true);
        fetchResponse(input);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    return (
        <div className={`max-w-5xl mx-auto h-[calc(100vh-6rem)] mt-4 ${colors.bg} backdrop-blur-md rounded-2xl shadow-soft border ${colors.border} flex flex-col overflow-hidden transition-colors duration-500`}>
            {/* Header */}
            <div className={`bg-white/60 border-b ${colors.border} p-4 flex items-center gap-4 sticky top-0 backdrop-blur-md z-10`}>
                <button onClick={onBack} className="p-2 hover:bg-white/50 rounded-full transition-colors text-charcoal/60 hover:text-charcoal shadow-sm border border-transparent hover:border-gray-200">
                    <ArrowLeft size={20} />
                </button>
                <div className="flex-1">
                    <h2 className={`text-lg font-bold ${colors.text}`}>{context} Assistant</h2>
                </div>
                {/* Mode Toggle */}
                <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
                    <button
                        onClick={() => handleModeChange("fast")}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${mode === "fast"
                            ? "bg-white shadow-sm text-amber-600"
                            : "text-gray-500 hover:text-gray-700"
                            }`}
                    >
                        <Zap size={14} />
                        <span>Fast</span>
                    </button>
                    <button
                        onClick={() => handleModeChange("thinking")}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${mode === "thinking"
                            ? "bg-white shadow-sm text-purple-600"
                            : "text-gray-500 hover:text-gray-700"
                            }`}
                    >
                        <Brain size={14} />
                        <span>Thinking</span>
                    </button>
                </div>
            </div>

            {/* Background Icon (Unique Page Vibe) */}
            <div className="absolute inset-0 pointer-events-none overflow-hidden flex items-center justify-center opacity-[0.03]">
                <div className={`transform scale-[5] rotate-12 ${colors.text}`}>
                    {(() => {
                        const iconMap: Record<string, React.ReactNode> = {
                            "Cultivation": <Sprout size={200} />,
                            "Recipe": <Utensils size={200} />,
                            "Nutritional Benefits": <Activity size={200} />,
                            "Pest Controls": <Bug size={200} />,
                            "Harvesting & Storage": <Warehouse size={200} />,
                            "Soil & Climate": <CloudSun size={200} />,
                            "General Query": <Cpu size={200} />
                        };
                        return iconMap[context] || <Bot size={200} />;
                    })()}
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6" ref={scrollRef}>
                {messages.map((msg) => (
                    <div key={msg.id} className={`flex gap-4 ${msg.type === "user" ? "justify-end" : "justify-start"}`}>

                        {/* Avatar AI */}
                        {msg.type === "ai" && (
                            <div className={`w-8 h-8 rounded-full ${colors.userBubble} flex-shrink-0 flex items-center justify-center text-white shadow-sm mt-1 opacity-80`}>
                                <Bot size={16} />
                            </div>
                        )}

                        {/* Bubble */}
                        <div className={`max-w-[80%] p-4 rounded-2xl text-sm leading-relaxed shadow-sm ${msg.type === "user"
                            ? `${colors.userBubble} text-white`
                            : "bg-white text-charcoal border border-gray-100"
                            }`}>
                            {msg.type === "ai" ? (
                                <div className="prose prose-sm max-w-none [&>ul]:list-disc [&>ul]:pl-4 [&>ul]:my-2 [&>ol]:list-decimal [&>ol]:pl-4 [&>ol]:my-2 [&>p]:my-2 [&>li]:my-1">
                                    {/* Thinking Dropdown */}
                                    {msg.thinking && (
                                        <div className="mb-3">
                                            <button
                                                onClick={() => setExpandedThinking(prev => ({ ...prev, [msg.id]: !prev[msg.id] }))}
                                                className={`flex items-center gap-2 text-xs font-medium text-purple-600 hover:opacity-80 transition-opacity py-1.5 px-3 rounded-md bg-purple-50 border border-purple-200`}
                                            >
                                                <Brain size={14} />
                                                <span>Show Reasoning</span>
                                                {expandedThinking[msg.id] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                            </button>
                                            {expandedThinking[msg.id] && (
                                                <div className="mt-2 p-3 bg-purple-50 border border-purple-200 rounded-lg text-xs text-gray-600 italic [&>ul]:list-disc [&>ul]:pl-4 [&>ul]:my-2 [&>p]:my-1">
                                                    <ReactMarkdown>{msg.thinking}</ReactMarkdown>
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    <ReactMarkdown>{msg.content}</ReactMarkdown>

                                    {/* Citations */}
                                    {msg.citations && msg.citations.length > 0 && (
                                        <div className={`mt-4 pt-3 border-t ${colors.border} opacity-80`}>
                                            <p className={`text-[10px] uppercase font-bold ${colors.accent} mb-1`}>Sources:</p>
                                            <div className="flex flex-wrap gap-2">
                                                {msg.citations.map((cite, i) => (
                                                    <span key={i} className={`px-2 py-1 bg-white/50 border ${colors.border} ${colors.accent} text-[10px] rounded-md truncate max-w-[200px]`}>
                                                        {cite}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <p className="whitespace-pre-wrap">{msg.content}</p>
                            )}
                        </div>

                        {/* Avatar User */}
                        {msg.type === "user" && (
                            <div className="w-8 h-8 rounded-full bg-gray-400 flex-shrink-0 flex items-center justify-center text-white shadow-sm mt-1">
                                <User size={16} />
                            </div>
                        )}
                    </div>
                ))}

                {/* Thinking Indicator */}
                {isThinking && (
                    <div className="flex gap-4 justify-start">
                        <div className={`w-8 h-8 rounded-full ${colors.userBubble} flex-shrink-0 flex items-center justify-center text-white shadow-sm mt-1 opacity-80`}>
                            <Bot size={16} />
                        </div>
                        <div className="bg-white px-4 py-3 rounded-2xl border border-gray-100 flex items-center gap-2 text-sm text-charcoal/60 shadow-sm">
                            <Loader2 size={16} className={`animate-spin ${colors.accent}`} />
                            <span className="animate-pulse">Processing...</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Input Area */}
            <div className={`p-4 bg-white/60 border-t ${colors.border} backdrop-blur-md`}>
                <div className={`relative flex items-end gap-2 bg-white border ${colors.border} rounded-xl p-2 focus-within:ring-2 focus-within:ring-black/5 transition-all shadow-inner`}>
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={`Ask about ${context.toLowerCase()}...`}
                        className="w-full bg-transparent border-none focus:ring-0 text-sm p-2 resize-none max-h-32 min-h-[44px]"
                        rows={1}
                        style={{ minHeight: '44px' }}
                    />
                    <button
                        onClick={handleSendMessage}
                        disabled={!input.trim() || isThinking}
                        className={`p-2 ${colors.userBubble} hover:opacity-90 text-white rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed mb-0.5 shadow-sm transform active:scale-95`}
                    >
                        <Send size={18} />
                    </button>
                </div>
            </div>
        </div>
    );
}
