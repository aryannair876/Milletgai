import { cn } from "@/lib/utils";
import { ConfidenceMeter } from "./ConfidenceMeter";
import ReactMarkdown from "react-markdown";

interface ChatDataCardProps {
    type: "user" | "ai";
    content: string;
    citations?: string[];
    confidence?: number;
}

export function ChatDataCard({ type, content, citations, confidence }: ChatDataCardProps) {
    if (type === "user") {
        return (
            <div className="flex justify-end mb-8">
                <div className="max-w-xl">
                    <div className="flex items-center justify-end gap-2 mb-1 opacity-70">
                        <span className="text-[10px] uppercase font-mono tracking-widest text-neon">{">"} OPERATOR</span>
                    </div>
                    <div className="p-4 border border-olive/50 bg-loam text-right font-mono text-sm text-foreground/90 shadow-none">
                        {content}
                    </div>
                </div>
            </div>
        );
    }

    // AI Response
    return (
        <div className="flex justify-start mb-8 w-full">
            <div className="max-w-2xl w-full">
                {/* Header */}
                <div className="flex items-center gap-2 mb-1 px-1">
                    <div className="w-1.5 h-1.5 bg-grain rounded-full" />
                    <span className="text-[10px] uppercase font-mono tracking-widest text-grain">FIELD NOTE_1024</span>
                    <div className="h-px flex-1 bg-gradient-to-r from-grain/30 to-transparent" />
                </div>

                {/* Body */}
                <div className="p-5 border-l-2 border-l-grain border-r border-t border-b border-olive/30 bg-olive/10 backdrop-blur-sm rounded-tr-lg rounded-br-lg relative overflow-hidden">
                    {/* Decorative Corner */}
                    <div className="absolute top-0 right-0 w-4 h-4 border-t border-r border-grain/40" />

                    <div className="prose prose-invert prose-sm font-sans mix-blend-lighten max-w-none text-gray-200 leading-relaxed">
                        <ReactMarkdown
                            components={{
                                a: ({ ...props }) => (
                                    <a {...props} className="text-neon underline hover:text-white transition-colors" target="_blank" rel="noopener noreferrer" />
                                ),
                                strong: ({ ...props }) => <strong {...props} className="text-grain font-bold" />
                            }}
                        >
                            {content}
                        </ReactMarkdown>
                    </div>

                    {/* Footer: Citations and Confidence */}
                    <div className="mt-6 pt-4 border-t border-dashed border-olive/30 grid grid-cols-2 gap-4">
                        <div>
                            {citations && citations.length > 0 && (
                                <div className="space-y-1">
                                    <span className="text-[10px] text-muted-foreground uppercase">References:</span>
                                    {citations.map((cite, i) => (
                                        <div key={i} className="text-xs font-mono text-neon truncate hover:text-white cursor-pointer transition-colors">
                                            <ReactMarkdown
                                                components={{
                                                    a: ({ ...props }) => (
                                                        <a {...props} className="text-neon hover:text-white transition-colors" target="_blank" rel="noopener noreferrer" />
                                                    )
                                                }}
                                            >
                                                {`[${i + 1}] ${cite}`}
                                            </ReactMarkdown>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                        <div className="flex justify-end">
                            {confidence && <div className="w-32"><ConfidenceMeter score={confidence} /></div>}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
