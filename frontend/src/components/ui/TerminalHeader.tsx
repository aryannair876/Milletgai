import { Sprout, Wifi, Cpu } from "lucide-react";
import { cn } from "@/lib/utils";

export function TerminalHeader() {
    return (
        <header className="h-16 border-b border-olive/30 bg-loam/80 backdrop-blur-md flex items-center justify-between px-6 shrink-0 z-20">
            {/* Logo Section */}
            <div className="flex items-center gap-3">
                <div className="relative flex items-center justify-center w-8 h-8 rounded border border-grain/20 bg-olive/20 text-grain">
                    <Sprout size={18} />
                    {/* Subtle glow effect */}
                    <div className="absolute inset-0 bg-grain/10 blur-md rounded-full animate-pulse-glow" />
                </div>
                <h1 className="font-serif text-2xl tracking-wider text-grain font-semibold">
                    Millets<span className="text-neon">GAI</span>
                </h1>
            </div>

            {/* Connection Status */}
            <div className="flex items-center gap-6 text-xs text-muted-foreground font-mono">
                <div className="flex items-center gap-2">
                    <Cpu size={14} className="text-neon" />
                    <span>MILLETSGAI: <span className="text-neon animate-pulse">ACTIVE</span></span>
                </div>
                <div className="flex items-center gap-2">
                    <Wifi size={14} className="text-grain" />
                    <span>NET: STABLE</span>
                </div>
                <div className="px-2 py-1 border border-olive rounded bg-olive/20">
                    v2.0.5-ALPHA
                </div>
            </div>
        </header>
    );
}
