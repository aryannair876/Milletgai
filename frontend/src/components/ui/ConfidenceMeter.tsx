import { cn } from "@/lib/utils";

interface ConfidenceMeterProps {
    score: number; // 0 to 100
    source?: string;
}

export function ConfidenceMeter({ score, source = "ICAR Database" }: ConfidenceMeterProps) {
    const isHigh = score > 80;
    const isMedium = score > 50;

    // Color logic
    const barColor = isHigh ? "bg-neon" : isMedium ? "bg-amber-400" : "bg-red-500";

    return (
        <div className="flex flex-col gap-1 mt-2">
            <div className="flex justify-between items-end text-[10px] uppercase tracking-wider font-mono text-muted-foreground">
                <span>RAG Confidence</span>
                <span>{score}%</span>
            </div>

            {/* Bar Track */}
            <div className="h-1.5 w-full bg-olive/40 rounded-sm overflow-hidden flex">
                {/* Segments for that 'high-tech' feel instead of solid bar */}
                <div className={cn("h-full transition-all duration-1000", barColor)} style={{ width: `${score}%` }} />
            </div>

            {/* Source Tag */}
            <div className="flex items-center gap-1.5 mt-1">
                <div className={cn("w-1.5 h-1.5 rounded-full", barColor, "animate-pulse")} />
                <span className="text-[10px] text-grain/80 font-mono">Verified by {source}</span>
            </div>
        </div>
    );
}
