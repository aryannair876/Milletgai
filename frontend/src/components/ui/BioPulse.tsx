import { cn } from "@/lib/utils";

export function BioPulse({ active = false }: { active?: boolean }) {
    return (
        <div className={cn("relative flex items-center justify-center w-12 h-12 transition-opacity duration-500", active ? "opacity-100" : "opacity-50")}>
            {/* Core Seed */}
            <div className="w-3 h-3 rounded-full bg-grain z-10" />

            {/* Rings */}
            <div className={cn("absolute inset-0 rounded-full border border-grain/30", active && "animate-[ping_3s_cubic-bezier(0,0,0.2,1)_infinite]")} />
            <div className={cn("absolute inset-2 rounded-full border border-neon/20", active && "animate-[ping_2s_cubic-bezier(0,0,0.2,1)_infinite_200ms]")} />

            {/* Glow */}
            {active && (
                <div className="absolute inset-0 bg-neon/10 blur-xl animate-pulse" />
            )}
        </div>
    );
}
