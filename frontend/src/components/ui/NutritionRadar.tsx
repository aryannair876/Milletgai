"use client";

import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer } from "recharts";

const data = [
    { subject: 'Protein', A: 120, fullMark: 150 },
    { subject: 'Fiber', A: 98, fullMark: 150 },
    { subject: 'Iron', A: 86, fullMark: 150 },
    { subject: 'Calcium', A: 99, fullMark: 150 },
    { subject: 'Carbs', A: 85, fullMark: 150 },
    { subject: 'Micro', A: 65, fullMark: 150 },
];

export function NutritionRadar() {
    return (
        <div className="w-full h-64 relative">
            {/* Chart Container */}
            <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
                    <PolarGrid stroke="#2F3325" />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#888888', fontSize: 10, fontFamily: 'var(--font-mono)' }} />
                    <PolarRadiusAxis angle={30} domain={[0, 150]} tick={false} axisLine={false} />
                    <Radar
                        name="Millet"
                        dataKey="A"
                        stroke="#00F0FF"
                        strokeWidth={2}
                        fill="#00F0FF"
                        fillOpacity={0.2}
                    />
                </RadarChart>
            </ResponsiveContainer>

            {/* Label */}
            <div className="absolute top-0 right-0">
                <span className="text-[10px] text-grain font-mono tracking-widest border border-grain/20 px-1 rounded">RAGI-01</span>
            </div>
        </div>
    );
}
