import Link from "next/link";
import { User, Bell, Settings } from "lucide-react";

export function Navbar() {
    return (
        <nav className="w-full bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16 items-center">
                    {/* Logo */}
                    <Link href="/" className="flex items-center gap-3 hover:opacity-90 transition-opacity">
                        <div className="w-8 h-8 rounded-lg bg-deep-green flex items-center justify-center shadow-sm">
                            <span className="text-white font-bold text-lg">M</span>
                        </div>
                        <span className="text-xl font-bold tracking-tight text-deep-green">
                            MilletGAI
                        </span>
                    </Link>

                    {/* Navigation Links - Removed as per request */}
                    <div className="hidden md:flex gap-8 text-sm font-medium text-charcoal/80">
                        {/* Placeholder for future links */}
                    </div>

                    {/* User Controls */}
                    <div className="flex items-center gap-4 text-charcoal/60">
                        <div className="h-8 w-px bg-gray-200 mx-1"></div>
                        <button className="flex items-center gap-2 hover:bg-gray-100 px-2 py-1/5 rounded-full transition-colors">
                            <div className="w-8 h-8 rounded-full bg-olive flex items-center justify-center text-white">
                                <User size={16} />
                            </div>
                        </button>
                    </div>
                </div>
            </div>
        </nav>
    );
}
