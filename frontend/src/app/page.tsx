"use client";

import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { Dashboard, Feature } from "@/components/Dashboard";
import { ChatView } from "@/components/ChatView";

export default function Home() {
  const [view, setView] = useState<"dashboard" | "chat">("dashboard");
  const [activeContext, setActiveContext] = useState<string>("General Query");
  const [activeTheme, setActiveTheme] = useState<string>("gray");
  const [initialQuery, setInitialQuery] = useState<string>("");
  const [mode, setMode] = useState<"fast" | "thinking">("fast");

  const handleFeatureSelect = (feature: Feature, query?: string) => {
    setActiveContext(feature.id);
    setActiveTheme(feature.theme);
    setInitialQuery(query || "");
    setView("chat");
  };

  const handleBackToDashboard = () => {
    setView("dashboard");
    setInitialQuery("");
  };

  return (
    <div className="min-h-screen">
      <Navbar />

      <main className="container mx-auto px-4">
        {view === "dashboard" ? (
          <Dashboard
            onFeatureSelect={handleFeatureSelect}
            mode={mode}
            onModeChange={setMode}
          />
        ) : (
          <ChatView
            context={activeContext}
            theme={activeTheme}
            initialQuery={initialQuery}
            onBack={handleBackToDashboard}
            initialMode={mode}
            onModeChange={setMode}
          />
        )}
      </main>
    </div>
  );
}
