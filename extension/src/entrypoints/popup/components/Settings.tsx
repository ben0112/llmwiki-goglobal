import React, { useEffect, useRef, useState } from "react";
import {
  checkLocalHealth,
  getMode,
  setMode,
  getLocalUrl,
  normalizeApiUrl,
  setLocalUrl,
  type Mode,
} from "@/lib/settings";

interface Props {
  onBack: () => void;
  onModeChange: (mode: Mode) => void;
  isSignedIn: boolean;
  onSignOut: () => void;
}

export default function Settings({ onBack, onModeChange, isSignedIn, onSignOut }: Props) {
  const [mode, setModeState] = useState<Mode>("cloud");
  const [localUrl, setLocalUrlState] = useState("http://localhost:8000");
  const [saved, setSaved] = useState(false);
  const [checking, setChecking] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [showLocalConfig, setShowLocalConfig] = useState(false);
  const flashTimer = useRef<number | null>(null);
  const validatingRef = useRef(false);

  useEffect(() => {
    getMode().then((storedMode) => {
      setModeState(storedMode);
      setShowLocalConfig(storedMode === "local");
    });
    getLocalUrl().then(setLocalUrlState);
    return () => {
      if (flashTimer.current) window.clearTimeout(flashTimer.current);
    };
  }, []);

  async function handleModeChange(newMode: Mode) {
    if (newMode === "local") {
      setModeState("local");
      setShowLocalConfig(true);
      setMessage("Enter your local API URL. Click away to test /health.");
      setSaved(false);
      return;
    }
    setModeState(newMode);
    setShowLocalConfig(false);
    setMessage(null);
    setSaved(false);
    await setMode(newMode);
    onModeChange(newMode);
    flash("Settings saved");
  }

  async function handleUrlSave() {
    await connectLocal(localUrl);
  }

  async function handleBack() {
    if (!showLocalConfig) {
      onBack();
      return;
    }
    const connected = await connectLocal(localUrl);
    if (connected) onBack();
  }

  async function connectLocal(url: string): Promise<boolean> {
    if (validatingRef.current) return false;
    validatingRef.current = true;
    const normalized = normalizeApiUrl(url);
    setChecking(true);
    setMessage(null);
    setSaved(false);
    const connected = await checkLocalHealth(normalized);
    validatingRef.current = false;
    setChecking(false);
    if (!connected) {
      await setMode("cloud");
      setModeState("cloud");
      setShowLocalConfig(true);
      onModeChange("cloud");
      setMessage(`Could not connect to ${normalized}/health. Cloud remains active.`);
      return false;
    }
    await setLocalUrl(normalized);
    await setMode("local");
    setLocalUrlState(normalized);
    setModeState("local");
    onModeChange("local");
    flash(`Connected to ${normalized}`);
    return true;
  }

  function flash(nextMessage: string) {
    if (flashTimer.current) window.clearTimeout(flashTimer.current);
    setMessage(nextMessage);
    setSaved(true);
    flashTimer.current = window.setTimeout(() => {
      setSaved(false);
      setMessage(null);
      flashTimer.current = null;
    }, 1800);
  }

  return (
    <div className="space-y-4">
      <button
        onClick={handleBack}
        disabled={checking}
        className="rounded-md px-2 py-1 text-xs font-medium text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900"
      >
        &larr; Back
      </button>

      <div>
        <label className="mb-2 block text-xs font-medium text-zinc-700">Mode</label>
        <div className="grid grid-cols-2 gap-1 rounded-md border border-zinc-200 bg-zinc-100 p-1">
          <button
            onClick={() => handleModeChange("cloud")}
            disabled={checking}
            className={`h-8 rounded-sm px-3 text-sm font-medium transition-colors ${
              mode === "cloud"
                ? "bg-white text-zinc-950 shadow-sm"
                : "text-zinc-500 hover:text-zinc-900"
            }`}
          >
            Cloud
          </button>
          <button
            onClick={() => handleModeChange("local")}
            disabled={checking}
            className={`h-8 rounded-sm px-3 text-sm font-medium transition-colors ${
              mode === "local"
                ? "bg-white text-zinc-950 shadow-sm"
                : "text-zinc-500 hover:text-zinc-900"
            }`}
          >
            Local
          </button>
        </div>
        <p className="mt-1.5 text-[11px] leading-4 text-zinc-500">
          {mode === "cloud"
            ? showLocalConfig
              ? "Cloud stays active until the local URL connects"
              : "Saves to llmwiki.app, requires sign in"
            : "Saves to your local LLM Wiki instance, no sign in needed"}
        </p>
      </div>

      {showLocalConfig && (
        <div>
          <label className="mb-1.5 block text-xs font-medium text-zinc-700">
            API URL
          </label>
          <div className="flex gap-2">
            <input
              value={localUrl}
              onChange={(e) => {
                setLocalUrlState(e.target.value);
                setSaved(false);
                setMessage("Click away to test /health.");
              }}
              onBlur={() => {
                if (!checking) handleUrlSave();
              }}
              onKeyDown={(e) => { if (e.key === "Enter") handleUrlSave(); }}
              disabled={checking}
              className="h-9 min-w-0 flex-1 rounded-md border border-zinc-200 bg-white px-3
                         font-mono text-xs text-zinc-950 shadow-sm outline-none
                         transition-colors focus:border-zinc-400 focus:ring-2
                         focus:ring-zinc-950/10"
              placeholder="http://localhost:8000"
            />
            <button
              onClick={handleUrlSave}
              disabled={checking}
              className="h-9 rounded-md bg-zinc-950 px-3 text-xs font-medium text-zinc-50
                         transition-colors hover:bg-zinc-800 disabled:cursor-default disabled:opacity-60"
            >
              {checking ? "Checking" : "Test"}
            </button>
          </div>
        </div>
      )}

      {message && (
        <p className={`text-xs ${saved ? "text-emerald-700" : "text-red-700"}`}>
          {message}
        </p>
      )}

      {isSignedIn && (
        <div className="border-t border-zinc-200 pt-4">
          <button
            onClick={onSignOut}
            className="h-9 w-full rounded-md border border-zinc-300 bg-white px-4 text-sm
                       font-medium text-zinc-700 shadow-sm transition-colors
                       hover:border-zinc-400 hover:bg-zinc-50
                       focus-visible:outline-none focus-visible:ring-2
                       focus-visible:ring-zinc-950 focus-visible:ring-offset-2"
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
