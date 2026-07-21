import React, { useEffect, useState, useRef } from "react";
import type { KnowledgeBase } from "@/lib/api";
import { fetchKnowledgeBases, createKnowledgeBase } from "@/lib/api";

interface Props {
  apiUrl: string;
  accessToken: string | null;
  value: string | null;
  onChange: (id: string) => void;
}

export default function KBPicker({ apiUrl, accessToken, value, onChange }: Props) {
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadKBs();
  }, [apiUrl, accessToken]);

  useEffect(() => {
    if (creating) inputRef.current?.focus();
  }, [creating]);

  async function loadKBs() {
    setLoading(true);
    setError(null);
    try {
      const list = await fetchKnowledgeBases(apiUrl, accessToken);
      setKbs(list);
      const selectionValid = value !== null && list.some((kb) => kb.id === value);
      if (!selectionValid && list.length > 0) {
        onChange(list[0].id);
      }
    } catch {
      setError("加载知识库失败");
    } finally {
      setLoading(false);
    }
  }

  async function handleCreate() {
    const name = newName.trim();
    if (!name) return;
    try {
      const kb = await createKnowledgeBase(apiUrl, accessToken, name);
      setKbs((prev) => [kb, ...prev]);
      onChange(kb.id);
      setCreating(false);
      setNewName("");
    } catch {
      setError("创建知识库失败");
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter") { e.preventDefault(); handleCreate(); }
    if (e.key === "Escape") { setCreating(false); setNewName(""); }
  }

  if (loading) {
    return <div className="py-1 text-xs text-zinc-500">正在加载知识库...</div>;
  }

  return (
    <div className="space-y-1.5">
      <label className="block text-xs font-medium text-zinc-700">知识库</label>

      {!creating ? (
        <div className="flex gap-2">
          <select
            value={value ?? ""}
            onChange={(e) => { if (e.target.value) onChange(e.target.value); }}
            className="h-9 min-w-0 flex-1 rounded-md border border-zinc-200 bg-white px-3
                       text-sm text-zinc-950 shadow-sm outline-none transition-colors
                       focus:border-zinc-400 focus:ring-2 focus:ring-zinc-950/10"
          >
            {kbs.length === 0 && (
              <option value="" disabled>暂无知识库——请先创建</option>
            )}
            {kbs.map((kb) => (
              <option key={kb.id} value={kb.id}>{kb.name}</option>
            ))}
          </select>
          <button
            onClick={() => setCreating(true)}
            className="h-9 rounded-md border border-zinc-200 bg-white px-3 text-xs
                       font-medium text-zinc-700 shadow-sm transition-colors
                       hover:bg-zinc-100 hover:text-zinc-950
                       focus-visible:outline-none focus-visible:ring-2
                       focus-visible:ring-zinc-950 focus-visible:ring-offset-2
                       whitespace-nowrap"
          >
            + 新建
          </button>
        </div>
      ) : (
        <div className="flex gap-2">
          <input
            ref={inputRef}
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="知识库名称"
            className="h-9 min-w-0 flex-1 rounded-md border border-zinc-200 bg-white px-3
                       text-sm text-zinc-950 shadow-sm outline-none transition-colors
                       placeholder:text-zinc-400 focus:border-zinc-400
                       focus:ring-2 focus:ring-zinc-950/10"
          />
          <button
            onClick={handleCreate}
            disabled={!newName.trim()}
            className="h-9 rounded-md bg-zinc-950 px-3 text-xs font-medium text-zinc-50
                       transition-colors hover:bg-zinc-800 disabled:cursor-not-allowed
                       disabled:opacity-40"
          >
            添加
          </button>
          <button
            onClick={() => { setCreating(false); setNewName(""); }}
            className="h-9 rounded-md px-2 text-xs font-medium text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900"
          >
            取消
          </button>
        </div>
      )}

      {error && <p className="text-xs text-red-600">{error}</p>}
    </div>
  );
}
