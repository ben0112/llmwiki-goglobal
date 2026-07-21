import React, { useState } from "react";

interface Props {
  onPasswordSignIn: (email: string, password: string) => void;
}

export default function AuthGate({ onPasswordSignIn }: Props) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onPasswordSignIn(email.trim(), password);
  }

  return (
    <div className="flex flex-col gap-4 py-2">
      <p className="max-w-[260px] text-center text-sm leading-5 text-zinc-500">
        Sign in to save pages to your knowledge base
      </p>

      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="space-y-1.5">
          <label htmlFor="llmwiki-email" className="text-xs font-medium text-zinc-600">
            Email
          </label>
          <input
            id="llmwiki-email"
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            autoComplete="email"
            required
            className="h-9 w-full rounded-md border border-zinc-200 bg-white px-3 text-sm text-zinc-950 shadow-sm outline-none transition-colors placeholder:text-zinc-400 focus:border-zinc-400 focus:ring-2 focus:ring-zinc-200"
          />
        </div>
        <div className="space-y-1.5">
          <label htmlFor="llmwiki-password" className="text-xs font-medium text-zinc-600">
            Password
          </label>
          <input
            id="llmwiki-password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            required
            className="h-9 w-full rounded-md border border-zinc-200 bg-white px-3 text-sm text-zinc-950 shadow-sm outline-none transition-colors placeholder:text-zinc-400 focus:border-zinc-400 focus:ring-2 focus:ring-zinc-200"
          />
        </div>
        <button
          type="submit"
          className="inline-flex h-9 w-full items-center justify-center rounded-md bg-zinc-950 px-3 text-sm font-medium text-white shadow-sm transition-colors hover:bg-zinc-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-950 focus-visible:ring-offset-2"
        >
          Sign in
        </button>
      </form>
    </div>
  );
}
