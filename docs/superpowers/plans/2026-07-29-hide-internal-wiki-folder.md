# Hide Internal Wiki Folder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the internal `/wiki/` folder from ever appearing in the original-files browser while preserving normal source folders and wiki page reading.

**Architecture:** Keep the backend source-only query as the primary invariant and add defense-in-depth filtering at the `useDocumentBrowse` client boundary. `FilesGrid` continues to render only the folders supplied by the hook, so no component-specific exception or duplicate filtering is required.

**Tech Stack:** React 19, Next.js 16, TypeScript, Vitest, Testing Library, Docker

---

## File structure

- Modify `web/src/hooks/useDocumentBrowse.ts`: normalize folder paths and remove the internal `/wiki/` folder before exposing browse state.
- Modify `web/src/hooks/__tests__/useDocumentBrowse.test.tsx`: preserve the stale-folder regression and add the internal-folder regression.
- No `FilesGrid` change: it remains a presentation component and receives an already-safe folder collection.

### Task 1: Enforce the source-folder boundary in the browse hook

**Files:**
- Modify: `web/src/hooks/useDocumentBrowse.ts:64-69`
- Test: `web/src/hooks/__tests__/useDocumentBrowse.test.tsx:31-49`

- [ ] **Step 1: Keep the existing stale-folder test focused on stale state**

Replace its `wiki` fixture with an ordinary folder so the test continues to cover refresh behavior independently:

```tsx
it('clears stale folders when a refresh returns an empty folder list', async () => {
  const reports = { name: 'reports', path: '/reports/', document_count: 2 }
  vi.stubGlobal(
    'fetch',
    vi.fn()
      .mockResolvedValueOnce(response(page([reports])))
      .mockResolvedValueOnce(response(page([]))),
  )
  const { result } = renderHook(() =>
    useDocumentBrowse({ kbId: 'kb-1', token: 'token', path: '/', pollInterval: 0 }),
  )
  await waitFor(() => expect(result.current.folders).toEqual([reports]))

  await act(async () => result.current.reload())

  await waitFor(() => expect(result.current.folders).toEqual([]))
})
```

- [ ] **Step 2: Write the failing internal-folder test**

Add a test that supplies both slash variants and a legitimate source folder:

```tsx
it('hides the internal wiki folder from source browsing', async () => {
  const reports = { name: 'reports', path: '/reports/', document_count: 3 }
  const internalFolders = [
    { name: 'wiki', path: '/wiki/', document_count: 2 },
    { name: 'wiki', path: '/wiki', document_count: 2 },
  ]
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(response(page([reports, ...internalFolders]))),
  )

  const { result } = renderHook(() =>
    useDocumentBrowse({ kbId: 'kb-1', token: 'token', path: '/', pollInterval: 0 }),
  )

  await waitFor(() => expect(result.current.folders).toEqual([reports]))
})
```

- [ ] **Step 3: Run the focused test and verify RED**

Run:

```bash
cd web
npm test -- --run src/hooks/__tests__/useDocumentBrowse.test.tsx
```

Expected: the new test fails because `result.current.folders` still contains the two `/wiki` fixtures.

- [ ] **Step 4: Add the minimal folder filter**

Update the page-to-state effect in `useDocumentBrowse`:

```tsx
React.useEffect(() => {
  if (!page || (read.hasPrevious && page.folders.length === 0)) return
  const visibleFolders = page.folders.filter((folder) => {
    const normalizedPath = `/${folder.path.split('/').filter(Boolean).join('/')}/`
    return normalizedPath !== '/wiki/'
  })
  setFolders(visibleFolders)
}, [page, read.hasPrevious])
```

This comparison hides only the exact normalized internal path. `/my-wiki/` and `/sources/wiki/` remain visible.

- [ ] **Step 5: Run the focused test and verify GREEN**

Run:

```bash
cd web
npm test -- --run src/hooks/__tests__/useDocumentBrowse.test.tsx
```

Expected: both `useDocumentBrowse` tests pass.

- [ ] **Step 6: Run all frontend checks**

Run:

```bash
cd web
npm test -- --run
npm run build
```

Expected: all Vitest tests pass and Next.js exits with status 0. Existing CSS `::highlight` and Sentry deprecation warnings may remain, but no new error is introduced.

- [ ] **Step 7: Commit the implementation**

```bash
git add web/src/hooks/useDocumentBrowse.ts web/src/hooks/__tests__/useDocumentBrowse.test.tsx
git commit -m "fix: hide internal wiki folder from file browser"
```

### Task 2: Validate and publish the running self-hosted build

**Files:**
- No source changes
- Preserve untracked local data: `workspace-branch-test/`

- [ ] **Step 1: Build the local image**

Run from the repository root:

```bash
docker build -f Dockerfile.local -t llmwiki-local:platform-architecture-evolution .
```

Expected: image build exits with status 0.

- [ ] **Step 2: Replace the local container without replacing its workspace**

Run:

```bash
docker stop llmwiki
docker rm llmwiki
docker run -d --name llmwiki --restart unless-stopped \
  -p 127.0.0.1:3000:3000 \
  -p 127.0.0.1:9000:8000 \
  -p 127.0.0.1:8080:8080 \
  -e PUBLIC_API_URL=http://localhost:9000 \
  -e PUBLIC_WEB_URL=http://localhost:3000 \
  -e PUBLIC_MCP_URL=http://localhost:8080/mcp \
  -v /Users/benjamin/Documents/Codex/2026-07-24/github-plugin-github-openai-api-curated-4/work/llmwiki-goglobal/workspace-branch-test:/workspace \
  llmwiki-local:platform-architecture-evolution
```

Expected: the replacement container starts with the same workspace bind mount.

- [ ] **Step 3: Check container and API health**

Run:

```bash
docker inspect llmwiki --format '{{.State.Health.Status}}'
curl -fsS http://localhost:9000/health
```

Expected: Docker reports `healthy` and the API returns `{"status":"ok"}`.

- [ ] **Step 4: Verify the rendered target flow**

Using the available Browser runtime, navigate to
`http://localhost:3000/wikis/workspace/files`, reload once, and verify:

```text
page identity: URL ends with /wikis/workspace/files and title is LLM Wiki
content: original-files controls or its empty state render
folder invariant: no folder card named wiki appears
health: no framework overlay and no relevant console warnings or errors
interaction: reload preserves the hidden state
```

- [ ] **Step 5: Push the current branch**

```bash
git push -u origin feat/platform-architecture-evolution
```

Expected: `origin/feat/platform-architecture-evolution` points at the local implementation commit and `workspace-branch-test/` remains untracked.
