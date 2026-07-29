# Hide the internal wiki folder from source browsing

## Goal

The original-files view at `/wikis/[slug]/files` must never render the internal
`/wiki/` folder. Generated wiki pages remain available through the wiki reader
and are not deleted or reclassified.

## Design

`useDocumentBrowse` is the client-side boundary for source browsing. It will
discard any `ReadFolder` whose normalized path is `/wiki/` before exposing the
folder list to consumers. Filtering at this boundary protects every source-file
view from stale caches or older server responses while keeping `FilesGrid`
focused on rendering its input.

The existing backend filtering remains the primary invariant. This client-side
filter is defense in depth and does not change API payloads, persistence, wiki
page loading, source counts, uploads, or direct wiki reading.

## Edge cases

- A user folder with a different path or a name merely containing `wiki` stays
  visible.
- Both `/wiki` and `/wiki/` normalize to the internal path and are hidden.
- An empty filtered result replaces previously displayed folders so stale
  internal entries cannot remain on screen.

## Verification

Extend the `useDocumentBrowse` tests with a response containing both `/wiki/`
and a normal source folder. The hook must expose only the normal folder. Run the
full frontend test suite, production build, and rendered browser check against
`http://localhost:3000/wikis/workspace/files`.
