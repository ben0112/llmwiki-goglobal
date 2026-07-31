import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const fixtures = vi.hoisted(() => ({
  knowledgeBases: [
    {
      id: 'kb-1',
      user_id: 'user-1',
      name: '测试空间',
      slug: '测试空间',
      description: null,
      source_count: 0,
      wiki_page_count: 2,
      created_at: '2026-07-29T00:00:00Z',
      updated_at: '2026-07-29T00:00:00Z',
      kind: 'wiki',
    },
  ],
  user: { id: 'user-1', email: 'local@localhost' },
  router: { push: vi.fn(), replace: vi.fn() },
}))

vi.mock('next/navigation', () => ({
  useParams: () => ({ slug: '%E6%B5%8B%E8%AF%95%E7%A9%BA%E9%97%B4' }),
  useRouter: () => fixtures.router,
  useSearchParams: () => new URLSearchParams(),
}))

vi.mock('@/stores', () => ({
  useKBStore: (selector: (state: unknown) => unknown) =>
    selector({ knowledgeBases: fixtures.knowledgeBases, loading: false }),
  useUserStore: (selector: (state: unknown) => unknown) =>
    selector({ user: fixtures.user, accessToken: 'local-token' }),
}))

vi.mock('@/hooks/useWikiPages', () => ({
  useWikiPages: () => ({ documents: [], loading: false }),
}))

vi.mock('@/components/kb/WikiOnlyDetail', () => ({
  WikiOnlyDetail: ({ kbName }: { kbName: string }) => (
    <div data-testid="wiki-detail">{kbName}</div>
  ),
}))

vi.mock('@/components/kb/KBDetail', () => ({
  KBDetail: ({ kbName }: { kbName: string }) => (
    <div data-testid="kb-detail">{kbName}</div>
  ),
}))

vi.mock('@/components/corpus/CorpusView', () => ({
  CorpusView: ({ kbName }: { kbName: string }) => (
    <div data-testid="corpus-view">{kbName}</div>
  ),
}))

import WikiPage from '@/app/(dashboard)/wikis/[slug]/[[...path]]/page'
import CorpusPage from '@/app/(dashboard)/wikis/[slug]/corpus/page'
import { KBDetailRoutePage } from '@/components/kb/KBDetailRoutePage'

afterEach(cleanup)

describe('wiki routes with renamed Unicode slugs', () => {
  it('opens the wiki overview from an encoded route slug', () => {
    render(<WikiPage />)

    expect(screen.getByTestId('wiki-detail').textContent).toBe('测试空间')
  })

  it('opens the file and graph routes from an encoded route slug', () => {
    render(<KBDetailRoutePage viewMode="files" />)

    expect(screen.getByTestId('kb-detail').textContent).toBe('测试空间')
  })

  it('opens the corpus route from an encoded route slug', () => {
    render(<CorpusPage />)

    expect(screen.getByTestId('corpus-view').textContent).toBe('测试空间')
  })
})
