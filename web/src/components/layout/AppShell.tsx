'use client'

import { UploadProgressPanel } from '@/components/uploads/UploadProgressPanel'

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="h-dvh overflow-hidden bg-background">
      <main className="h-full overflow-y-auto">{children}</main>
      <UploadProgressPanel />
    </div>
  )
}
