import Link from 'next/link'
import { ArrowRight, BookOpen, FileText, PenTool, Search, GitBranch } from 'lucide-react'
import { AuthRedirect } from './AuthRedirect'
import { MotionDiv, MotionP } from './LandingMotion'

const ease: [number, number, number, number] = [0.16, 1, 0.3, 1]

const WIKI_TREE = [
  { label: '概览', active: true, depth: 0 },
  { label: '概念', depth: 0, folder: true },
  { label: '注意力机制', depth: 1 },
  { label: '缩放定律', depth: 1 },
  { label: '实体', depth: 0, folder: true },
  { label: 'Transformer 架构', depth: 1 },
  { label: '源文件', depth: 0, folder: true },
  { label: '日志', depth: 0 },
]

const jsonLd = {
  '@context': 'https://schema.org',
  '@type': 'SoftwareApplication',
  name: 'LLM Wiki',
  applicationCategory: 'ProductivityApplication',
  operatingSystem: 'Web',
  offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },
  url: 'https://llmwiki.app',
  description:
    "Karpathy LLM Wiki 的免费开源实现。上传文档,由 AI 智能体直接构建持续积累的维基。",
}

export default function LandingPage() {
  return (
    <div className="min-h-svh bg-background text-foreground">
      <AuthRedirect />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      {/* Nav */}
      <nav className="fixed top-0 inset-x-0 z-50 flex items-center justify-between px-6 lg:px-10 h-14 bg-background/80 backdrop-blur-sm">
        <span className="flex items-center gap-2.5 text-sm font-semibold tracking-tight">
          <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 32 32">
            <rect width="32" height="32" rx="7" fill="currentColor" className="text-foreground" />
            <polyline points="11,8 21,16 11,24" fill="none" stroke="currentColor" className="text-background" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          LLM Wiki
        </span>
        <div className="flex items-center gap-5">
          <Link
            href="https://github.com/lucasastorian/llmwiki"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            GitHub
          </Link>
          <Link
            href="/login"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            登录
          </Link>
          <Link
            href="/signup"
            className="hidden sm:inline-flex items-center gap-1.5 rounded-full bg-foreground text-background px-4 py-1.5 text-sm font-medium hover:opacity-90 transition-opacity"
          >
            开始使用
          </Link>
        </div>
      </nav>

      {/* Hero */}
      <section className="pt-32 pb-20 px-6 lg:px-10">
        <div className="max-w-2xl mx-auto text-center">
          <MotionDiv
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, ease }}
          >
            <p className="text-sm text-muted-foreground mb-4">
              开源实现,基于{' '}
              <Link
                href="https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f"
                className="text-foreground underline underline-offset-2 decoration-foreground/30 hover:decoration-foreground transition-colors"
              >
                Karpathy&apos;s LLM&nbsp;Wiki
              </Link>
            </p>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight leading-[1.05]">
              LLM Wiki
            </h1>
          </MotionDiv>

          <MotionP
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.12, ease }}
            className="mt-6 text-base sm:text-lg text-muted-foreground max-w-md mx-auto leading-relaxed"
          >
            让 LLM 从原始资料中编译并维护一个结构化的维基。
          </MotionP>

          <MotionDiv
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7, delay: 0.25, ease }}
            className="mt-9 flex items-center justify-center gap-3"
          >
            <Link
              href="/signup"
              className="inline-flex items-center gap-2 rounded-full bg-foreground text-background px-6 py-2.5 text-sm font-medium hover:opacity-90 transition-opacity"
            >
              开始使用
              <ArrowRight className="size-3.5 opacity-60" />
            </Link>
            <Link
              href="https://github.com/lucasastorian/llmwiki"
              className="inline-flex items-center gap-2 rounded-full border border-border px-6 py-2.5 text-sm font-medium hover:bg-accent transition-colors"
            >
              GitHub
            </Link>
          </MotionDiv>
        </div>
      </section>

      {/* Product Preview */}
      <section className="px-6 lg:px-10 pb-28">
        <MotionDiv
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.9, delay: 0.4, ease }}
          className="max-w-5xl mx-auto"
        >
          <div className="bg-card rounded-2xl border border-border shadow-lg overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-3 border-b border-border bg-muted/30">
              <div className="flex gap-1.5">
                <div className="size-2.5 rounded-full bg-border" />
                <div className="size-2.5 rounded-full bg-border" />
                <div className="size-2.5 rounded-full bg-border" />
              </div>
              <div className="flex-1 flex justify-center">
                <span className="text-xs text-muted-foreground/50 font-mono">
                  llmwiki.app
                </span>
              </div>
              <div className="w-14" />
            </div>

            <div className="flex min-h-[400px]">
              {/* Sidebar */}
              <div className="w-52 shrink-0 border-r border-border p-3 hidden sm:block">
                <div className="flex items-center gap-2 px-2 py-1.5 mb-2">
                  <Search className="size-3 text-muted-foreground/30" />
                  <span className="text-xs text-muted-foreground/30">搜索维基...</span>
                </div>
                <div className="space-y-0.5">
                  {WIKI_TREE.map((item, i) => (
                    <div
                      key={i}
                      className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs ${
                        item.active
                          ? 'bg-accent font-medium text-foreground'
                          : 'text-muted-foreground'
                      }`}
                      style={{ paddingLeft: `${item.depth * 14 + 8}px` }}
                    >
                      {item.folder ? (
                        <GitBranch className="size-3 opacity-40" />
                      ) : (
                        <FileText className="size-3 opacity-40" />
                      )}
                      {item.label}
                    </div>
                  ))}
                </div>
              </div>

              {/* Content */}
              <div className="flex-1 p-8 sm:p-10">
                <div className="max-w-lg">
                  <h2 className="text-xl font-semibold tracking-tight mb-1">概览</h2>
                  <p className="text-xs text-muted-foreground mb-6">
                    12 个源文件 &middot; 最近更新于 2 小时前
                  </p>
                  <p className="text-sm text-muted-foreground leading-relaxed mb-3">
                    本维基追踪 Transformer 架构及其缩放特性的研究,
                    综合了 <span className="font-medium text-foreground">12 个源文件</span>、共 47 页的研究发现。
                  </p>
                  <h3 className="text-sm font-semibold mt-5 mb-2">关键发现</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed mb-3">
                    模型规模与性能之间的关系遵循可预测的{' '}
                    <span className="font-medium text-foreground">缩放定律</span> &mdash;
                    损失随算力、数据集规模和参数量呈幂律下降。
                  </p>
                  <h3 className="text-sm font-semibold mt-5 mb-2">最近更新</h3>
                  <ul className="space-y-1 ml-4">
                    <li className="text-sm text-muted-foreground list-disc">新增稀疏注意力变体的分析</li>
                    <li className="text-sm text-muted-foreground list-disc">用新基准更新了缩放定律</li>
                    <li className="text-sm text-muted-foreground list-disc">标记了 Chen 等人与 Wei 等人研究间的矛盾</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </MotionDiv>
      </section>

      {/* Divider */}
      <div className="max-w-5xl mx-auto border-t border-border" />

      {/* Three Layers */}
      <section className="px-6 lg:px-10 py-24">
        <div className="max-w-5xl mx-auto">
          <MotionDiv
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.6 }}
            className="text-center mb-14"
          >
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">三层结构</h2>
            <p className="mt-3 text-muted-foreground max-w-md mx-auto">
              您几乎不需要亲自撰写维基 &mdash; 维基是 LLM 的领地。
            </p>
          </MotionDiv>

          <div className="grid sm:grid-cols-3 gap-6">
            {[
              {
                icon: FileText,
                title: '原始资料',
                body: '文章、论文、笔记、转录稿。您不可变的事实来源。LLM 只读取它们,绝不修改。',
              },
              {
                icon: BookOpen,
                title: '维基',
                body: '由 LLM 生成的 Markdown 页面,包含摘要、实体页面和交叉引用。这一层归 LLM 所有:您来阅读,LLM 来撰写。',
              },
              {
                icon: PenTool,
                title: '架构约定',
                body: '一个配置文件,告诉 LLM 维基的结构、要遵循的约定,以及收录时要运行的工作流。',
              },
            ].map((item, i) => (
              <MotionDiv
                key={item.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-50px' }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                className="bg-card rounded-xl border border-border p-6"
              >
                <item.icon className="size-5 text-muted-foreground mb-4" strokeWidth={1.5} />
                <h3 className="font-semibold text-sm mb-2">{item.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{item.body}</p>
              </MotionDiv>
            ))}
          </div>
        </div>
      </section>

      {/* Divider */}
      <div className="max-w-5xl mx-auto border-t border-border" />

      {/* How It Works */}
      <section className="px-6 lg:px-10 py-24">
        <div className="max-w-5xl mx-auto">
          <MotionDiv
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.6 }}
            className="text-center mb-14"
          >
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">工作方式</h2>
          </MotionDiv>

          <div className="grid sm:grid-cols-3 gap-10 sm:gap-8">
            {[
              {
                step: '01',
                title: '收录',
                body: '把资料放进 raw/。LLM 会阅读它、撰写摘要、更新维基中的实体页和概念页,并标记与既有知识矛盾之处。一份资料可能会涉及 10\u201315 个维基页面。',
              },
              {
                step: '02',
                title: '查询',
                body: '基于编译好的维基提出复杂问题。知识已经完成综合 \u2014 无需每次从原始片段重新推导。好的答案会归档为新页面,让您的探索不断积累。',
              },
              {
                step: '03',
                title: '体检',
                body: '对维基进行健康检查:找出不一致的数据、过时的论断、孤立页面和缺失的交叉引用。LLM 会建议值得追问的新问题和值得寻找的新资料。',
              },
            ].map((item, i) => (
              <MotionDiv
                key={item.step}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-50px' }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
              >
                <span className="text-xs font-mono text-muted-foreground/40 mb-3 block">{item.step}</span>
                <h3 className="font-semibold mb-2">{item.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{item.body}</p>
              </MotionDiv>
            ))}
          </div>
        </div>
      </section>

      {/* Divider */}
      <div className="max-w-5xl mx-auto border-t border-border" />

      {/* Quote */}
      <section className="px-6 lg:px-10 py-24">
        <MotionDiv
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.8 }}
          className="max-w-2xl mx-auto text-center"
        >
          <blockquote className="text-lg sm:text-xl leading-relaxed text-foreground/80 italic">
            &ldquo;维护知识库最繁琐的不是阅读或思考,而是琐碎的记录工作。LLM 不会厌倦,不会忘记更新交叉引用,并且一次就能修改 15 个文件。&rdquo;
          </blockquote>
          <p className="mt-5 text-sm text-muted-foreground">
            Andrej Karpathy
          </p>
        </MotionDiv>
      </section>

      {/* Divider */}
      <div className="max-w-5xl mx-auto border-t border-border" />

      {/* CTA */}
      <section className="px-6 lg:px-10 py-24">
        <MotionDiv
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.6 }}
          className="max-w-md mx-auto text-center"
        >
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight mb-4">开始构建您的维基</h2>
          <p className="text-muted-foreground mb-8">
            一个出色的产品,而不是一堆拼凑的脚本。
          </p>
          <Link
            href="/signup"
            className="inline-flex items-center gap-2 rounded-full bg-foreground text-background px-7 py-3 text-sm font-medium hover:opacity-90 transition-opacity"
          >
            免费开始使用
            <ArrowRight className="size-3.5 opacity-60" />
          </Link>
        </MotionDiv>
      </section>

      {/* Footer */}
      <footer className="border-t border-border px-6 lg:px-10 py-6 flex items-center justify-between text-xs text-muted-foreground/50">
        <span>LLM Wiki</span>
        <div className="flex items-center gap-4">
          <Link href="/terms" className="hover:text-muted-foreground transition-colors">服务条款</Link>
          <Link href="/privacy" className="hover:text-muted-foreground transition-colors">隐私政策</Link>
          <Link href="/dmca" className="hover:text-muted-foreground transition-colors">DMCA</Link>
          <span>免费开源 &middot; Apache 2.0</span>
        </div>
      </footer>
    </div>
  )
}
