import type { Metadata } from 'next'
import { SignupForm } from './SignupForm'

export const metadata: Metadata = {
  title: '注册 | LLM Wiki',
  description: "免费注册 LLM Wiki 账号。上传文档,由 AI 智能体构建持续积累的维基。",
  openGraph: {
    title: '注册 | LLM Wiki',
    description: "免费注册 LLM Wiki 账号。上传文档,由 AI 智能体构建持续积累的维基。",
  },
}

export default function SignupPage() {
  return <SignupForm />
}
