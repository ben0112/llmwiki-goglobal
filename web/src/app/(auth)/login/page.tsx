import type { Metadata } from 'next'
import { LoginForm } from './LoginForm'

export const metadata: Metadata = {
  title: '登录 | LLM Wiki',
  description: '登录 LLM Wiki,管理您的知识库和维基。',
  openGraph: {
    title: '登录 | LLM Wiki',
    description: '登录 LLM Wiki,管理您的知识库和维基。',
  },
}

export default function LoginPage() {
  return <LoginForm />
}
