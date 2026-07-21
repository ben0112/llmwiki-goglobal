const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const DEFAULT_MCP_URL = 'http://localhost:8080/mcp'

export const MCP_URL =
  process.env.NEXT_PUBLIC_MCP_URL ||
  (process.env.NEXT_PUBLIC_API_URL ? `${API_URL}/mcp` : DEFAULT_MCP_URL)

export function buildApiKeyMcpConfig(apiKey: string): string {
  return JSON.stringify(
    {
      mcpServers: {
        llmwiki: {
          url: MCP_URL,
          headers: {
            Authorization: `Bearer ${apiKey}`,
          },
        },
      },
    },
    null,
    2,
  )
}
