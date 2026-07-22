export const privacy = `# Privacy Policy

**LLM Wiki** · Effective date: June 3, 2026

LLM Wiki is operated by Polybius, L.L.C., a Delaware limited liability company ("Polybius," "we," "us," "our"). LLM Wiki is a free, open-source knowledge base service available at llmwiki.app. This policy explains what data we collect, how we use it, and your rights regarding that data.

## What we collect

### Account information
When you sign up, we collect your email address and display name via Supabase Auth.

### Content you upload
Documents, notes, PDFs, and other files you add to your knowledge bases are stored on our infrastructure. This includes the original files, extracted text, and generated wiki pages. This is the core function of the service — we store your content so you and your connected AI tools can access it.

### Processed content
When you upload PDFs or office documents, we process them server-side to extract text. The extracted text is stored alongside the original file.

### Browser extension data
If you use the LLM Wiki Chrome extension, it captures the content of web pages and PDFs you explicitly choose to clip, including copies of the page's images. The extension only acts on a page when you invoke it — it does not passively monitor your browsing. Clipped content is sent to our API and stored in your knowledge base.

Clipped pages may contain personal data about third parties (for example, names or contact details that appear in an article). You are responsible for having a lawful basis to store such content, and we process it solely on your behalf to provide the service.

### Usage data
We collect basic usage analytics: page views, feature usage, and error logs. We do not use third-party tracking scripts or advertising pixels.

### Legal requests and copyright notices
If you send us a legal request, copyright complaint, DMCA notice, or counter-notice, we collect the information you provide so we can review, respond to, and preserve records of the request.

## How your content is stored

| Component | Provider | Location | Purpose |
|-----------|----------|----------|---------|
| Database | Supabase (Postgres) | AWS US regions | Account data, documents, wiki pages, metadata |
| File storage | Amazon S3 | US East | Raw uploaded files (PDFs, images) |
| API hosting | Railway | US regions | API and MCP servers |
| Frontend hosting | Netlify | Global CDN | Web application |

All data is encrypted at rest (AES-256) and in transit (TLS 1.2+). Database access is enforced through row-level security (RLS) — each user can only access their own data.

## Third-party services that process your content

| Service | What it sees | Why |
|---------|-------------|-----|
| Supabase | All stored data | Database and authentication provider |
| Amazon S3 | Raw uploaded files | File storage |
| Railway | All data in transit through API | API and MCP server hosting |
| Netlify | Frontend assets, request logs | Web application hosting |
| Your connected AI assistant provider | Document content during AI conversations | Wiki generation and knowledge base tools via MCP |

We do not send your content to any service for the purpose of AI model training.

## How AI tools access your content

LLM Wiki connects to AI assistants via the Model Context Protocol (MCP). When you connect an AI assistant:

- The assistant can search, read, and write to your knowledge bases using MCP tools
- Your content is sent to the assistant provider's infrastructure as part of your conversations
- This access is governed by your relationship with that provider and their privacy policy
- You can disconnect at any time by removing the MCP connector in your assistant settings

We do not control how assistant providers process content sent through their conversations. Refer to your provider's privacy policy for details on their data handling.

## What we do NOT do

- We do not sell your data
- We do not serve advertisements
- We do not use your content to train AI models
- We do not share your content with other users
- We do not access your content for any purpose other than providing the service, unless required by law

## Legal bases for processing (GDPR)

Where the GDPR applies, we process your data on these legal bases: performance of our contract with you (to provide the service), your consent (where you grant it — for example, connecting an AI tool), and our legitimate interests (security, abuse prevention, and basic analytics).

## International data transfers

Your data is stored in US regions. Where data is transferred from the EEA, UK, or Switzerland to the United States, we and our subprocessors rely on the European Commission's Standard Contractual Clauses and the EU-US Data Privacy Framework (and its UK and Swiss extensions) as transfer mechanisms.

## Security incidents

If a security incident affects your personal data, we will notify you and, where required, the relevant supervisory authority without undue delay.

## Data retention and deletion

Your content is stored as long as you maintain an account. You can delete individual documents, knowledge bases, or your entire account at any time.

When you delete content:
- Documents and wiki pages are removed from the database
- Uploaded files are removed from S3
- Search index entries are removed
- Deletion is permanent — we do not retain backups of deleted content beyond our standard database backup window (7 days)

When you delete your account:
- All knowledge bases, documents, wiki pages, and uploaded files are permanently deleted
- Your authentication credentials are removed from Supabase
- This process is irreversible

To request account deletion, email lucas@llmwiki.app.

## Your rights

You can at any time:
- Export your data (download your documents and wiki pages)
- Delete specific content or your entire account
- Disconnect AI tool access by removing MCP connectors
- Request information about what data we hold (email lucas@llmwiki.app)

If you are in the EU, UK, or Switzerland, you have additional rights under GDPR including the right to data portability, rectification, and erasure. Contact lucas@llmwiki.app to exercise these rights.

If you are a California resident, the CCPA/CPRA gives you the right to know, delete, and correct the personal information we hold. We do not sell or share your personal information as those terms are defined under the CCPA. Contact lucas@llmwiki.app to exercise these rights.

## Self-hosting

LLM Wiki is open source (Apache 2.0). If you require full data sovereignty, you can self-host the entire stack on your own infrastructure. When self-hosted, no data passes through our systems. See the GitHub repository for deployment instructions.

## Children

LLM Wiki is not intended for use by anyone under the age of 13. We do not knowingly collect personal information from children under 13.

## Changes to this policy

We may update this policy from time to time. We will notify you of material changes by email or by posting a notice in the application. Continued use of the service after changes constitutes acceptance of the updated policy.

## Contact

For privacy questions or data requests: lucas@llmwiki.app
`
