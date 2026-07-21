FROM node:22-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# NEXT_PUBLIC_* are baked into the client bundle at build time. Railway
# passes service variables to declared ARGs automatically; for a plain
# `docker build` (self-hosting) pass them with --build-arg.
ARG NEXT_PUBLIC_MODE
ARG NEXT_PUBLIC_API_URL
ARG NEXT_PUBLIC_MCP_URL
ARG NEXT_PUBLIC_SUPABASE_URL
ARG NEXT_PUBLIC_SUPABASE_ANON_KEY
RUN npm run build

FROM node:22-alpine
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
