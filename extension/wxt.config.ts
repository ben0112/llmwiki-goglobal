import { defineConfig } from "wxt";

const apiOrigin = new URL(process.env.VITE_API_BASE_URL ?? "https://api.llmwiki.app").origin;

export default defineConfig({
  srcDir: "src",
  modules: ["@wxt-dev/module-react"],
  // Dev runner config:
  //   - Persistent profile so the Supabase sign-in survives reloads
  //   - Opens a known testbed URL so we can verify the content script bootstraps
  //     against a real site (CSP, CORS, real DOM)
  //   - Uses your actual Chrome binary, in a separate profile dir, so this
  //     doesn't interfere with your normal browsing session
  runner: {
    binaries: {
      chrome: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    },
    chromiumProfile: "/tmp/llmwiki-ext-profile",
    keepProfileChanges: true,
    startUrls: ["https://example.com/"],
  },
  manifest: {
    name: "LLM Wiki",
    description: "将任意网页或 PDF 保存到您的 LLM Wiki 知识库",
    version: "0.1.0",
    permissions: ["activeTab", "storage", "scripting"],
    // The page is reached via activeTab on the toolbar click; host_permissions
    // is only the API origin so the extension can call its own backend.
    host_permissions: [`${apiOrigin}/*`, "http://localhost/*"],
    icons: {
      16: "icon/16.png",
      32: "icon/32.png",
      48: "icon/48.png",
      96: "icon/96.png",
      128: "icon/128.png",
    },
    action: {
      default_icon: {
        16: "icon/16.png",
        32: "icon/32.png",
      },
    },
  },
});
