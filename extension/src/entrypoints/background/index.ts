import { getSupabase } from "@/lib/supabase";
import { getApiUrl, clearAccountSelections } from "@/lib/settings";
import { isAllowedApiFetchUrl } from "@/lib/security";
import type { AuthChangeEvent, Session } from "@supabase/auth-js";

type Message =
  | { type: "SIGN_IN_WITH_PASSWORD"; email: string; password: string }
  | { type: "SIGN_OUT" }
  | { type: "GET_SESSION" }
  | { type: "DOWNLOAD_PDF"; url: string }
  | {
      type: "API_FETCH";
      url: string;
      method?: string;
      headers?: Record<string, string>;
      body?: string;
    };

interface ApiFetchResponse {
  ok: boolean;
  status: number;
  data?: unknown;
  error?: string;
}

export default defineBackground(() => {
  const supabase = getSupabase();

  supabase.auth.onAuthStateChange((_event: AuthChangeEvent, _session: Session | null) => {});

  chrome.runtime.onMessage.addListener(
    (message: Message, sender, sendResponse) => {
      // Only our own contexts (popup, injected content scripts) may drive these
      // privileged handlers. Without externally_connectable no web page can
      // reach here, but reject foreign senders as defense in depth.
      if (sender.id !== chrome.runtime.id) return false;
      handleMessage(message)
        .then(sendResponse)
        .catch((err: unknown) => {
          sendResponse({ error: err instanceof Error ? err.message : "后台错误" });
        });
      return true; // will respond asynchronously
    },
  );

  async function handleMessage(msg: Message) {
    switch (msg.type) {
      case "SIGN_IN_WITH_PASSWORD":
        return signInWithPassword(msg.email, msg.password);
      case "SIGN_OUT":
        return signOut();
      case "GET_SESSION":
        return getSession();
      case "DOWNLOAD_PDF":
        return downloadPdf(msg.url);
      case "API_FETCH":
        return apiFetchProxy(msg);
      default:
        return { error: "未知消息类型" };
    }
  }

  // ── API fetch proxy ─────────────────────────────────────
  //
  // Content scripts in MV3 fetch from the page's origin, which means most
  // sites (Substack, console.cloud.google.com, NYT, etc.) block our calls
  // via CORS or strict CSP. The background service worker has the privileged
  // chrome-extension origin and the required host_permission for the API
  // origin, so it can make the request and forward the result. The target is
  // gated to the configured API origin by isAllowedApiFetchUrl.

  async function apiFetchProxy(
    msg: {
      url: string;
      method?: string;
      headers?: Record<string, string>;
      body?: string;
    },
  ): Promise<ApiFetchResponse> {
    try {
      if (!isAllowedApiFetchUrl(msg.url, await getApiUrl())) {
        return { ok: false, status: 403, error: "Blocked extension fetch target" };
      }
      const res = await fetch(msg.url, {
        method: msg.method ?? "GET",
        headers: msg.headers,
        body: msg.body,
      });
      let data: unknown = null;
      const text = await res.text();
      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          data = text;
        }
      }
      return { ok: res.ok, status: res.status, data };
    } catch (err) {
      const message = err instanceof Error ? err.message : "Network error";
      return { ok: false, status: 0, error: message };
    }
  }

  async function signInWithPassword(
    email: string,
    password: string,
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      if (error) {
        return { success: false, error: error.message };
      }
      return { success: true };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "登录失败";
      return { success: false, error: message };
    }
  }

  async function signOut() {
    await supabase.auth.signOut();
    await clearAccountSelections();
    return { success: true };
  }

  async function getSession() {
    const {
      data: { session },
    } = await supabase.auth.getSession();
    return {
      accessToken: session?.access_token ?? null,
      userId: session?.user?.id ?? null,
    };
  }

  // ── PDF Download ────────────────────────────────────────

  async function downloadPdf(
    url: string,
  ): Promise<{ blob: number[]; filename: string } | { error: string }> {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        return { error: `Download failed: ${response.status}` };
      }

      const buffer = await response.arrayBuffer();
      const bytes = Array.from(new Uint8Array(buffer));

      // Derive filename
      let filename = "document.pdf";
      const disposition = response.headers.get("content-disposition");
      if (disposition) {
        const match = disposition.match(
          /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/,
        );
        if (match?.[1]) {
          filename = match[1].replace(/['"]/g, "");
        }
      } else {
        const lastSegment = new URL(url).pathname.split("/").pop();
        if (lastSegment) {
          filename = lastSegment.endsWith(".pdf") ? lastSegment : `${lastSegment}.pdf`;
        }
      }

      return { blob: bytes, filename };
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "PDF download failed";
      return { error: message };
    }
  }

});
