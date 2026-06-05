export type Mode = "cloud" | "local";

const STORAGE_KEY = "llmwiki_mode";
const LOCAL_URL_KEY = "llmwiki_local_url";
const SELECTED_KB_KEY = "llmwiki_selected_knowledge_base_id";
const SELECTED_FOLDER_KEY = "llmwiki_selected_folder_path";

const DEFAULT_CLOUD_URL = import.meta.env.VITE_API_BASE_URL ?? "https://api.llmwiki.app";
const DEFAULT_LOCAL_URL = "http://localhost:8000";
const LOCAL_HEALTH_TIMEOUT_MS = 2500;

export async function getMode(): Promise<Mode> {
  const result = await chrome.storage.local.get(STORAGE_KEY);
  return result[STORAGE_KEY] === "local" ? "local" : "cloud";
}

export async function setMode(mode: Mode): Promise<void> {
  await chrome.storage.local.set({ [STORAGE_KEY]: mode });
}

export async function getApiUrl(): Promise<string> {
  const mode = await getMode();
  if (mode === "local") {
    const result = await chrome.storage.local.get(LOCAL_URL_KEY);
    return result[LOCAL_URL_KEY] || DEFAULT_LOCAL_URL;
  }
  return DEFAULT_CLOUD_URL;
}

export async function setLocalUrl(url: string): Promise<void> {
  await chrome.storage.local.set({ [LOCAL_URL_KEY]: normalizeApiUrl(url) });
}

export async function getLocalUrl(): Promise<string> {
  const result = await chrome.storage.local.get(LOCAL_URL_KEY);
  return result[LOCAL_URL_KEY] || DEFAULT_LOCAL_URL;
}

export function normalizeApiUrl(url: string): string {
  const value = (url || DEFAULT_LOCAL_URL).trim();
  try {
    const parsed = new URL(value);
    parsed.hash = "";
    parsed.search = "";
    return parsed.toString().replace(/\/+$/, "");
  } catch {
    return value.replace(/\/+$/, "");
  }
}

export async function checkLocalHealth(url: string): Promise<boolean> {
  const apiUrl = normalizeApiUrl(url);
  let timeout: ReturnType<typeof setTimeout> | null = null;
  const controller = new AbortController();
  try {
    timeout = setTimeout(() => controller.abort(), LOCAL_HEALTH_TIMEOUT_MS);
    const response = await fetch(`${apiUrl}/health`, {
      method: "GET",
      signal: controller.signal,
      cache: "no-store",
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

export async function getSelectedKnowledgeBaseId(): Promise<string | null> {
  const result = await chrome.storage.local.get(SELECTED_KB_KEY);
  return typeof result[SELECTED_KB_KEY] === "string"
    ? result[SELECTED_KB_KEY]
    : null;
}

export async function setSelectedKnowledgeBaseId(id: string): Promise<void> {
  await chrome.storage.local.set({ [SELECTED_KB_KEY]: id });
}

export async function getSelectedFolderPath(): Promise<string> {
  const result = await chrome.storage.local.get(SELECTED_FOLDER_KEY);
  const value = result[SELECTED_FOLDER_KEY];
  return typeof value === "string" && value.trim() ? value : "/webclipper/";
}

export async function setSelectedFolderPath(path: string): Promise<void> {
  await chrome.storage.local.set({ [SELECTED_FOLDER_KEY]: normalizeFolderPath(path) });
}

export function normalizeFolderPath(path: string): string {
  let value = (path || "/webclipper/").trim();
  if (!value.startsWith("/")) value = `/${value}`;
  if (!value.endsWith("/")) value = `${value}/`;
  value = value.replace(/\/+/g, "/");
  if (value.includes("..") || value.includes("\\")) return "/webclipper/";
  return value;
}

const DISABLED_DOMAINS_KEY = "llmwiki_disabled_domains";

// Domains where the extension never injects, regardless of user preference.
// The wiki app itself ships its own highlight UI; the content script's pill
// and popover would collide with it.
const BUILT_IN_DISABLED_SUFFIXES = ["llmwiki.app"];

function matchesSuffix(host: string, suffix: string): boolean {
  return host === suffix || host.endsWith(`.${suffix}`);
}

export function isBuiltInDisabledHost(host: string): boolean {
  const h = host.toLowerCase();
  return BUILT_IN_DISABLED_SUFFIXES.some((suffix) => matchesSuffix(h, suffix));
}

export async function getDisabledDomains(): Promise<string[]> {
  const result = await chrome.storage.local.get(DISABLED_DOMAINS_KEY);
  const value = result[DISABLED_DOMAINS_KEY];
  return Array.isArray(value) ? value.filter((v) => typeof v === "string") : [];
}

export async function isDomainDisabled(host: string): Promise<boolean> {
  if (isBuiltInDisabledHost(host)) return true;
  const list = await getDisabledDomains();
  const h = host.toLowerCase();
  return list.includes(h);
}

export async function setDomainDisabled(host: string, disabled: boolean): Promise<void> {
  const h = host.toLowerCase();
  const list = await getDisabledDomains();
  const has = list.includes(h);
  if (disabled && !has) {
    await chrome.storage.local.set({ [DISABLED_DOMAINS_KEY]: [...list, h] });
  } else if (!disabled && has) {
    await chrome.storage.local.set({
      [DISABLED_DOMAINS_KEY]: list.filter((d) => d !== h),
    });
  }
}
