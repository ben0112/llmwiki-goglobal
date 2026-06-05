const TRACKING_PARAMS = new Set([
  "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
  "utm_id", "utm_name", "utm_brand", "utm_social",
  "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src",
  "_branch_match_id", "igshid",
]);

export function canonicalize(href: string): string {
  try {
    const u = new URL(href);
    u.hash = "";
    const keep = new URLSearchParams();
    u.searchParams.forEach((v, k) => {
      if (!TRACKING_PARAMS.has(k.toLowerCase())) keep.append(k, v);
    });
    u.search = keep.toString() ? `?${keep.toString()}` : "";
    if (u.pathname.length > 1 && u.pathname.endsWith("/")) {
      u.pathname = u.pathname.replace(/\/+$/, "");
    }
    return u.toString();
  } catch {
    return href;
  }
}
