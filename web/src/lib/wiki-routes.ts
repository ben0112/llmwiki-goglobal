export function decodeWikiRouteSlug(routeSlug: string): string {
  try {
    return decodeURIComponent(routeSlug)
  } catch {
    return routeSlug
  }
}
