from ddgs import DDGS
from googlesearch import search as gsearch
import asyncio


async def fetch_ddgs(query):
    """Fetch results from DuckDuckGo in a thread."""

    def sync_ddgs():
        results = []
        try:
            with DDGS() as ddgs:
                # Add explicit timeout for Free Tier optimization (e.g. 4 seconds)
                ddgs_results = ddgs.text(
                    query,
                    max_results=5,
                    timeout=4
                )

                for r in ddgs_results or []:
                    results.append({
                        "title": r.get("title", "IEEE Content"),
                        "link": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
        except Exception as e:
            print(f"DDGS Error ({type(e).__name__}): {e}")

        return results

    return await asyncio.to_thread(sync_ddgs)


async def fetch_google(query):
    """Fetch results from Google in a thread."""

    def sync_google():
        results = []
        try:
            for link in gsearch(query, num_results=5):
                results.append({
                    "title": "IEEE Source (via Google)",
                    "link": link,
                    "snippet": "Retrieved from IEEE.org — click link for full details."
                })
        except Exception as e:
            print(f"Google Search Error ({type(e).__name__}): {e}")

        return results

    try:
        # Google search can be slow, wrap it in wait_for to enforce the 4s budget
        return await asyncio.wait_for(asyncio.to_thread(sync_google), timeout=4.0)
    except asyncio.TimeoutError:
        print("Google search timed out (exceeded 4s)")
        return []


async def search_ieee(query):
    """Search IEEE.org using DuckDuckGo and Google concurrently."""

    search_query = f"site:ieee.org {query}"

    ddgs_results, google_results = await asyncio.gather(
        fetch_ddgs(search_query),
        fetch_google(search_query),
        return_exceptions=True
    )

    if isinstance(ddgs_results, Exception):
        print(f"DDGS task failed: {ddgs_results}")
        ddgs_results = []

    if isinstance(google_results, Exception):
        print(f"Google task failed: {google_results}")
        google_results = []

    all_results = list(ddgs_results)

    # Add Google results only if DDG returned too few results
    seen_links = {r["link"] for r in all_results}

    if len(all_results) < 5:
        for result in google_results:
            if result["link"] not in seen_links:
                all_results.append(result)
                seen_links.add(result["link"])

            if len(all_results) >= 5:
                break

    return all_results
