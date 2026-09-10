# Vendored: whoscored-event-data

`src/pitch_edge/vendor/whoscored/` is vendored from
[Ali-Hasan-Khan/Scrape-Whoscored-Event-Data](https://github.com/Ali-Hasan-Khan/Scrape-Whoscored-Event-Data)
(`whoscored-event-data` v2.0.0), MIT licensed — see `LICENSE_whoscored` in this directory.

It is not published on PyPI, so it is vendored directly rather than pulled as a dependency.

## Local changes vs upstream

`whoscored/discovery.py` — fixture/league discovery requires a real browser (Cloudflare-protected
pages) and upstream's `list_fixtures`/`list_leagues` never dismiss WhoScored's cookie-consent
overlay (a Sourcepoint CMP, "We value your privacy" / "Accept all"). That overlay intercepts the
first click on the season `<select>` (`ElementClickInterceptedException`), which is the exact,
previously-known-broken behavior this project's `PITCH_EDGE_CLAUDE.md`/history flagged. Local fix:

* `_dismiss_cookie_banner()` — dismissed once per browser session, immediately after the first
  navigation to whoscored.com, before any further interaction. The consent choice is
  cookie/session-scoped by the CMP, so it does not reappear on later navigations within the same
  driver session — no changes needed to `client.py`/`transports.py`.
* Season/stage `<option>` elements are now re-located by visible text immediately before each
  click, instead of clicking a reference captured earlier from a list built before the page
  finished re-rendering (this was producing an intermittent `StaleElementReferenceException` on
  top of the interception bug — confirmed by reproducing the flow live with headless Firefox).

Nothing else in the vendored package was modified. Do not run `pip`/`uv` "upgrades" against this
directory — it isn't a normal dependency, it's a source copy with a targeted patch.
