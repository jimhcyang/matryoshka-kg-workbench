# Visitor Registration Endpoint

This spreadsheet-bound Google Apps Script records development-preview sessions,
page views, and a bounded set of meaningful product interactions. It is a
visitor/activity log, not an authentication or authorization system.

## Deploy

1. Create a private Google Sheet named `Matryoshka Visitor Log`.
2. In the Sheet, open **Extensions > Apps Script**.
3. Replace `Code.gs` with this directory's `Code.gs` and save.
4. In **Project Settings**, enable the manifest file and replace it with
   `appsscript.json` if desired. The code also works with the default manifest.
5. Select `setupVisitorLog`, click **Run**, and approve the requested access.
   This creates and formats the `Visitor Events` tab and records the spreadsheet
   ID in Script Properties. An older `Visitors` tab is left untouched.
6. Select **Deploy > New deployment > Web app**. Set **Execute as** to `Me` and
   **Who has access** to `Anyone`, then deploy.
7. Copy the URL ending in `/exec`. Do not use the `/dev` test URL.
8. Put that URL in `web/visitor-config.js`, set `enabled` to `true`, rebuild the
   static site, and deploy it.

For an existing deployment, replace `Code.gs`, run `setupVisitorLog` once, then
use **Deploy > Manage deployments > Edit > New version > Deploy**. Editing the
script without creating a new deployment version does not update the public
`/exec` endpoint.

## Session And Event Behavior

- The email gate appears once per browser session, not once forever. A new
  browser session therefore creates a new `session_start` row followed by the
  initial `page_view` row.
- Reloading during the same session does not ask for email again; it creates a
  `page_view` row.
- Meaningful actions create `interaction` rows, including mode/task selection,
  search use, graph node or edge inspection, hierarchy drill-down, route and
  atom inspection, source-cable opening, and live retrieval execution.
- Hovers, graph dragging, pointer movement, and other high-volume cosmetic
  events are deliberately excluded.
- Raw cable text and raw free-form analyst query text are not written to the
  visitor log. Live-query events contain task, profile, cutoff, character and
  atom counts, retrieval counts, and completion status only.

If a Workspace policy does not offer anonymous `Anyone` deployment, this
static Pages design cannot submit to Apps Script for external visitors. Use an
approved serverless endpoint instead rather than weakening the Sheet's sharing
settings.

## Data And Safeguards

The private Sheet receives server time, event type, action, normalized email,
page URL, application route, bounded target metadata, referrer, timezone,
language, an anonymous persistent visitor ID, a session ID, registration
version, and request ID. Email is retained only in browser session storage; the
persistent local value is an anonymous visitor ID. The endpoint rejects invalid
emails and unexpected page URLs, bounds every field, escapes spreadsheet-formula
prefixes, ignores a honeypot field, deduplicates exact request IDs, and
serializes writes with a script lock.

The public browser bundle contains no Sheet ID, Google credential, API key, or
GitHub token. The `/exec` endpoint is necessarily public, so these controls
reduce accidental noise but do not constitute authentication. Keep the Sheet
private and periodically review or delete visitor records according to the
project's retention policy.

Submission uses a credential-free cross-origin POST. Google does not expose the
Apps Script response to a GitHub Pages origin, so the browser reports
`Registration submitted` after the network accepts the request; the private
Sheet remains the authoritative record of the server-side append. A network
failure keeps the gate open and exposes retry and fail-open controls.
