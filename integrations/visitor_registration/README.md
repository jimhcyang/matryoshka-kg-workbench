# Visitor Registration Endpoint

This spreadsheet-bound Google Apps Script records one consented registration
event before a browser opens the Matryoshka demo. It is a visitor log, not an
authentication or authorization system.

## Deploy

1. Create a private Google Sheet named `Matryoshka Visitor Log`.
2. In the Sheet, open **Extensions > Apps Script**.
3. Replace `Code.gs` with this directory's `Code.gs` and save.
4. In **Project Settings**, enable the manifest file and replace it with
   `appsscript.json` if desired. The code also works with the default manifest.
5. Select `setupVisitorLog`, click **Run**, and approve the requested access.
   This creates and formats the `Visitors` tab and records the spreadsheet ID
   in Script Properties.
6. Select **Deploy > New deployment > Web app**. Set **Execute as** to `Me` and
   **Who has access** to `Anyone`, then deploy.
7. Copy the URL ending in `/exec`. Do not use the `/dev` test URL.
8. Put that URL in `web/visitor-config.js`, set `enabled` to `true`, rebuild the
   static site, and deploy it.

If a Workspace policy does not offer anonymous `Anyone` deployment, this
static Pages design cannot submit to Apps Script for external visitors. Use an
approved serverless endpoint instead rather than weakening the Sheet's sharing
settings.

## Data And Safeguards

The private Sheet receives server time, normalized email, page URL, referrer,
timezone, language, an anonymous browser-generated visitor ID, and registration
version. The endpoint rejects invalid emails and unexpected page URLs, escapes
spreadsheet-formula prefixes, ignores a honeypot field, suppresses immediate
duplicates, and serializes writes with a script lock.

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
