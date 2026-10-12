"use strict";

// Registration remains inert until a deployed Google Apps Script /exec URL is
// supplied. Never place API keys, GitHub tokens, or spreadsheet credentials here.
window.MATRYOSHKA_VISITOR_REGISTRATION = Object.freeze({
  enabled: true,
  endpoint: "https://script.google.com/macros/s/AKfycbw6PS4pE_-2cKV5JX-3epuZqcFZ7txvLujchpg7xVLjsRTDcQYX_Re2A6zDV2LMwlO0Rg/exec",
  registrationVersion: "2026-10-12-events-v2",
  sessionStorageKey: "matryoshka_visitor_session_v2",
  visitorStorageKey: "matryoshka_visitor_id_v1",
});
