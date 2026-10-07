---
id: INT-07
title: Ledgerly API
category: Integrations
last_updated: 2026-06-20
---

# Ledgerly API

The Ledgerly REST API is available on the Scale and Enterprise plans. Create API keys in Settings > Developers. Each key can be read-only or read-write.

The API has a rate limit of 120 requests per minute per workspace on Scale and 600 requests per minute on Enterprise. Requests over the limit receive a 429 response with a Retry-After header.

Webhooks can notify your server when invoices are created, sent, viewed, paid, or voided. Webhook deliveries that fail are retried up to 5 times over 24 hours.

API keys do not expire, but you can revoke them at any time. Revoked keys stop working immediately.
