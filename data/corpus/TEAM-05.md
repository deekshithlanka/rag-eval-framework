---
id: TEAM-05
title: Single sign-on (SSO)
category: Team and permissions
last_updated: 2026-06-20
---

# Single sign-on (SSO)

SAML single sign-on is available on the Scale and Enterprise plans. Ledgerly supports Okta, Microsoft Entra ID, Google Workspace, and any identity provider that supports SAML 2.0.

Set up SSO in Settings > Security > Single sign-on. You will need your identity provider's metadata URL or XML file. Test the connection before you turn on enforcement.

When SSO is enforced, team members must sign in through your identity provider. Password sign-in is turned off for everyone except the account owner, who keeps password access as a backup in case the SSO connection breaks.

Automatic user provisioning with SCIM is available on Enterprise only.
