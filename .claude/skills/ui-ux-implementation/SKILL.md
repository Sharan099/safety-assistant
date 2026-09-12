---
name: ui-ux-implementation
description: Implement or review Safety Assistant frontend screens and components according to the approved evidence-first UI/UX design system and app flows.
---

# UI/UX Implementation

Read `03_UI_UX_DESIGN_SPEC.md` and `04_APP_FLOWS.md` when needed.

Before a screen:
1. identify route and primary job;
2. list loading/empty/success/error/permission states;
3. reuse tokens/components;
4. define keyboard behavior;
5. define responsive behavior.

Rules:
- use the approved Next.js/shadcn stack;
- do not create a second design system;
- keep evidence first-class;
- distinguish authoritative/private/workspace sources;
- no gratuitous animation;
- no one-off colors in page components;
- avoid monolithic page components.

After:
- run frontend checks;
- verify desktop/tablet visually;
- run relevant Playwright flow.
