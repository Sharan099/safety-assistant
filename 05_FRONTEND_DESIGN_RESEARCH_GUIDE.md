# Frontend Design Research Guide

## Correct design order

```text
problem
→ users
→ tasks
→ information architecture
→ user flows
→ wireframes
→ visual direction
→ design tokens
→ components
→ high-fidelity prototype
→ usability check
→ implementation
```

Do not start with colors.

## What to study in reference products

Do not copy screenshots. Analyze:
- navigation;
- density;
- primary/secondary actions;
- empty/error states;
- tables;
- filters/search;
- detail panels;
- keyboard behavior;
- responsive behavior;
- accessibility.

Study product categories such as developer tools, enterprise search, document intelligence, knowledge management, compliance software and engineering dashboards.

## Free / open resources

### Figma
Use for sitemap, wireframes, tokens, components and clickable prototypes.

Useful official resources:
- https://www.figma.com/community
- https://help.figma.com/hc/en-us/articles/360038510693-Guide-to-the-Figma-Community
- https://www.figma.com/templates/wireframe-kits/
- https://www.figma.com/templates/dashboard-designs/

### shadcn/ui
Use for implementation-ready project-owned UI source.

- https://ui.shadcn.com/docs
- https://ui.shadcn.com/docs/changelog/2026-06-chat-components

### Radix / accessible primitives
- https://www.radix-ui.com/primitives
- https://www.radix-ui.com/primitives/docs/overview/accessibility

### Palette exploration
Useful tools include Figma variables/styles, Coolors and Realtime Colors. Always validate contrast and status semantics.

## Reference-board exercise

Create a Figma page named:

```text
00_REFERENCE_BOARD
```

Collect 10–15 relevant patterns. For each write:

```text
KEEP: why this interaction fits engineers
AVOID: why another part does not fit this product
```

Example:

```text
KEEP: fixed evidence panel keeps citations visible.
AVOID: oversized AI prompt hero wastes technical workspace.
```

## Wireframes to create

1. Login
2. Home
3. New chat
4. Chat with evidence open
5. Document library
6. Upload document
7. Ingestion processing/failure
8. Document detail
9. Optional admin corpus
10. Optional audit view

## UX questions before code

### Chat
- Can the engineer tell which sources were searched?
- Can they distinguish authoritative/private evidence?
- Is evidence one click away?
- Can they resume work?

### Upload
- Is privacy scope obvious?
- Is processing state honest?
- Can the failure be understood?

### Documents
- Can a version be found quickly?
- Are stale/superseded versions distinguishable?

### Navigation
Can a first-time user identify Ask, Documents and History quickly?

## Design tokens first

Define variables for:
- color;
- typography;
- spacing;
- radius;
- border;
- shadow;
- z-index;
- motion.

Then build pages from tokens instead of one-off values.

## Suggested Figma naming

```text
Foundations/
  Color
  Typography
  Spacing

Components/
  Button
  Input
  Select
  Badge
  Table
  Dialog

Product/
  EvidenceCard
  DocumentStatus
  ConversationItem
  SourceScope

Screens/
  Login
  Home
  Chat
  Documents
  Upload
```

## Handoff to Claude

Provide:
- design tokens;
- component states;
- layout measurements;
- responsive rules;
- interaction rules;
- screenshots/prototype;
- acceptance criteria.

Do visual verification in-browser at several viewport sizes instead of trusting code alone.
