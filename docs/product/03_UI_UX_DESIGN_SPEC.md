# UI/UX Design Specification — Safety Assistant v2

## Design intent

Desired qualities:

```text
calm
precise
dense-but-readable
traceable
professional
predictable
safety-conscious
```

Avoid:

```text
crypto-dashboard styling
excessive gradients
glowing AI effects
oversized marketing cards
chat bubbles that hide evidence
random color usage
animation without information value
```

## UX principle

The core object is:

```text
engineering question + answer + evidence + source/version + investigation history
```

Evidence must therefore be first-class.

## Recommended desktop layout

At 1440px+:

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ Top bar: Workspace | Source Scope | Search | System Status | User         │
├───────────────┬──────────────────────────────────┬─────────────────────────┤
│ Left nav      │ Main work area                   │ Evidence panel          │
│ ~240px        │ flexible                         │ ~400px                  │
│               │                                  │                         │
│ New chat      │ Conversation title               │ Evidence 1              │
│ Conversations │ messages                         │ clause/page/version     │
│ Documents     │                                  │ source preview          │
│ Uploads       │                                  │                         │
│ Admin*        │ composer                         │ Evidence 2              │
│ Settings      │                                  │                         │
└───────────────┴──────────────────────────────────┴─────────────────────────┘
```

Evidence panel is collapsible, keyboard accessible and opens when a citation is selected.

## Information architecture

```text
/login

/app
  /home
  /chat
  /chat/[conversationId]
  /documents
  /documents/upload
  /documents/[documentId]
  /ingestion
  /settings

/app/admin
  /corpus
  /ingestion
  /audit
```

## Home/dashboard

Purpose: orient the engineer, not show vanity metrics.

Sections:
1. Resume work — recent conversations.
2. Knowledge readiness — corpus state and uploads processing.
3. Quick actions — new investigation, upload, clause search.
4. Recent documents — scope/version/status.
5. Optional admin alert — failed ingestion/stale sources.

## Chat screen

### Header
- title;
- workspace;
- source-scope selector;
- status;
- actions.

### Assistant message
Show:
- answer;
- answer-mode badge: Grounded / Evidence only / Insufficient evidence;
- inline citations;
- View evidence action.

### Composer
- multiline input;
- attachment/upload shortcut;
- source-scope summary;
- send;
- stop;
- keyboard hint.

Do not expose model/temperature settings to ordinary engineers.

## Evidence panel

Example:

```text
[1] UN R94
Series/version
Clause 5.2.1
Page 18
Validity date
Scope: Verified regulation

Relevant excerpt...

[Open source] [Copy citation]
```

User/workspace evidence must explicitly show its scope.

## Document library

Desktop table columns:

```text
Document | Scope | Version | Status | Updated | Owner/Workspace | Actions
```

Filters:
- scope;
- status;
- type;
- workspace;
- search.

Status uses text/icon plus color, never color alone.

## Upload experience

### Step 1 — choose file
Display allowed format, max size and privacy/scope.

### Step 2 — metadata
Possible fields:
- display title;
- document type;
- workspace;
- scope;
- version label;
- effective date;
- notes.

### Step 3 — processing
Show real backend stages:

```text
Uploaded
Validating
Parsing
Chunking
Embedding
Indexing
Verifying
Ready
```

Do not fake percentages if backend only knows stage state.

### Step 4 — ready
Actions:
- Ask this document;
- View details;
- Add to conversation scope.

Failure shows plain-language reason, diagnostic reference, replace/retry when safe.

## Login

Minimal. Product mark + one-sentence value + organization sign-in. Dev login is never exposed in production.

## Conversation history

Left navigation:
- New investigation;
- searchable recent conversations;
- archive.

Titles may be generated, but user-edited titles are never overwritten automatically.

## Design system — Engineering Cobalt

```css
--background:        #F7F9FC;
--surface:           #FFFFFF;
--surface-subtle:    #F1F4F8;
--border:            #D8DEE8;

--text-primary:      #142033;
--text-secondary:    #526176;
--text-muted:        #7B8798;

--primary:           #2457D6;
--primary-hover:     #1C46B1;
--primary-soft:      #EAF0FF;

--evidence:          #087D73;
--evidence-soft:     #E5F6F3;

--warning:           #A76500;
--warning-soft:      #FFF3D6;

--danger:            #B42318;
--danger-soft:       #FDECEA;

--success:           #16794D;
--success-soft:      #E8F5EE;
```

Dark mode is optional after light mode is complete.

Typography:
- Inter or Geist for UI;
- mono only for technical identifiers where useful.

Scale:
```text
12 caption
14 body-small
16 body
18 section title
24 page title
```

Spacing uses a 4px base: 4/8/12/16/24/32.

Radius:
```text
controls 8px
panels 10–12px
```

Minimal shadows; use borders and surface hierarchy.

## Component inventory

Foundation:
- Button, Input, Textarea, Select, Checkbox;
- Dialog, Dropdown, Tooltip, Tabs, Sheet;
- Table, Badge, Progress, Skeleton, Toast, Alert.

Product:
- AppShell;
- WorkspaceSwitcher;
- SourceScopeSelector;
- ConversationList/Header;
- MessageList/AssistantMessage;
- CitationMarker;
- EvidencePanel/Card;
- DocumentTable/Status;
- UploadDropzone;
- IngestionTimeline;
- SystemStatus;
- EmptyState/ErrorState.

## Accessibility

Required:
- keyboard operation;
- visible focus;
- semantic controls;
- accessible names;
- contrast;
- screen-reader announcements for processing/streaming;
- correct dialog focus;
- keyboard citation navigation;
- reduced motion.

## Responsive behavior

### Desktop ≥1280
Three panels.

### Tablet 768–1279
Collapsible nav, evidence as drawer/sheet.

### Mobile
Support basic chat, evidence and document status. Complex admin is desktop-first.

## State design

Every major screen defines:

```text
loading
empty
success
recoverable error
permission denied
backend unavailable
```

## Approval gate

Before implementation approve:
- sitemap;
- five primary flows;
- low-fi chat;
- low-fi upload;
- low-fi library;
- palette;
- component inventory;
- responsive behavior.
