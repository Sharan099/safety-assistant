"use client";
import { FileText, FolderOpen, Home, LogOut, Menu, MessageSquare, Settings, ShieldAlert, Upload } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";

import { ProductMark } from "@/components/common/ProductMark";
import { EvidencePanel } from "@/components/evidence/EvidencePanel";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuGroup, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { useLogout } from "@/features/queries";
import type { Me } from "@/lib/types";
import { cn } from "@/lib/utils";

import { SystemStatus } from "./SystemStatus";

const NAV = [
  { href: "/app/home", label: "Home", Icon: Home },
  { href: "/app/chat", label: "Investigations", Icon: MessageSquare },
  { href: "/app/documents", label: "Documents", Icon: FolderOpen },
  { href: "/app/documents/upload", label: "Upload", Icon: Upload },
  { href: "/app/ingestion", label: "Ingestion", Icon: FileText },
  { href: "/app/settings", label: "Settings", Icon: Settings },
];

function NavLinks({ me, onNavigate }: { me: Me; onNavigate?: () => void }) {
  const pathname = usePathname();
  const isAdmin = me.user.scopes.includes("audit:read") || me.user.scopes.includes("document:ingest");
  const items = isAdmin ? [...NAV, { href: "/app/admin", label: "Admin", Icon: ShieldAlert }] : NAV;
  return (
    <nav aria-label="Primary" className="flex flex-col gap-1 p-2">
      <Button nativeButton={false}
        className="mb-2 justify-start"
        onClick={onNavigate}
        render={<Link href="/app/chat?new=1" data-testid="nav-new" />}
      >
        <MessageSquare className="size-4" aria-hidden /> New investigation
      </Button>
      {items.map(({ href, label, Icon }) => {
        const active = href === "/app/documents" ? pathname === href || /^\/app\/documents\/[^/]+$/.test(pathname) && !pathname.endsWith("/upload") : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            onClick={onNavigate}
            aria-current={active ? "page" : undefined}
            data-testid={`nav-${label.toLowerCase()}`}
            className={cn(
              "flex items-center gap-2 rounded-md px-2.5 py-2 text-sm text-text-secondary hover:bg-secondary hover:text-foreground",
              active && "bg-primary-soft font-medium text-primary hover:bg-primary-soft",
            )}
          >
            <Icon className="size-4" aria-hidden /> {label}
          </Link>
        );
      })}
    </nav>
  );
}

export function AppShell({ me, children }: { me: Me; children: React.ReactNode }) {
  const router = useRouter();
  const logout = useLogout();
  const [navOpen, setNavOpen] = useState(false);
  const workspace = me.workspaces.find((w) => w.id === me.preferences.default_workspace_id) ?? me.workspaces[0];

  return (
    <div className="flex h-screen flex-col">
      <header className="flex h-12 shrink-0 items-center gap-3 border-b bg-card px-3">
        <Button size="icon" variant="ghost" className="lg:hidden" aria-label="Open navigation" onClick={() => setNavOpen(true)}>
          <Menu className="size-5" />
        </Button>
        <Link href="/app/home" className="shrink-0" aria-label="Home">
          <ProductMark compact />
        </Link>
        <div className="hidden min-w-0 items-center gap-2 text-sm text-text-secondary sm:flex" data-testid="workspace-indicator">
          <span className="truncate">{workspace ? `Workspace: ${workspace.name}` : "No workspace"}</span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <SystemStatus />
          <DropdownMenu>
            <DropdownMenuTrigger nativeButton render={<Button variant="outline" size="sm" data-testid="user-menu" />}>
              {me.user.display_name}
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuGroup>
                <DropdownMenuLabel className="font-normal">
                  <div className="text-sm font-medium">{me.user.display_name}</div>
                  <div className="text-xs text-muted-foreground">{me.user.email}</div>
                  <div className="text-xs text-muted-foreground">{me.user.roles.join(", ")}</div>
                </DropdownMenuLabel>
              </DropdownMenuGroup>
              <DropdownMenuSeparator />
              <DropdownMenuItem render={<Link href="/app/settings" />}>Settings</DropdownMenuItem>
              <DropdownMenuItem
                data-testid="logout"
                onClick={() => logout.mutate(undefined, { onSuccess: () => router.replace("/login") })}
              >
                <LogOut className="size-4" aria-hidden /> Sign out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>
      <div className="flex min-h-0 flex-1">
        <aside className="hidden w-60 shrink-0 border-r bg-sidebar lg:block" aria-label="Sidebar">
          <NavLinks me={me} />
        </aside>
        <Sheet open={navOpen} onOpenChange={setNavOpen}>
          <SheetContent side="left" className="w-72 p-0">
            <SheetTitle className="sr-only">Navigation</SheetTitle>
            <div className="border-b p-3">
              <ProductMark />
            </div>
            <NavLinks me={me} onNavigate={() => setNavOpen(false)} />
          </SheetContent>
        </Sheet>
        <main className="min-w-0 flex-1 overflow-y-auto" id="main">
          {children}
        </main>
        <EvidencePanel />
      </div>
    </div>
  );
}
