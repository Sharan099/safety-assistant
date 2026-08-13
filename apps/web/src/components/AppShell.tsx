// UI_UX_DESIGN_BRIEF.md Section 3: Header + Sidebar + Main workspace.
// "Not a ChatGPT clone" — no chat panel dominates this shell.
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/runs", label: "Runs" },
  { href: "/investigations/new", label: "New Investigation" },
  { href: "/knowledge", label: "Knowledge" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="flex min-h-screen flex-col bg-neutral-950 text-neutral-100">
      <header className="flex items-center border-b border-neutral-800 px-6 py-3">
        <span className="text-sm font-semibold tracking-wide text-neutral-200">
          Passive Safety CAE Investigation Agent
        </span>
        <span className="ml-3 rounded border border-neutral-700 px-1.5 py-0.5 text-[10px] uppercase text-neutral-500">
          V1
        </span>
      </header>
      <div className="flex flex-1">
        <nav className="w-52 shrink-0 border-r border-neutral-800 px-3 py-4">
          <ul className="space-y-1">
            {NAV.map((item) => {
              const active = pathname === item.href;
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    className={`block rounded px-3 py-1.5 text-sm ${
                      active ? "bg-neutral-800 text-neutral-100" : "text-neutral-400 hover:bg-neutral-900"
                    }`}
                  >
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
        <main className="flex-1 overflow-x-auto p-6">{children}</main>
      </div>
    </div>
  );
}
