"use client";

import {
  BellIcon,
  BrainIcon,
  SparklesIcon,
  WrenchIcon,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import {

  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

export function WorkspaceNavSettingsOptions() {
  const { open: isSidebarOpen } = useSidebar();
  const { t } = useI18n();
  const pathname = usePathname();

  const SETTINGS_OPTIONS = [
    {
      label: t.settings.sections.memory,
      icon: BrainIcon,
      href: "/workspace/memory",
    },
    {
      label: t.settings.sections.skills,
      icon: SparklesIcon,
      href: "/workspace/skills",
    },
    {
      label: t.settings.sections.tools,
      icon: WrenchIcon,
      href: "/workspace/tools",
    },
    {
      label: t.settings.sections.notification,
      icon: BellIcon,
      href: "/workspace/notifications",
    },
  ];

  return (
    <>

      <SidebarMenu>
        <SidebarMenuItem>
          {SETTINGS_OPTIONS.map(({ label, icon: Icon, href }) => {
            const active = pathname === href;
            return (
              <Link className="text-muted-foreground"  href={href} key={label}>
                <SidebarMenuButton isActive={pathname === href} asChild >
                  {isSidebarOpen ? (
                    <div className="flex w-full items-center gap-2 text-left text-sm">
                      <Icon className={cn( "size-4", active ? "" : "text-muted-foreground" )} />
                      <span 
                      className={cn(active ? "" : "text-muted-foreground")}
                      >
                        {label}
                      </span>
                    </div>
                  ) : (
                    <div className="flex size-full items-center justify-center">
                      <Icon
                        className={cn(
                          "size-4",
                          active ? "" : "text-muted-foreground"
                        )}
                      />
                    </div>
                  )}
                </SidebarMenuButton>
              </Link>
            );
          })}
        </SidebarMenuItem>
      </SidebarMenu>
   
    </>
  );
}
