"use client";

import { useEffect, useState } from "react";

import {
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarFooter,
  SidebarRail,
  useSidebar,
} from "@/components/ui/sidebar";
import { RecentChatList } from "./recent-chat-list";
import { UserProfile } from "./user-profile";
import { WorkspaceHeader } from "./workspace-header";
import { WorkspaceNavChatList } from "./workspace-nav-chat-list";
import { WorkspaceNavSettingsOptions } from "./workspace-nav-settings-options";

export function WorkspaceSidebar({
  ...props
}: React.ComponentProps<typeof Sidebar>) {
  const { open: isSidebarOpen } = useSidebar();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <>
      <Sidebar variant="sidebar" collapsible="icon" {...props}>
        <SidebarHeader className="py-0">
          <WorkspaceHeader />
          <WorkspaceNavSettingsOptions />
          <WorkspaceNavChatList />
        </SidebarHeader>
        <SidebarContent>
          {mounted && isSidebarOpen && <RecentChatList />}
        </SidebarContent>
        <SidebarFooter>
          <UserProfile />
        </SidebarFooter>
        <SidebarRail />
      </Sidebar>
    </>
  );
}
