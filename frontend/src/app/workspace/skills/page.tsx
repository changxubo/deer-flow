"use client";
import { SkillSettingsPage } from "@/components/workspace/settings/skill-settings-page";

export default function SkillsPage() {
  return (
    <div className="h-full overflow-y-auto px-4 mt-4">
      <SkillSettingsPage onClose={() => {}} />
    </div>
  );
}
