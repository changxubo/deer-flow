"use client";

import { Github } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { authClient } from "@/server/better-auth/client";

export default function LoginPage() {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleGitHubSignIn = async () => {
    setError(null);
    setLoading(true);
    try {
      await authClient.signIn.social({
        provider: "github",
        callbackURL: "/workspace/chats/new",
      });
    } catch {
      setError("Failed to sign in with GitHub. Please try again.");
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="w-full max-w-sm space-y-6 px-4">
        <div className="space-y-2 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">
            Welcome back
          </h1>
          <p className="text-muted-foreground text-sm">
            Sign in to continue to your workspace
          </p>
        </div>
        {error && (
          <div className="bg-destructive/10 text-destructive rounded-md p-3 text-center text-sm">
            {error}
          </div>
        )}
        <Button
          className="w-full"
          size="lg"
          onClick={handleGitHubSignIn}
          disabled={loading}
        >
          <Github className="size-5" />
          {loading ? "Signing in..." : "Sign in with GitHub"}
        </Button>
      </div>
    </div>
  );
}
