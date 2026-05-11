import { DatabaseSync } from "node:sqlite";
import { betterAuth } from "better-auth";
import { getMigrations } from "better-auth/db";

import { env } from "@/env";

const database = new DatabaseSync(
  env.BETTER_AUTH_DATABASE_PATH ?? "./better-auth.db",
);

const authOptions = {
  baseURL: env.BETTER_AUTH_URL ?? "http://localhost:3000",
  secret: env.BETTER_AUTH_SECRET ?? "a-default-secret-for-dev-only",
  database,
  session: {
    cookieCache: {
      enabled: true,
      maxAge: 5 * 60, // 5 minutes
    },
  },
  advanced: {
    useSecureCookies: process.env.NODE_ENV === "production",
    trustProxy: true,
  },
  emailAndPassword: {
    enabled: true,
  },
  socialProviders: {
    github: {
      clientId: env.BETTER_AUTH_GITHUB_CLIENT_ID ?? "",
      clientSecret: env.BETTER_AUTH_GITHUB_CLIENT_SECRET ?? "",
      enabled: !!(
        env.BETTER_AUTH_GITHUB_CLIENT_ID &&
        env.BETTER_AUTH_GITHUB_CLIENT_SECRET
      ),
    },
  },
};

export const auth = betterAuth(authOptions);

let schemaBootstrapPromise: Promise<void> | null = null;

export async function ensureAuthSchema() {
  if (!schemaBootstrapPromise) {
    schemaBootstrapPromise = (async () => {
      const { runMigrations } = await getMigrations(authOptions);
      await runMigrations();
    })();
  }

  return schemaBootstrapPromise;
}

export type Session = typeof auth.$Infer.Session;
