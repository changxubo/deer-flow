import { betterAuth } from "better-auth";
import { Pool } from "pg";

import { env } from "@/env";

const pool = new Pool({
  connectionString: env.DATABASE_URL,
});

export const auth = betterAuth({
  baseURL: env.BETTER_AUTH_URL ?? "http://localhost:3000",
  secret: env.BETTER_AUTH_SECRET ?? "a-default-secret-for-dev-only",
  database: pool,
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
});

export type Session = typeof auth.$Infer.Session;
