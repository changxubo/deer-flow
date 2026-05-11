import { toNextJsHandler } from "better-auth/next-js";

import { auth } from "@/server/better-auth";
import { ensureAuthSchema } from "@/server/better-auth/config";

const { GET: authGET, POST: authPOST } = toNextJsHandler(auth.handler);

export async function GET(request: Request) {
	await ensureAuthSchema();
	return authGET(request);
}

export async function POST(request: Request) {
	await ensureAuthSchema();
	return authPOST(request);
}
