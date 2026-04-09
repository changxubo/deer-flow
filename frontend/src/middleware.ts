import { type NextRequest, NextResponse } from "next/server";

// better-auth v1 uses these cookie names
const SESSION_COOKIES = ["better-auth.session_token", "session_token"];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Check for any of the common session cookie names
  const sessionToken = SESSION_COOKIES.find(name => request.cookies.get(name));
  const hasSession = !!sessionToken;

  console.log(`[Middleware] Path: ${pathname}, Session: ${hasSession ? 'Found (' + sessionToken + ')' : 'Not Found'}`);

  // Redirect unauthenticated users from /workspace to /login
  if (pathname.startsWith("/workspace") && !hasSession) {
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("callbackUrl", pathname);
    return NextResponse.redirect(loginUrl);
  }

  // Redirect authenticated users away from /login to workspace
  if (pathname === "/login" && hasSession) {
    return NextResponse.redirect(new URL("/workspace/chats/new", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/workspace/:path*", "/login"],
};
