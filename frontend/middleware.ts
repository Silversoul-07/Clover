import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export async function middleware(request: NextRequest) {
  const token = request.cookies.get('access_token')?.value;
  const username = request.cookies.get('username')?.value;
  const path = request.nextUrl.pathname;

  if (!token && !['/auth'].includes(path)) {
    return NextResponse.redirect(new URL('/auth', request.url));
  }

  if (token && ['/auth'].includes(path)) {
    return NextResponse.redirect(new URL('/', request.url));
  }

  if (path === '/home') {
    return NextResponse.redirect(new URL('/', request.url));
  }

  if (username && path === `/p/${username}`) {
    return NextResponse.redirect(new URL('/profile', request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!_next/static|auth|_next/image|assets|favicon.ico|public|test|sitemap.xml).*)'],
};