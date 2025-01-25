import '@/styles/globals.css';
import React from 'react';
import { ThemeProvider } from "@/components/themes"
import { Toaster } from "@/components/ui/sonner"
import { Suspense } from 'react';
import Loading from '@/app/loading';

interface RootLayoutProps {
  children: React.ReactNode;
}

// improve layout add dynamic header and react suspense
export const metadata = {
  title: {
    template: '%s • Clover',
  },
  description: 'Store and view media from the internet',
};

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head />
      <body className='h-screen'>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <Suspense fallback={<Loading />} >
            {children}
          </Suspense>
          <Toaster position="top-right" richColors expand={true} toastOptions={{ className: 'mt-[40px]' }} />
        </ThemeProvider>
      </body>
    </html>
  )
}
