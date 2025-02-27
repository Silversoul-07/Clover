import React from 'react';
import Header from '@/components/header';
import SessionWrapper from '@/components/session';
import { DataProvider } from '@/components/context';
import { Suspense } from 'react';
import Loading from '@/app/loading';

export default function Layout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col h-screen">
      <Header />
      <main style={{ height: `calc(100vh - 62px)` }} className="pt-[62px]">
        <SessionWrapper>
          <DataProvider>
            <Suspense fallback={<Loading />} >
              {children}
            </Suspense >
          </DataProvider>
        </SessionWrapper>
      </main>
    </div>
  );
}