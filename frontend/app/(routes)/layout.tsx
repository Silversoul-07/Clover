import React from 'react';
import Header from '@/components/header';
import SessionWrapper from '@/components/session';
import { DataProvider } from '@/components/context';

export default function Layout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col h-screen">
      <Header />
      <main className="flex-1 overflow-auto pt-[62px]">
        <SessionWrapper>
          <DataProvider>
            {children}
          </DataProvider>
        </SessionWrapper>
      </main>
    </div>
  );
}