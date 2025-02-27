import React from 'react';
import { Suspense } from 'react';
import Loading from '../loading';
export default function Layout({
    children,
}: {
  children: React.ReactNode
}) {
  return (
        <main>
                    <Suspense fallback={<Loading />} >
          
            {children}
            </Suspense>
        </main>
  );
}