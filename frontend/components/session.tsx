'use client';
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Cookies from 'js-cookie';

const SessionWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const router = useRouter();
      useEffect(() => {
      const handleCookieChange = () => {
        const token = Cookies.get('access_token');
        if (!token) {
          router.push('/auth?unauthorized=true');
        }
      };
  
      const cookieChangeInterval = setInterval(handleCookieChange, 5000);
  
      return () => clearInterval(cookieChangeInterval);
    }, [router]);

    useEffect(() => {
      window.scrollTo(0, 0);
    });
  
    return <>{children}</>;
  };

export default SessionWrapper;