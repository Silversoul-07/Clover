'use client';
import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { useTheme } from 'next-themes';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';
import { Search, Settings, LogOut, User, Sun, Moon, Menu, Image as ImageIcon, Layers, Bell } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { Sheet, SheetTitle, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { VisualSearchDialog, ImageDialog, ClusterDialog } from '@/components/dialogs';
import Cookies from 'js-cookie';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar';


const Header = () => {
  const [mounted, setMounted] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeDropZone, setActiveDropZone] = useState<'visual-search' | 'image' | 'cluster' | null>(null);
  const { theme, setTheme } = useTheme();
  const router = useRouter();
  const pathname = usePathname();
  const searchRef = useRef<HTMLDivElement>(null);
  const uploadButtonRef = useRef<HTMLButtonElement>(null);
  const searchParams = useSearchParams()

  const navItems = [
    { href: '/', label: 'Home' },
    { href: '/explore', label: 'Explore' },
  ];

  useEffect(() => {
    setMounted(true);
    const query = searchParams.get('q');
    setSearchQuery(query || '');
    
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl + 1 for image upload
      if (e.ctrlKey && e.key === '1') {
        e.preventDefault();
        setActiveDropZone((prev) => (prev === 'image' ? null : 'image'));
      }
      else if (e.ctrlKey && e.key === '2') {
        e.preventDefault();
        setActiveDropZone((prev) => (prev === 'visual-search' ? null : 'visual-search'));
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [searchParams]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    // setActiveDropZone(null);
    if (searchQuery.trim()) {
      router.push(`/search?q=${encodeURIComponent(searchQuery)}`);
    }
  };

  const handleLogout = () => {
    Cookies.remove('access_token');
    router.push('/auth?logout=true');
  };

  return (
    <>
    {/* bottom border */}
      <header className="transition-all duration-300 pt-1 bg-background/80 backdrop-blur-md fixed top-0 left-0 right-0 z-50
      border-b border-muted-foreground/20 shadow-sm
      ">
        <div className=" mx-10 py-2">
          <nav className="flex items-center">
            {/* Left section */}
            <div className="flex items-center gap-4">
              <Link href="/" className="text-2xl font-bold">Clover</Link>

              {/* Desktop Navigation */}
              <div className="hidden md:flex items-center">
                {navItems.map((item) => (
                  <Button className='rounded-full' key={item.href} variant={item.href === pathname ? 'default' : 'ghost'}>
                    <Link href={item.href}>{item.label}</Link>
                  </Button>

                ))}
              </div>
            </div>

            {/* Center section - Search */}
            <div className="flex-1 flex justify-center relative" ref={searchRef} >
              <form onSubmit={handleSearch} className="hidden md:flex w-full max-w-md relative">
                <div className="relative flex w-full">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <input
                      type="search"
                      placeholder="Search..."
                      className="w-full pr-12 pl-10 py-2 rounded-md border border-input bg-background focus:outline-none focus:ring-2 focus:ring-pink-300"                      
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                  <Button
                    type='button'
                    variant="ghost"
                    size="icon"
                    className={`absolute right-2 top-1/2 -translate-y-1/2 ${activeDropZone === 'visual-search' ? 'opacity-50 pointer-events-none' : ''
                      }`}
                    onClick={() => setActiveDropZone('visual-search')}
                  >
                    <ImageIcon className="h-5 w-5" />
                  </Button>
                </div>
              </form>
            </div>

            {/* Right section */}
            <div className="flex items-center gap-1">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    ref={uploadButtonRef}
                    className='rounded-full'
                  >
                    Create
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  <DropdownMenuItem onSelect={() => setActiveDropZone('image')}>
                    <div className="flex items-start">
                      <ImageIcon className="mt-1 mr-2 h-4 w-4" />
                      <div className="flex flex-col">
                        <span>Image</span>
                        <span className="text-muted-foreground ml-auto">Create a Image</span>
                      </div>
                    </div>
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => setActiveDropZone('cluster')}>
                    <div className="flex items-start">
                      <Layers className="mt-1 mr-2 h-4 w-4" />
                      <div className="flex flex-col">
                        <span>Cluster</span>
                        <span className="text-muted-foreground ml-auto">Create a cluster</span>
                      </div>
                    </div>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>

              <Button
                variant="ghost"
                size="icon"
                className="rounded-full"
                onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              >
                {mounted ? (
                  theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />
                ) : (
                  <Moon className="h-5 w-5" />
                )}
              </Button>

              <Popover>
                <PopoverTrigger asChild>
                  <Button variant="ghost" size="icon" className="rounded-full">
                    <Bell className="h-5 w-5" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent className='lg:mr-10' >
                  {/* Notification content goes here */}
                  <div className="p-2">No new notifications</div>
                </PopoverContent>
              </Popover>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Avatar className="ml-2 h-7 w-7">
                    <AvatarImage src="https://github.com/shadcn.png" />
                    <AvatarFallback>CN</AvatarFallback>
                  </Avatar>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  <DropdownMenuLabel>My Account</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem>
                    <Link href="/profile" className="flex items-center">
                      <User className="mr-2 h-4 w-4" />
                      Profile
                    </Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <Link href="/settings" className="flex items-center">
                      <Settings className="mr-2 h-4 w-4" />
                      Settings
                    </Link>
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem>
                    <button onClick={handleLogout} className="flex items-center">
                      <LogOut className="mr-2 h-4 w-4" />
                      Logout
                    </button>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>

              {/* Mobile Menu */}
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="icon" className="md:hidden">
                    <Menu className="h-6 w-6" />
                  </Button>
                </SheetTrigger>
                <SheetContent side="top">
                  <SheetTitle>Menu</SheetTitle>
                  <div className="flex flex-col gap-4 pt-8">
                    {navItems.map((item) => (
                      <Link
                        key={item.href}
                        href={item.href}
                        className={`text-lg hover:text-primary ${pathname === item.href ? 'text-primary font-semibold' : ''}`}
                      >
                        {item.label}
                      </Link>
                    ))}
                    <form onSubmit={handleSearch} className="mt-4">
                      <div className="relative">
                        <input
                          type="search"
                          placeholder="Search..."
                          className="w-full px-3 py-2 rounded-md border border-input bg-background"
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                        />
                        <Search className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      </div>
                    </form>
                  </div>
                </SheetContent>
              </Sheet>
            </div>
          </nav>
        </div>
      </header>

      {activeDropZone === 'image' && (
        <ImageDialog onClose={() => setActiveDropZone(null)} />
      )}
      {activeDropZone === 'cluster' && (
        <ClusterDialog onClose={() => setActiveDropZone(null)} />
      )}
      {activeDropZone === 'visual-search' && (
        <VisualSearchDialog onClose={() => setActiveDropZone(null)} />
      )}
    </>
  );
};

export default Header;