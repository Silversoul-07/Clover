'use client'

import { useEffect, useState } from 'react'
import { GalleryVerticalEnd, Play, Pause } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { toast } from 'sonner'
import { useTheme } from 'next-themes'

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Dialog, DialogContent, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { loginUser, signup, fetchUser } from '@/lib/api'

const LogoutAlert = () => {
    const [open, setOpen] = useState(true);

    const handleClose = () => {
        setOpen(false);
    };

    return (
        <Dialog open={open} onOpenChange={setOpen}>
            <DialogContent>
                <DialogTitle>Logged Out</DialogTitle>
                <DialogDescription>
                    You have been logged out.
                </DialogDescription>
                <DialogFooter>
                    <Button onClick={handleClose}>OK</Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
};

export default function LoginForm({ unauthorized }: { unauthorized?: boolean }) {
    const [isLogin, setIsLogin] = useState(true)
    const [username, setUsername] = useState('')
    const [usernameAvailable, setUsernameAvailable] = useState(true)
    const { setTheme } = useTheme()
    const [play, setPlay] = useState(true)
    const router = useRouter()

    useEffect(() => {
        if (play) {
            setTheme('dark');
        } else {
            setTheme('system');
        }
    }, [play, setTheme])

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault()
        const formData = new FormData(event.currentTarget)

        try {
            if (isLogin) {
                await loginUser(formData.get('username') as string, formData.get('password') as string)
                toast.success('Login successful!')
                setTimeout(() => router.push('/'), 1000)
            } else {
                if (!usernameAvailable) {
                    toast.error('Username is not available')
                    return
                }
                if (formData.get('avatar') && !(formData.get('avatar') instanceof File)) {
                    toast.error('Please upload a valid image file');
                    return;
                }

                const { error } = await signup(formData);
                if (error) {
                    toast.error(error.detail || 'Registration failed. Please try again.');
                    return;
                }
                toast.success('Registration successful!');
                setTimeout(() => setIsLogin(true), 1000);

            }
        } catch (error) {
            toast.error(error.message)
        }
    }

    const checkUsername = async (username: string) => {
        if (username.length > 2) {
            const { data } = await fetchUser(username)
            setUsernameAvailable(!data)
        }
    }

    const handleNotImplemented = (e: React.MouseEvent) => {
        e.preventDefault()
        toast.error('Not implemented yet')
    }

    return (
        <>
            <div className="flex min-h-svh flex-col items-center justify-center gap-6 bg-transparent p-6 md:p-10 relative">
                <Button
                    size='icon'
                    variant="link"
                    onClick={() => setPlay(!play)}
                    className="absolute top-4 right-4 z-10 px-4 py-2 rounded">
                    {play ? <Pause /> : <Play />}
                </Button>
                {play && (
                    <section>
                        <video autoPlay muted loop className="absolute top-0 left-0 w-full h-full object-cover z-0">
                            <source
                                src={`/assets/login-bg.mp4`}
                                type="video/mp4"
                            />
                            Your browser does not support the video tag.
                        </video>
                        <div className="absolute top-0 left-0 w-full h-full bg-black opacity-40 z-1" />
                    </section>
                )}

                <div className="z-10">
                    <div className="w-full max-w-sm">
                        <div className="flex flex-col gap-6">
                            <form onSubmit={handleSubmit}>
                                <div className="flex flex-col gap-6">
                                    <div className="flex flex-col items-center gap-2">
                                        <a href="#" className="flex flex-col items-center gap-2 font-medium">
                                            <div className="flex h-8 w-8 items-center justify-center rounded-md">
                                                <GalleryVerticalEnd className="size-6" />
                                            </div>
                                            <span className="sr-only">Acme Inc.</span>
                                        </a>
                                        <h1 className="text-xl font-bold">Welcome to Clover</h1>
                                        <div className="text-center text-sm">
                                            {isLogin ? "Don't have an account? " : "Already have an account? "}
                                            <span className="underline underline-offset-4" onClick={() => setIsLogin(!isLogin)}>
                                                {isLogin ? "Sign up" : "Log in"}
                                            </span>
                                        </div>
                                    </div>
                                    <div className="flex flex-col gap-4">
                                        {!isLogin && (
                                            <div className="grid gap-2">
                                                <Label htmlFor="name">Name *</Label>
                                                <Input id="name" name="name" required autoComplete="off" />
                                            </div>
                                        )}
                                        <div className="grid gap-2">
                                            <Label htmlFor="username">Username *</Label>
                                            <Input
                                                id="username"
                                                name="username"
                                                required
                                                value={username}
                                                onChange={(e) => {
                                                    setUsername(e.target.value)
                                                    checkUsername(e.target.value)
                                                }}
                                                autoComplete="off"
                                            />
                                            {!isLogin && !usernameAvailable && (
                                                <p className="text-sm text-red-500">Username is not available</p>
                                            )}
                                        </div>
                                        <div className="grid gap-2">
                                            <Label htmlFor="password">Password *</Label>
                                            <Input id="password" name="password" type="password" required autoComplete="off" />
                                        </div>
                                        {!isLogin && (
                                            <div className="grid gap-2">
                                                <Label htmlFor="avatar">Avatar</Label>
                                                <Input id="avatar" name="avatar" type="file" accept="image/*" />
                                            </div>
                                        )}
                                        <Button type="submit" className="w-full">
                                            {isLogin ? "Login" : "Sign Up"}
                                        </Button>
                                    </div>
                                    <div className="relative text-center text-sm after:absolute after:inset-0 after:top-1/2 after:z-0 after:flex after:items-center after:border-t after:border-border">
                                        <span className="relative z-10 px-0 text-muted-foreground">
                                            Or
                                        </span>
                                    </div>
                                    <div className="grid gap-4 sm:grid-cols-2">
                                        <Button className="w-full" onClick={handleNotImplemented}>
                                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                                                <path
                                                    d="M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 1.147-2.933 2.4-6.053 2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.307-2.307C18.747 1.44 16.133 0 12.48 0 5.867 0 .307 5.387.307 12s5.56 12 12.173 12c3.573 0 6.267-1.173 8.373-3.36 2.16-2.16 2.84-5.213 2.84-7.667 0-.76-.053-1.467-.173-2.053H12.48z"
                                                    fill="currentColor"
                                                />
                                            </svg>
                                            Continue with Google
                                        </Button>
                                        <Button className="w-full" onClick={handleNotImplemented}>
                                            <svg viewBox="0 0 24 24" className="mr-2 h-4 w-4">
                                                <path
                                                    fill="currentColor"
                                                    d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"
                                                />
                                            </svg>
                                            Continue with GitHub
                                        </Button>
                                    </div>
                                </div>
                            </form>
                            <div className="text-balance text-center text-xs text-muted-foreground [&_a]:underline [&_a]:underline-offset-4 hover:[&_a]:text-primary">
                                By clicking continue, you agree to our <a href="#">Terms of Service</a>{" "}
                                and <a href="#">Privacy Policy</a>.
                            </div>
                        </div>
                    </div>
                </div>
            </div >
            {unauthorized && <LogoutAlert />
            }
        </>
    )
}

