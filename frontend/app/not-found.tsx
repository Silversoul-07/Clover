import Link from "next/link";
import { buttonVariants } from "@/components/ui/button";
import Image from 'next/image';

export default function NotFound() {
    return (
        <main className="flex justify-center items-center h-full">
            <div className="text-center">
                <Image  alt="404" height={500} width={500} draggable={false} priority src={`/assets/404.png`} />
                <h1 className="mt-4 text-3xl font-bold tracking-tight sm:text-5xl">Page not found</h1>
                <p className="mt-6 text-base leading-7 ">{"Sorry, we couldn't find the page you're looking for."}</p>
                <div className="mt-6 flex items-center justify-center gap-x-6">
                    <Link href={'/'} className={buttonVariants({ variant: 'secondary' })}>Go Home</Link>
                </div>
            </div>
        </main>
    );
}