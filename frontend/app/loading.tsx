import React from 'react';

export default function Loading() {
    return (
        <div className="fixed inset-0 flex flex-col items-center justify-center z-50 backdrop-opacity-10">
            <div className="relative h-8 w-8 rounded-full overflow-hidden
                before:content-[''] before:absolute before:inset-[3px] before:bg-background before:rounded-full">
                <span className="absolute inset-0 rounded-full bg-gradient-to-b from-[#14ffe9] via-[#ffeb3b] to-[#ff00c0] -z-10 [animation:spin_1s_linear_infinite]"></span>
            </div>
            <div className="pl-2 pt-2 text-sm font-bold bg-clip-text animate-pulse">
                Loading...
            </div>
        </div>
    );
}