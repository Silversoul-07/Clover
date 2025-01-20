import React from 'react';

export default function Loading() {
    return (
        <div className="fixed inset-0 backdrop-blur-lg flex items-center justify-center">
            <div className="relative h-8 w-8 rounded-full overflow-hidden
                shadow-[-5px_-5px_10px_rgba(255,255,255,0.1),10px_10px_10px_rgba(0,0,0,0.4)] 
                before:content-[''] before:absolute before:inset-2 before:bg-[#1e0c33] before:rounded-full 
                before:shadow-[inset_-2px_-2px_4px_rgba(255,255,255,0.1),inset_3px_3px_4px_rgba(0,0,0,0.4)]">
                <span className="absolute inset-0 rounded-full bg-gradient-to-b from-[#14ffe9] via-[#ffeb3b] to-[#ff00c0] -z-10 [animation:spin_1s_linear_infinite]"></span>
            </div>
        </div>
    );
}