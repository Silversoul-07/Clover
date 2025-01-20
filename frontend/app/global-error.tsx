'use client'

import { useEffect } from 'react'
import { AlertTriangle } from 'lucide-react'

export default function Error({
    error,
    reset,
}: {
    error: Error & { digest?: string }
    reset: () => void
}) {
    useEffect(() => {
        // Log the error to an error reporting service
        console.log(error)
    }, [error])

    return (
        <div className="h-fit flex items-center justify-center text-black p-4 dark:text-white">
            <div className="max-w-md w-full space-y-8 text-center">
                <div className="animate-bounce">
                    <AlertTriangle className="mx-auto h-16 w-16 text-yellow-400" />
                </div>
                <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl">
                    Oops! Something went wrong.
                </h1>
                <p className="mt-2 text-lg text-gray-300">
                    We&apos;ve encountered an unexpected error.
                </p>
                <div className="mt-4 bg-gray-800 rounded-lg p-4 text-left">
                    <h2 className="text-xl font-semibold mb-2">Error Details:</h2>
                    <p className="text-sm text-gray-400 break-all">{error.message}</p>
                    {error.digest && (
                        <p className="mt-2 text-xs text-gray-500">
                            Error ID: {error.digest}
                        </p>
                    )}
                </div>
                <div className="mt-6 flex justify-center space-x-4">
                    <button
                        onClick={() => reset()}
                        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 transition-colors duration-200"
                    >
                        Try again
                    </button>
                    <button
                        onClick={() => {
                            // Implement error reporting logic here
                            alert('Error reported!')
                        }}
                        className="px-4 py-2 bg-gray-700 text-white rounded-md hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 focus:ring-offset-gray-900 transition-colors duration-200"
                    >
                        Report issue
                    </button>
                </div>
            </div>
        </div>
    )
}
