'use client';
import React from 'react';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

interface ClusterProps {
  clusterInfo: {
        title: string;
        user: string;
        username: string;
        stats: {
            followers: number;
            elements: number;
        };
        description?: string;
    }
}

const Cluster: React.FC<ClusterProps> = ({ clusterInfo }) => {
    return (
        <div className="w-full max-w-sm mx-auto p-6 text-center">
            {/* Title */}
            <h1 className="text-4xl font-bold mb-4">{clusterInfo.title}</h1>

            {/* Owner Info */}
            <div className="flex justify-center items-center space-x-1 mb-4 text-sm">
                <Link href={`/p/${clusterInfo.user}`}>
                    <p className="font-semibold">@{clusterInfo.username}</p>
                </Link>
                <span>·</span>
                <p className="">{clusterInfo.stats.elements} Images</p>
                <span>·</span>
                <p className="">{clusterInfo.stats.followers} Followers</p>
            </div>

            {!clusterInfo.description && (
                <div className="mb-4 text-center">
                    <p className="text-sm">
                        {'A cluster of images that I have collected over the years. I hope you enjoy them!'}
                    </p>
                </div>
            )}

            {/* Action Buttons */}
            <div className="mt-4 flex justify-center space-x-4">
                <Button variant="default" size="sm" className="rounded-full">Share</Button>
            </div>
        </div>
    );
};

// future plan add a follow button, backend follow implementation is done

export default Cluster;