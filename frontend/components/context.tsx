'use client';
import React, { createContext, useState } from 'react';

// json.stringify({ data: 'hello' }) => '{"data":"hello"}'

export const DataContext = createContext<{
    data: string | null;
    setData: (data: string) => void;
}>({
    data: null,
    setData: () => {},
});

interface Props {
    children: React.ReactNode;
}

export const DataProvider: React.FC<Props> = ({ children }) => {
    const [data, setData] = useState<string|null>(null);

    return (
        <DataContext.Provider value={{ data, setData }}>
            {children}
        </DataContext.Provider>
    );
};
