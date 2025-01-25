import LoginForm from "@/components/auth-form"
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Auth",
  description: "Authentication page",
};

type Props = {
  params: Promise<{ name: string }>
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>
}

export default async function AuthPage({ searchParams }: Props) {
  const unauthorized = (await searchParams).unauthorized as string;
  if (unauthorized) {
    return <LoginForm unauthorized />;
  }
  return <LoginForm />;
}

