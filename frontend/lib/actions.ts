'use server'
import { cookies } from 'next/headers'

export async function getCookie(name: string) {
  const cookiesStore = await cookies();
  return cookiesStore.get(name)?.value
}