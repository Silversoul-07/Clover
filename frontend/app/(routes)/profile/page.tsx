import { Metadata } from "next";
import Profile from "@/components/profile";
import { cookies } from 'next/headers'
import { fetchUser } from "@/lib/api";

export const metadata: Metadata = {
    title: 'My Profile',
    description: "Prodile page",
}

const ProfilePage: React.FC = async () => {
  const cookieStore = await cookies()
  const token = cookieStore.get('access_token')?.value as string;

  const username = 'me';
  const { data, error } = await fetchUser(username);
  if (error){
    throw new Error(error);
  }
  return (
    <Profile 
      userData={data} 
      token={token}
    />
  );
};

export default ProfilePage;