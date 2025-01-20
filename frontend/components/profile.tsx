'use client'  
import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar';
import { Card, CardContent } from '@/components/ui/card';
import Link from 'next/link';
import { toast } from "sonner"

interface Cluster {
  title: string;
  desc: string;
  href: string;
  username: string;
  avatar: string;
}

interface User {
  name: string;
  username: string;
  bio: string;
  avatar: string;
  clusters: Cluster[];
}


interface Profile {
  name: string;
  username: string;
  bio: string;
  avatar?: string;
  password?: string;
}




type ProfileProps = {
  userData: User;
  token: string;
};

const EditProfile= ({profile}: {profile:Profile}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [formData, setFormData] = useState<Profile>(profile);
  const [newPassword, setNewPassword] = useState('');
  const [avatarFile, setAvatarFile] = useState<File | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      // Add avatar and newPassword handling if needed before sending data
      const updatedData = { ...formData, avatar: avatarFile, password: newPassword };

      await updateUserData(updatedData);
      setIsOpen(false);
    } catch (error) {
      console.error('Error updating profile:', error);
      alert('Failed to update profile. Please try again.');
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setAvatarFile(e.target.files[0]);
    }
  };



  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline">Edit Profile</Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Edit Profile</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="avatar">Avatar</Label>
            <Input
              id="avatar"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="cursor-pointer"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="name">Name</Label>
            <Input
              id="name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="username">Username</Label>
            <Input
              id="username"
              value={formData.username}
              onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="bio">Bio</Label>
            <Textarea
              id="bio"
              value={formData.bio}
              onChange={(e) => setFormData({ ...formData, bio: e.target.value })}
              className="h-24"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="password">New Password</Label>
            <Input
              id="password"
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
            />
          </div>

          <Button type="submit" className="w-full">
            Save Changes
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  );
};


const ProfilePage: React.FC<ProfileProps> = ({ userData, token }) => {
  const [isFollowing, setIsFollowing] = React.useState(null);
  const [followers, setFollowers] = React.useState(null);
  const handleFollow = async () => {
    try {
      await addFollower(userData.username);
      setIsFollowing(true);
      setFollowers(followers + 1);
    } catch (error) {
      console.error(error);
    }
  }
  const handleUnfollow = async () => {
    try {
      await removeFollower(userData.username);
      setIsFollowing(false);
      setFollowers(followers - 1);
    } catch (error) {
      console.error(error);
    }
  }

  const handleCopyToken = () => {
    navigator.clipboard.writeText(token);
    toast.success('Token copied to clipboard');
  }

  return (
    <div className="w-full max-w-[2000px] mx-auto">
      {/* Profile Header */}
      <div className="w-full text-center py-6">
        <Avatar className="w-32 h-32 mx-auto mb-4">
          <AvatarImage src={userData.avatar} alt={`${userData.name}'s avatar`} />
          <AvatarFallback>
            {userData.name.charAt(0)}
          </AvatarFallback>
        </Avatar>

        <h2 className="text-2xl font-bold mb-1">{userData.name}</h2>

        <p className="text-muted-foreground mb-2">
          @{userData.username} • {followers} Followers
        </p>

        <p className="text-sm text-muted-foreground max-w-xs mx-auto mb-4">
          {userData.bio || 'No bio provided'}
        </p>

        {token !== null ? (
                  <div className="space-x-4">
                    <EditProfile profile={userData} />
                    <Button variant="default" size="sm" className="mt-2" onClick={handleCopyToken}>
                      Copy Token
                    </Button>
                    </div>
                ) : (isFollowing ? (
                  <Button variant="default" size="sm" className="mt-2" onClick={handleUnfollow}>
                    Unfollow
                  </Button>
                  ) : (
                    <Button variant="default" size="sm" className="mt-2" onClick={handleFollow}>
                      Follow
                    </Button>
                  )
                )}
      </div>

      {/* Explore Items Grid */}
      <div className="w-full mt-4 px-4 md:px-8">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {userData.clusters.map((item, index) => (
            <Card
              key={index}
              style={{ border: 'none', boxShadow: 'none' }}
              className="w-full cursor-pointer hover:shadow-md transition-shadow"
            >
              <CardContent className="p-0">
                <Link href={item.href}>
                  <img
                    src={item.thumbnail}
                    alt={item.title}
                    className="w-full h-48 object-cover rounded-lg"
                  />
                  <div className="p-3 text-center">
                    <h3 className="font-bold text-sm mb-1 line-clamp-2">{item.title}</h3>
                    <p className="text-xs text-muted-foreground">{item.count || null} images</p>
                  </div>
                </Link>
              </CardContent>
            </Card>
          ))}
        </div>

        {userData.clusters.length === 0 && (
          <div className="text-center text-muted-foreground py-10">
            No explore items found
          </div>
        )}
      </div>
    </div>
  );
};

export default ProfilePage;