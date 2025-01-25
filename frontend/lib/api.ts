import { getCookie } from "@/lib/actions";


const apiClient = async (endpoint: string, options: RequestInit = {}) => {
  const baseUrl =
    typeof window === "undefined" // Running on the server
      ? "http://localhost:8000/api/v1" // e.g., "http://backend:port"
      : "http://localhost:8000/api/v1"; // e.g., "http://localhost:3000"

  const token = await getCookie('access_token');
  if (token) {
    options.headers = {
      ...options.headers,
      Authorization: `Bearer ${token}`,
    };
  }
  
  const headers: HeadersInit = {
    ...(options.headers || {}),
  };
  

  try {
    const response = await fetch(`${baseUrl}${endpoint}`, {
      headers,
      ...options,
    });

    if (!response.ok) {
      const { detail } = await response.json();
      return { data: null, error: detail };
    }

    const data = await response.json();
    return { data, error: null };
  } catch (error) {
    return { data: null, error };
  }
};

export const signup = async (formData: FormData) => {
    const response = await apiClient('/user', {
    method: 'POST',
    body: formData
  });
  return response;
}

export const loginUser = async (username: string, password: string) => {
  const response = await apiClient('/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({ username, password }).toString(),
    credentials: 'include',
  });

  return response;
};

export const fetchUser = async (username: string) => {
  const response = await apiClient(`/user/${username}`, {
    method: 'GET',
  });

  return response;
}

export const createCluster = async (title: string, desc: string) => {
  const response = await apiClient('/cluster', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ title, desc }),
  });

  return response;
}

export const fetchCluster = async (username: string, title:string) => {
  const response = await apiClient(`/user/${username}/cluster/${title}`, {
    method: 'GET',
  });

  return response;
}

export const createElement = async (formData: FormData) => {
  const response = await apiClient('/element', {
    method: 'POST',
    body: formData,
  });

  return response;
}

export const fetchSearch = async (query: string) => {
  const formData = new URLSearchParams();
  formData.append('query', query);

  const response = await apiClient(`/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: formData.toString(),
  });

  return response;
};

export const fetchElement = async (id: string) => { 
  const response = await apiClient(`/element/${id}`, {
    method: 'GET',
  });

  return response;
}

export const fetchSimilar = async (image?: File, id?: string) => {
  const formData = new FormData();
  
  if (image) {
    formData.append('image', image);
  }
  
  if (id) {
    formData.append('id', id);
  }

  const response = await apiClient('/visual-search', {
    method: 'POST',
    body: formData,
  });

  return response;
};

export async function fetchFeed() {
  const response = await apiClient('/feed');
  return response;
  
}








