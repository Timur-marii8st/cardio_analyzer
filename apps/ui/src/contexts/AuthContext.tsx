import React, { createContext, useContext, useState, useEffect, useMemo, useCallback } from 'react';

interface User {
  user_id: string;
  email: string;
  role: string;
  
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  isLoading: boolean;
  error: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(localStorage.getItem('access_token'));
  const [refreshToken, setRefreshToken] = useState<string | null>(localStorage.getItem('refresh_token'));
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const logout = useCallback(() => {
    setUser(null);
    setToken(null);
    setRefreshToken(null);
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
  }, []);

  const refreshAccessToken = useCallback(async () => {
    if (!refreshToken) return;

    try {
      const response = await fetch(`${API_BASE}/api/v1/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken })
      });

      if (response.ok) {
        const data = await response.json();
        setToken(data.access_token);
        localStorage.setItem('access_token', data.access_token);
        
        const userResponse = await fetch(`${API_BASE}/api/v1/me`, {
          headers: { 'Authorization': `Bearer ${data.access_token}` }
        });

        if (userResponse.ok) {
          const userData = await userResponse.json();
          setUser(userData);
        }
      } else {
        logout();
      }
    } catch (err) {
      console.error('Token refresh error:', err);
      logout();
    }
  }, [refreshToken, logout]);

  useEffect(() => {
    const verifyAuth = async () => {
      if (!token) {
        setIsLoading(false);
        return;
      }

      try {
        const response = await fetch(`${API_BASE}/api/v1/me`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.ok) {
          const userData = await response.json();
          setUser(userData);
        } else if (response.status === 401) {
          await refreshAccessToken();
        } else {
          logout();
        }
      } catch (err) {
        console.error('Auth verification error:', err);
        logout();
      } finally {
        setIsLoading(false);
      }
    };

    verifyAuth();
  }, [token, logout, refreshAccessToken]);

  const login = useCallback(async (email: string, password: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/v1/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Login failed' }));
        throw new Error(errorData.detail || 'Login failed');
      }

      const data = await response.json();
      
      setToken(data.access_token);
      setRefreshToken(data.refresh_token);
      localStorage.setItem('access_token', data.access_token);
      localStorage.setItem('refresh_token', data.refresh_token);

      const userResponse = await fetch(`${API_BASE}/api/v1/me`, {
        headers: { 'Authorization': `Bearer ${data.access_token}` }
      });

      if (userResponse.ok) {
        const userData = await userResponse.json();
        setUser(userData);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Login failed';
      setError(errorMessage);
      console.error('Login error:', errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const memoizedValue = useMemo(() => ({
    user,
    token,
    login,
    logout,
    isLoading,
    error,
  }), [user, token, isLoading, error, login, logout]);

  return (
    <AuthContext.Provider value={memoizedValue}>
      {children}
    </AuthContext.Provider>
  );
};