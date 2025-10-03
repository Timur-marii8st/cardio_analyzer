export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
export const WS_BASE = import.meta.env.VITE_WS_BASE || "ws://localhost:8000";

export async function fetchSessions(token: string): Promise<any[]> {
  const res = await fetch(`${API_BASE}/v1/sessions`, {
    method: "GET",
    headers: {
      "Authorization": `Bearer ${token}`
    }
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to fetch sessions: ${text}`);
  }

  return res.json();
}

export async function uploadChannelCsv(
  sessionId: string, 
  files: File[],
  channel: "bpm" | "uterus",
  token: string
): Promise<any> {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("channel", channel);
  
  for (const file of files) {
    form.append("files", file, file.name);
  }
  
  const res = await fetch(`${API_BASE}/v1/ingest/csv`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`
    },
    body: form
  });
  
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text);
  }
  
  return res.json();
}