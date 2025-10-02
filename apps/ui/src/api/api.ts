// Файл: apps/ui/src/api/api.ts

export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
export const WS_BASE = import.meta.env.VITE_WS_BASE || "ws://localhost:8000";

export async function uploadChannelCsv(
  sessionId: string, 
  files: File[], // <-- ИЗМЕНЕНИЕ 1: Принимаем массив файлов
  channel: "bpm" | "uterus",
  token: string
): Promise<any> {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("channel", channel);
  
  // ИЗМЕНЕНИЕ 2: Добавляем каждый файл в FormData
  // Бэкенд FastAPI сможет прочитать их как список
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