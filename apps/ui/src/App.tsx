// apps/ui/src/App.tsx
import React, { useState, useCallback, useMemo } from "react"; // <-- Добавьте useMemo
import CtgChart from "./components/CtgChart";
import RiskGauge from "./components/RiskGauge";
import EventTimeline from "./components/EventTimeline";
import { Login } from "./components/Login";
import { useAuth } from "./contexts/AuthContext";
import { useWebSocket } from "./hooks/useWebSocket";
import { WS_BASE, uploadChannelCsv } from "./api/api";

type RtPayload = {
  ts: string;
  risk: { hypoxia_prob: number; band: string };
  series: { ts: string[]; bpm: number[]; ua: number[]; baseline_60s: number[] };
  decel_events: { 
    start_ts: string; 
    end_ts: string; 
    dur_s?: number; 
    min_bpm?: number; 
    max_drop?: number 
  }[];
};

const App: React.FC = () => {
  const { user, token, logout } = useAuth();
  const [sessionId, setSessionId] = useState("demo-session");
  const [last, setLast] = useState<RtPayload | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string>("");
  
  // ИЗМЕНЕНИЕ 1: Храним массивы файлов
  const [bpmFiles, setBpmFiles] = useState<File[]>([]);
  const [uterusFiles, setUterusFiles] = useState<File[]>([]);

  const handleWebSocketMessage = useCallback((data: RtPayload) => {
    if (data && data.risk && data.series) {
      setLast(data);
    }
  }, []);

  const handleWebSocketError = useCallback((error: Event) => {
    console.error('WebSocket error:', error);
  }, []);

  const webSocketUrl = useMemo(() => {
    return `${WS_BASE}/v1/stream/${encodeURIComponent(sessionId)}`;
  }, [sessionId]);

  const { isConnected } = useWebSocket({
    url: webSocketUrl,
    token: token,
    onMessage: handleWebSocketMessage,
    onError: handleWebSocketError,
  });

  // ИЗМЕНЕНИЕ 2: Обновляем обработчики для работы с несколькими файлами
  const handleBpmFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setBpmFiles(Array.from(e.target.files));
    }
  };

  const handleUterusFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setUterusFiles(Array.from(e.target.files));
    }
  };

  // ИЗМЕНЕНИЕ 3: Обновляем логику загрузки
  const handleUpload = async () => {
    if (!token) {
      setUploadStatus("Please login first");
      return;
    }

    if (bpmFiles.length === 0 && uterusFiles.length === 0) {
      setUploadStatus("Please select at least one file");
      return;
    }

    setUploadStatus("Uploading...");
    
    try {
      const promises = [];
      
      if (bpmFiles.length > 0) {
        promises.push(uploadChannelCsv(sessionId, bpmFiles, "bpm", token));
      }
      
      if (uterusFiles.length > 0) {
        promises.push(uploadChannelCsv(sessionId, uterusFiles, "uterus", token));
      }
      
      await Promise.all(promises);
      
      setUploadStatus("Upload successful! Processing...");
      setTimeout(() => setUploadStatus(""), 3000);
    } catch (err) {
      setUploadStatus(`Upload failed: ${err}`);
    }
  };

  if (!user || !token) {
    return <Login />;
  }

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h2>CTG Monitor</h2>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span>Welcome, {user.email}</span>
          <button onClick={logout}>Logout</button>
        </div>
      </div>

      <div style={{ marginBottom: 16, padding: 16, border: "1px solid #ddd", borderRadius: 8 }}>
        <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
          <label>Session ID:</label>
          <input 
            value={sessionId} 
            onChange={(e) => setSessionId(e.target.value)} 
            disabled={isConnected}
            style={{ flex: 1 }}
          />
          <span style={{ color: isConnected ? "green" : "orange" }}>
            {isConnected ? "● Connected" : "○ Connecting..."}
          </span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
          <div>
            <label style={{ display: "block", marginBottom: 4, fontWeight: "bold" }}>
              BPM File(s) (Heart Rate):
            </label>
            <input 
              type="file" 
              accept=".csv,.tsv" 
              onChange={handleBpmFileChange}
              multiple // <-- ИЗМЕНЕНИЕ 4: Разрешаем выбор нескольких файлов
            />
            {/* ИЗМЕНЕНИЕ 5: Отображаем количество выбранных файлов */}
            {bpmFiles.length > 0 && <div style={{ fontSize: 12, color: "green", marginTop: 4 }}>✓ {bpmFiles.length} file(s) selected</div>}
          </div>

          <div>
            <label style={{ display: "block", marginBottom: 4, fontWeight: "bold" }}>
              Uterus File(s) (Contractions):
            </label>
            <input 
              type="file" 
              accept=".csv,.tsv" 
              onChange={handleUterusFileChange}
              multiple // <-- ИЗМЕНЕНИЕ 4: Разрешаем выбор нескольких файлов
            />
            {uterusFiles.length > 0 && <div style={{ fontSize: 12, color: "green", marginTop: 4 }}>✓ {uterusFiles.length} file(s) selected</div>}
          </div>
        </div>

        <button 
          onClick={handleUpload}
          disabled={bpmFiles.length === 0 && uterusFiles.length === 0} // <-- ИЗМЕНЕНИЕ 6: Обновляем условие
          style={{
            width: "100%",
            padding: 12,
            backgroundColor: (bpmFiles.length === 0 && uterusFiles.length === 0) ? "#ccc" : "#007bff",
            color: "white",
            border: "none",
            borderRadius: 4,
            cursor: (bpmFiles.length === 0 && uterusFiles.length === 0) ? "not-allowed" : "pointer"
          }}
        >
          Upload Files
        </button>

        {uploadStatus && (
          <div style={{ 
            marginTop: 8, 
            padding: 8, 
            backgroundColor: uploadStatus.includes("failed") ? "#ffe6e6" : "#e6f7ff",
            color: uploadStatus.includes("failed") ? "red" : "green",
            borderRadius: 4
          }}>
            {uploadStatus}
          </div>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 16 }}>
        <div>
          <CtgChart
            ts={last?.series.ts || []}
            bpm={last?.series.bpm || []}
            baseline={last?.series.baseline_60s || []}
            ua={last?.series.ua || []}
            decelEvents={last?.decel_events || []}
          />
        </div>
        <div>
          <RiskGauge prob={last?.risk.hypoxia_prob || 0} />
          <div style={{ marginTop: 12 }}>
            <h4>Decelerations</h4>
            <EventTimeline events={last?.decel_events || []} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;