import React from "react";

type Ev = { start_ts: string; end_ts: string; dur_s?: number; min_bpm?: number; max_drop?: number };

const EventTimeline: React.FC<{ events: Ev[] }> = ({ events }) => {
  if (!events?.length) return <div>Событий не обнаружено</div>;
  return (
    <div style={{ maxHeight: 160, overflowY: "auto" }}>
      <table style={{ width: "100%", fontSize: 13 }}>
        <thead>
          <tr>
            <th align="left">Начало</th>
            <th align="left">Окончание</th>
            <th align="right">Длит., c</th>
            <th align="right">Падение, bpm</th>
            <th align="right">Min FHR</th>
          </tr>
        </thead>
        <tbody>
          {events.map((e, i) => (
            <tr key={i}>
              <td>{e.start_ts.slice(11,19)}</td>
              <td>{e.end_ts.slice(11,19)}</td>
              <td align="right">{e.dur_s?.toFixed(0)}</td>
              <td align="right">{e.max_drop?.toFixed(1)}</td>
              <td align="right">{e.min_bpm?.toFixed(0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default EventTimeline;