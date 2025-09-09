import Store from "electron-store";

type Counters = {
  tokensBeforeTotal: number;
  tokensAfterTotal: number;
  requestsTotal: number;
  tokensBeforeToday: number;
  tokensAfterToday: number;
  requestsToday: number;
  lastDayISO: string;
};

const store = new Store<Counters>({
  name: "metrics",
  defaults: {
    tokensBeforeTotal: 0,
    tokensAfterTotal: 0,
    requestsTotal: 0,
    tokensBeforeToday: 0,
    tokensAfterToday: 0,
    requestsToday: 0,
    lastDayISO: new Date().toISOString().slice(0,10)
  }
});

function maybeRollover() {
  const today = new Date().toISOString().slice(0,10);
  const last = store.get("lastDayISO");
  if (last !== today) {
    store.set("tokensBeforeToday", 0);
    store.set("tokensAfterToday", 0);
    store.set("requestsToday", 0);
    store.set("lastDayISO", today);
  }
}

export function recordRequest(tokensBefore: number, tokensAfter: number) {
  maybeRollover();
  store.set("tokensBeforeTotal", store.get("tokensBeforeTotal") + tokensBefore);
  store.set("tokensAfterTotal", store.get("tokensAfterTotal") + tokensAfter);
  store.set("requestsTotal", store.get("requestsTotal") + 1);
  store.set("tokensBeforeToday", store.get("tokensBeforeToday") + tokensBefore);
  store.set("tokensAfterToday", store.get("tokensAfterToday") + tokensAfter);
  store.set("requestsToday", store.get("requestsToday") + 1);
}

export function getMetricsSnapshot() {
  maybeRollover();
  const bTot = store.get("tokensBeforeTotal");
  const aTot = store.get("tokensAfterTotal");
  const savedTot = Math.max(0, bTot - aTot);
  const pctTot = bTot > 0 ? Math.round((savedTot / bTot) * 100) : 0;

  const bDay = store.get("tokensBeforeToday");
  const aDay = store.get("tokensAfterToday");
  const savedDay = Math.max(0, bDay - aDay);
  const pctDay = bDay > 0 ? Math.round((savedDay / bDay) * 100) : 0;

  return {
    totals: { before: bTot, after: aTot, saved: savedTot, pct: pctTot, requests: store.get("requestsTotal") },
    today: { before: bDay, after: aDay, saved: savedDay, pct: pctDay, requests: store.get("requestsToday") }
  };
}

export function resetToday() {
  store.set("tokensBeforeToday", 0);
  store.set("tokensAfterToday", 0);
  store.set("requestsToday", 0);
}
