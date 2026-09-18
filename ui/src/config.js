const configuredApiUrl = (process.env.REACT_APP_API_URL || "").trim();
const isLocalBrowser =
  typeof window !== "undefined"
  && ["localhost", "127.0.0.1"].includes(window.location.hostname);

// Production must use the same-origin Vercel function, which invokes SageMaker.
// REACT_APP_API_URL is only for direct local FastAPI development.
export const API_URL = isLocalBrowser ? configuredApiUrl : "";

const SESSION_STORAGE_KEY = "repo_qa_session_id";

export function getSessionId() {
  let value = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (!value) {
    value = window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, value);
    console.log("Created session id:", value);
  }
  return value;
}

export function getSessionHeaders() {
  return {
    "X-Session-Id": getSessionId(),
  };
}
