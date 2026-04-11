export const API_URL = process.env.REACT_APP_API_URL;

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
