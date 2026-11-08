/**
 * API client communicating with FastAPI RAG backend.
 */

import {
  DocumentItem,
  HealthStatus,
  MetricsSummaryResponse,
  QueryRequest,
  RAGResponse,
} from './types';

const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1';

export async function fetchHealth(): Promise<HealthStatus> {
  const resp = await fetch(`${API_BASE}/health`);
  if (!resp.ok) {
    throw new Error(`خطای بررسی سلامت سامانه: ${resp.status}`);
  }
  return resp.json();
}

export async function fetchDocuments(): Promise<DocumentItem[]> {
  const resp = await fetch(`${API_BASE}/documents`);
  if (!resp.ok) {
    throw new Error(`خطای دریافت فهرست اسناد: ${resp.status}`);
  }
  return resp.json();
}

export interface UploadResult {
  status: number;
  data: DocumentItem | { message: string; document_id?: string; task_id?: string };
}

export async function uploadDocument(file: File): Promise<UploadResult> {
  const formData = new FormData();
  formData.append('file', file);

  const resp = await fetch(`${API_BASE}/documents/upload?async_mode=true`, {
    method: 'POST',
    body: formData,
  });

  const data = await resp.json();
  return {
    status: resp.status,
    data,
  };
}

export async function deleteDocument(id: string): Promise<void> {
  const resp = await fetch(`${API_BASE}/documents/${id}`, {
    method: 'DELETE',
  });
  if (!resp.ok) {
    const errorBody = await resp.json().catch(() => ({}));
    throw new Error(errorBody.detail || `خطا در حذف سند: ${resp.status}`);
  }
}

export async function fetchMetricsSummary(timeWindowHours?: number): Promise<MetricsSummaryResponse> {
  let url = `${API_BASE}/metrics/summary`;
  if (timeWindowHours) {
    const startTime = new Date(Date.now() - timeWindowHours * 3600 * 1000).toISOString();
    url += `?start_time=${encodeURIComponent(startTime)}`;
  }
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`خطا در دریافت شاخص‌های عملکرد: ${resp.status}`);
  }
  return resp.json();
}

export async function queryRagSync(
  request: QueryRequest
): Promise<{ response: RAGResponse; cacheHit: boolean }> {
  const resp = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  const cacheHitHeader = resp.headers.get('X-Cache-Hit');
  const cacheHit = cacheHitHeader === 'true';

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    const message = err.detail || err.message || `خطای سرور: ${resp.status}`;
    const pattern = err.pattern || (err.details && err.details.pattern);
    const customError = new Error(typeof message === 'string' ? message : JSON.stringify(message));
    (customError as unknown as Record<string, unknown>).status = resp.status;
    (customError as unknown as Record<string, unknown>).pattern = pattern;
    throw customError;
  }

  const data: RAGResponse = await resp.json();
  return { response: data, cacheHit };
}

export interface VerificationPayload extends Partial<RAGResponse> {
  verified_answer?: string;
}

export interface StreamCallbacks {
  onToken?: (token: string) => void;
  onSources?: (sources: unknown[]) => void;
  onVerification?: (verification: VerificationPayload) => void;
  onError?: (err: Error & { pattern?: string; status?: number }) => void;
  onDone?: () => void;
  onCacheHitHeader?: (cacheHit: boolean) => void;
}

export async function streamQueryRag(
  request: QueryRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  try {
    const resp = await fetch(`${API_BASE}/query/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ...request, stream: true }),
      signal,
    });

    const cacheHitHeader = resp.headers.get('X-Cache-Hit');
    if (callbacks.onCacheHitHeader) {
      callbacks.onCacheHitHeader(cacheHitHeader === 'true');
    }

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      const message = err.detail || err.message || `خطای سرور: ${resp.status}`;
      const pattern = err.pattern || (err.details && err.details.pattern);
      const customError = new Error(typeof message === 'string' ? message : JSON.stringify(message)) as Error & {
        pattern?: string;
        status?: number;
      };
      customError.status = resp.status;
      customError.pattern = pattern;
      if (callbacks.onError) {
        callbacks.onError(customError);
      }
      return;
    }

    if (!resp.body) {
      throw new Error('پاسخ سرور فاقد جریان داده است.');
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data: ')) continue;
        const dataStr = trimmed.slice(6);
        if (dataStr === '[DONE]') {
          if (callbacks.onDone) callbacks.onDone();
          return;
        }

        try {
          const parsed = JSON.parse(dataStr);
          if (parsed.event === 'token' && callbacks.onToken) {
            callbacks.onToken(parsed.content || '');
          } else if (parsed.event === 'sources' && callbacks.onSources) {
            callbacks.onSources(parsed.data || []);
          } else if (parsed.event === 'verification' && callbacks.onVerification) {
            callbacks.onVerification(parsed.data || {});
          }
        } catch {
          // Ignore malformed partial chunks
        }
      }
    }

    if (callbacks.onDone) callbacks.onDone();
  } catch (err: unknown) {
    if ((err as Error).name === 'AbortError') {
      if (callbacks.onDone) callbacks.onDone();
      return;
    }
    if (callbacks.onError) {
      callbacks.onError(err as Error);
    }
  }
}
