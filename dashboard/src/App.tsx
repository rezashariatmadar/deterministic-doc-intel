import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { ControlsPanel } from './components/ControlsPanel';
import { AnswerCanvas } from './components/AnswerCanvas';
import { Inspector } from './components/Inspector';
import {
  ChatMessage,
  DocumentItem,
  HealthStatus,
  MetricsSummaryResponse,
  Preset,
  QueryRequest,
  RAGResponse,
  RetrievedChunk,
} from './types';
import {
  deleteDocument,
  fetchDocuments,
  fetchHealth,
  fetchMetricsSummary,
  streamQueryRag,
  uploadDocument,
  VerificationPayload,
} from './api';

const DEFAULT_OPTIONS: QueryRequest = {
  query: '',
  search_mode: 'hybrid',
  enable_reranker: null,
  enable_agent: false,
  route_override: null,
  allow_external_fallback: false,
  bypass_cache: false,
  top_k: 5,
  weight_dense: 1.0,
  weight_sparse: 1.0,
  enable_query_rewrite: false,
  enable_hyde: false,
  max_crag_retries: 2,
  resolve_parents: true,
};

export const App: React.FC = () => {
  // Session & Options State
  const [sessionId, setSessionId] = useState<string>(() => `sess-${Date.now()}`);
  const [requestOptions, setRequestOptions] = useState<QueryRequest>(DEFAULT_OPTIONS);
  const [inputValue, setInputValue] = useState<string>('');

  // Messages & Streaming State
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Inspector State
  const [currentResponse, setCurrentResponse] = useState<RAGResponse | null>(null);
  const [selectedSourceId, setSelectedSourceId] = useState<number | null>(null);
  const [inspectorTab, setInspectorTab] = useState<'trace' | 'evidence' | 'metrics'>('trace');

  // Documents & Health State
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [docsLoading, setDocsLoading] = useState<boolean>(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(false);

  // Metrics State
  const [metrics, setMetrics] = useState<MetricsSummaryResponse | null>(null);
  const [metricsLoading, setMetricsLoading] = useState<boolean>(false);

  // Responsive UI Drawer State
  const [mobileTab, setMobileTab] = useState<'canvas' | 'controls' | 'inspector'>('canvas');
  const [showInspectorDrawer, setShowInspectorDrawer] = useState<boolean>(false);

  // 1. Load Health
  const loadHealth = useCallback(async () => {
    setHealthLoading(true);
    try {
      const data = await fetchHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  }, []);

  // 2. Load Documents
  const loadDocuments = useCallback(async () => {
    try {
      const docs = await fetchDocuments();
      setDocuments(docs);
    } catch {
      // Fallback
    }
  }, []);

  // 3. Load Metrics
  const loadMetrics = useCallback(async (hours?: number) => {
    setMetricsLoading(true);
    try {
      const data = await fetchMetricsSummary(hours);
      setMetrics(data);
    } catch {
      // Fallback
    } finally {
      setMetricsLoading(false);
    }
  }, []);

  // Initial Data Fetch
  useEffect(() => {
    loadHealth();
    loadDocuments();
    loadMetrics();

    const healthInterval = setInterval(loadHealth, 15000);
    return () => clearInterval(healthInterval);
  }, [loadHealth, loadDocuments, loadMetrics]);

  // Document status polling (every 1.5s while any document is PENDING or PROCESSING)
  useEffect(() => {
    const hasPendingOrProcessing = documents.some(
      (d) => d.status === 'PENDING' || d.status === 'PROCESSING'
    );

    if (hasPendingOrProcessing) {
      const interval = setInterval(async () => {
        try {
          const docs = await fetchDocuments();
          setDocuments(docs);
        } catch {
          // ignore error
        }
      }, 1500);
      return () => clearInterval(interval);
    }
  }, [documents]);

  // Options updater
  const handleUpdateOptions = (updated: Partial<QueryRequest>) => {
    setRequestOptions((prev) => ({ ...prev, ...updated }));
  };

  // Reset conversation
  const handleNewConversation = () => {
    if (isStreaming && abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setSessionId(`sess-${Date.now()}`);
    setMessages([]);
    setCurrentResponse(null);
    setSelectedSourceId(null);
    setInputValue('');
  };

  // Preset selection
  const handleSelectPreset = (preset: Preset) => {
    const newOptions: Partial<QueryRequest> = {
      ...preset.options,
    };
    if (preset.id === 'memory-followup') {
      newOptions.session_id = sessionId;
    }
    setRequestOptions((prev) => ({ ...prev, ...newOptions }));
    setInputValue(preset.query);
    // Switch to canvas on mobile
    setMobileTab('canvas');
  };

  // Document upload
  const handleUploadFile = async (file: File) => {
    setDocsLoading(true);
    try {
      const result = await uploadDocument(file);
      if (result.status === 200 || result.status === 202) {
        await loadDocuments();
      }
    } finally {
      setDocsLoading(false);
    }
  };

  // Document delete
  const handleDeleteDocument = async (doc: DocumentItem) => {
    await deleteDocument(doc.id);
    await loadDocuments();
  };

  // Stop streaming
  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsStreaming(false);
    }
  };

  // Select inline citation chip from canvas -> open Evidence tab and highlight chunk
  const handleSelectCitation = (sourceId: number) => {
    setSelectedSourceId(sourceId);
    setInspectorTab('evidence');
    setShowInspectorDrawer(true);
    setMobileTab('inspector');
  };

  // Send question message
  const handleSendMessage = async (queryText: string) => {
    const text = queryText.trim();
    if (!text || text.length < 3 || isStreaming) return;

    setInputValue('');
    const userMsgId = `user-${Date.now()}`;
    const assistantMsgId = `asst-${Date.now()}`;

    const userMsg: ChatMessage = {
      id: userMsgId,
      role: 'user',
      content: text,
      originalQuery: text,
    };

    const assistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      originalQuery: text,
      isLoading: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    const requestPayload: QueryRequest = {
      ...requestOptions,
      query: text,
      session_id: sessionId,
      stream: true,
    };

    const startTime = performance.now();
    let firstTokenTime: number | null = null;
    let accumulatedContent = '';

    await streamQueryRag(
      requestPayload,
      {
        onCacheHitHeader: (cacheHit: boolean) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantMsgId ? { ...m, cacheHit } : m))
          );
        },
        onToken: (token: string) => {
          if (firstTokenTime === null) {
            firstTokenTime = performance.now();
            const ttft = Math.round(firstTokenTime - startTime);
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantMsgId ? { ...m, ttftMs: ttft } : m))
            );
          }
          accumulatedContent += token;
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantMsgId ? { ...m, content: accumulatedContent } : m))
          );
        },
        onVerification: (verification: VerificationPayload) => {
          const finalResponse: RAGResponse = {
            answer: verification.verified_answer || verification.answer || accumulatedContent,
            is_answerable: verification.is_answerable !== false,
            citations: verification.citations || [],
            sources: (verification.sources as unknown as RetrievedChunk[]) || [],
            confidence_score: verification.confidence_score || 0.0,
            latency_ms: verification.latency_ms || Math.round(performance.now() - startTime),
            route: verification.route,
            route_taken: verification.route || 'FAST_TRACK_HYBRID_RAG',
            rewritten_query: verification.rewritten_query,
            crag_retries: verification.crag_retry_count || 0,
            crag_retry_count: verification.crag_retry_count || 0,
            sql_query_executed: verification.sql_query_executed,
            candidate_facts: verification.candidate_facts || [],
            execution_trace: verification.execution_trace || [],
            stage_latencies: verification.stage_latencies || {},
          };

          setCurrentResponse(finalResponse);

          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    content: finalResponse.answer,
                    rewrittenQuery: finalResponse.rewritten_query,
                    response: finalResponse,
                    isAbstain: !finalResponse.is_answerable,
                    isLoading: false,
                  }
                : m
            )
          );

          // Refresh metrics summary in background
          loadMetrics();
        },
        onError: (err) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    content: '',
                    error: err.message,
                    blockedPattern: err.pattern,
                    isLoading: false,
                  }
                : m
            )
          );
        },
        onDone: () => {
          setIsStreaming(false);
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantMsgId ? { ...m, isLoading: false } : m))
          );
        },
      },
      abortController.signal
    );
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-100 text-slate-900 font-sans">
      {/* Top Header Strip */}
      <Header
        health={health}
        healthLoading={healthLoading}
        onNewConversation={handleNewConversation}
        onRefreshHealth={loadHealth}
        onToggleControls={() => setMobileTab((t) => (t === 'controls' ? 'canvas' : 'controls'))}
        onToggleInspector={() => setShowInspectorDrawer((d) => !d)}
      />

      {/* Main 3-Zone Workspace */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Zone 1: Controls and Corpus (Right Edge - Reading Start in RTL) */}
        <aside
          className={`w-80 md:w-88 xl:w-96 border-e border-slate-200 bg-white shrink-0 h-full overflow-hidden transition-all duration-300 z-20 ${
            mobileTab === 'controls' ? 'fixed inset-0 top-14 w-full z-40' : 'hidden lg:flex flex-col'
          }`}
        >
          {mobileTab === 'controls' && (
            <div className="p-2 border-b border-slate-200 flex justify-end lg:hidden">
              <button
                onClick={() => setMobileTab('canvas')}
                className="px-3 py-1 bg-slate-900 text-white rounded text-xs font-semibold"
              >
                بستن و بازگشت به گفتگو
              </button>
            </div>
          )}
          <ControlsPanel
            requestOptions={requestOptions}
            onChangeOptions={handleUpdateOptions}
            onSelectPreset={handleSelectPreset}
            documents={documents}
            docsLoading={docsLoading}
            onUploadFile={handleUploadFile}
            onDeleteDocument={handleDeleteDocument}
            onResetDefaults={() => setRequestOptions(DEFAULT_OPTIONS)}
          />
        </aside>

        {/* Zone 2: Answer Canvas (Center) */}
        <main
          className={`flex-1 h-full overflow-hidden flex flex-col transition-all ${
            mobileTab !== 'canvas' ? 'hidden md:flex' : 'flex'
          }`}
        >
          <AnswerCanvas
            messages={messages}
            inputValue={inputValue}
            onChangeInput={setInputValue}
            onSendMessage={handleSendMessage}
            onStopStreaming={handleStopStreaming}
            isStreaming={isStreaming}
            onSelectCitation={handleSelectCitation}
          />
        </main>

        {/* Zone 3: Inspector (Left Edge - End in RTL) */}
        {/* Desktop View (>= 1280px) */}
        <aside className="w-96 xl:w-[440px] 2xl:w-[480px] border-s border-slate-200 bg-white shrink-0 h-full overflow-hidden hidden xl:flex flex-col z-20">
          <Inspector
            currentResponse={currentResponse}
            selectedSourceId={selectedSourceId}
            metrics={metrics}
            metricsLoading={metricsLoading}
            onRefreshMetrics={loadMetrics}
            activeTab={inspectorTab}
            onChangeTab={setInspectorTab}
          />
        </aside>

        {/* Tablet / Drawer View (< 1280px) */}
        {showInspectorDrawer && (
          <div className="fixed inset-0 z-50 flex justify-start bg-slate-900/40 backdrop-blur-xs xl:hidden">
            <div className="w-full max-w-md h-full bg-white shadow-2xl flex flex-col border-e border-slate-200 animate-in slide-in-from-start duration-200">
              <div className="p-3 border-b border-slate-200 flex items-center justify-between bg-slate-50">
                <span className="font-bold text-xs text-slate-800">پنل بازرسی (Inspector)</span>
                <button
                  type="button"
                  onClick={() => setShowInspectorDrawer(false)}
                  className="px-2.5 py-1 text-xs bg-slate-200 hover:bg-slate-300 rounded font-medium text-slate-700"
                >
                  بستن
                </button>
              </div>
              <div className="flex-1 overflow-hidden">
                <Inspector
                  currentResponse={currentResponse}
                  selectedSourceId={selectedSourceId}
                  metrics={metrics}
                  metricsLoading={metricsLoading}
                  onRefreshMetrics={loadMetrics}
                  activeTab={inspectorTab}
                  onChangeTab={setInspectorTab}
                />
              </div>
            </div>
            <div className="flex-1" onClick={() => setShowInspectorDrawer(false)} />
          </div>
        )}
      </div>

      {/* Mobile Bottom Tab Switcher (< 768px) */}
      <div className="md:hidden h-12 bg-white border-t border-slate-200 grid grid-cols-3 text-xs shrink-0 select-none">
        <button
          onClick={() => setMobileTab('controls')}
          className={`flex items-center justify-center font-medium ${
            mobileTab === 'controls' ? 'text-teal-700 font-bold bg-teal-50' : 'text-slate-600'
          }`}
        >
          تنظیمات و اسناد
        </button>
        <button
          onClick={() => setMobileTab('canvas')}
          className={`flex items-center justify-center font-medium ${
            mobileTab === 'canvas' ? 'text-teal-700 font-bold bg-teal-50' : 'text-slate-600'
          }`}
        >
          گفتگو و پاسخ
        </button>
        <button
          onClick={() => {
            setMobileTab('canvas');
            setShowInspectorDrawer(true);
          }}
          className="flex items-center justify-center font-medium text-slate-600 hover:text-teal-700"
        >
          بازرسی شواهد
        </button>
      </div>
    </div>
  );
};
