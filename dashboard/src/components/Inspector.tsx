import React, { useState, useEffect } from 'react';
import {
  Activity,
  Layers,
  BarChart3,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  XCircle,
  Copy,
  Check,
  Clock,
  Maximize2,
  Minimize2,
  Database,
  Info,
} from 'lucide-react';
import {
  ExtractionStatus,
  MetricsSummaryResponse,
  RAGResponse,
} from '../types';

interface InspectorProps {
  currentResponse: RAGResponse | null;
  selectedSourceId: number | null;
  metrics: MetricsSummaryResponse | null;
  metricsLoading: boolean;
  onRefreshMetrics: (hours?: number) => void;
  activeTab: 'trace' | 'evidence' | 'metrics';
  onChangeTab: (tab: 'trace' | 'evidence' | 'metrics') => void;
}

export const Inspector: React.FC<InspectorProps> = ({
  currentResponse,
  selectedSourceId,
  metrics,
  metricsLoading,
  onRefreshMetrics,
  activeTab,
  onChangeTab,
}) => {
  const [expandedParents, setExpandedParents] = useState<Record<number, boolean>>({});
  const [copiedSql, setCopiedSql] = useState(false);
  const [timeWindowHours, setTimeWindowHours] = useState<number | undefined>(undefined);

  // Auto-scroll to selected source chunk
  useEffect(() => {
    if (selectedSourceId != null && activeTab === 'evidence') {
      const el = document.getElementById(`chunk-card-${selectedSourceId}`);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [selectedSourceId, activeTab]);

  const toggleParentExpansion = (sourceId: number) => {
    setExpandedParents((prev) => ({
      ...prev,
      [sourceId]: !prev[sourceId],
    }));
  };

  const copySqlToClipboard = (sql: string) => {
    navigator.clipboard.writeText(sql);
    setCopiedSql(true);
    setTimeout(() => setCopiedSql(false), 2000);
  };

  const getStatusBadge = (status: ExtractionStatus) => {
    switch (status) {
      case 'verified':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
            <CheckCircle2 className="w-3 h-3" />
            <span>تاییدشده (Verified)</span>
          </span>
        );
      case 'ambiguous':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-amber-50 text-amber-700 border border-amber-200">
            <HelpCircle className="w-3 h-3" />
            <span>مبهم (Ambiguous)</span>
          </span>
        );
      case 'conflicting':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-orange-50 text-orange-700 border border-orange-200">
            <AlertCircle className="w-3 h-3" />
            <span>متناقض (Conflicting)</span>
          </span>
        );
      case 'extraction_failed':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-rose-50 text-rose-700 border border-rose-200">
            <XCircle className="w-3 h-3" />
            <span>خطای استخراج</span>
          </span>
        );
      case 'unsupported':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-purple-50 text-purple-700 border border-purple-200">
            <AlertCircle className="w-3 h-3" />
            <span>پشتیبانی‌نشده</span>
          </span>
        );
      case 'not_found':
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-700 border border-slate-200">
            <Info className="w-3 h-3" />
            <span>یافت نشد</span>
          </span>
        );
    }
  };

  // Helper to parse markdown table if present in chunk text
  const renderChunkContent = (text: string, isTable: boolean) => {
    const lines = text.trim().split('\n');
    const isMarkdownTable =
      lines.length >= 2 &&
      lines[0].includes('|') &&
      lines[1].includes('|') &&
      lines[1].includes('-');

    if (isTable || isMarkdownTable) {
      const tableRows = lines.filter((l) => l.includes('|'));
      if (tableRows.length >= 2) {
        const parseRow = (rowStr: string) =>
          rowStr
            .split('|')
            .map((c) => c.trim())
            .filter((c, idx, arr) => !(idx === 0 && c === '') && !(idx === arr.length - 1 && c === ''));

        const headerCols = parseRow(tableRows[0]);
        const bodyRows = tableRows.slice(2).map(parseRow);

        return (
          <div className="overflow-x-auto my-2 border border-slate-200 rounded-lg">
            <table className="w-full text-[11px] text-start border-collapse dir-ltr">
              <thead>
                <tr className="bg-slate-100 border-b border-slate-200 font-semibold text-slate-700">
                  {headerCols.map((col, idx) => (
                    <th key={idx} className="p-2 border-e border-slate-200 text-start">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {bodyRows.map((row, rIdx) => (
                  <tr key={rIdx} className="hover:bg-slate-50 font-mono-numbers">
                    {row.map((cell, cIdx) => (
                      <td key={cIdx} className="p-2 border-e border-slate-100 text-start">
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      }
    }

    return (
      <div className="text-[11px] font-sans leading-relaxed text-slate-800 whitespace-pre-wrap dir-ltr text-start">
        {text}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-white select-none text-xs">
      {/* Tab Navigation Strip */}
      <div className="flex items-center border-b border-slate-200 px-2 bg-slate-50 shrink-0">
        <button
          type="button"
          onClick={() => onChangeTab('trace')}
          className={`flex items-center gap-1.5 px-3 py-3 font-semibold text-xs border-b-2 transition-all ${
            activeTab === 'trace'
              ? 'border-teal-600 text-teal-700 bg-white'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          <Activity className="w-3.5 h-3.5" />
          <span>ردیابی عامل (Trace)</span>
          {currentResponse?.execution_trace && currentResponse.execution_trace.length > 0 && (
            <span className="ms-1 px-1.5 py-0.2 rounded-full bg-teal-100 text-teal-800 text-[10px] font-mono-numbers">
              {currentResponse.execution_trace.length}
            </span>
          )}
        </button>

        <button
          type="button"
          onClick={() => onChangeTab('evidence')}
          className={`flex items-center gap-1.5 px-3 py-3 font-semibold text-xs border-b-2 transition-all ${
            activeTab === 'evidence'
              ? 'border-teal-600 text-teal-700 bg-white'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>شواهد و قطعات (Evidence)</span>
          {currentResponse?.sources && currentResponse.sources.length > 0 && (
            <span className="ms-1 px-1.5 py-0.2 rounded-full bg-teal-100 text-teal-800 text-[10px] font-mono-numbers">
              {currentResponse.sources.length}
            </span>
          )}
        </button>

        <button
          type="button"
          onClick={() => onChangeTab('metrics')}
          className={`flex items-center gap-1.5 px-3 py-3 font-semibold text-xs border-b-2 transition-all ${
            activeTab === 'metrics'
              ? 'border-teal-600 text-teal-700 bg-white'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          <BarChart3 className="w-3.5 h-3.5" />
          <span>شاخص‌های سیستم (Metrics)</span>
        </button>
      </div>

      {/* Tab 1: Agent Execution Trace */}
      {activeTab === 'trace' && (
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* SQL Query Monospace Block (if executed) */}
          {currentResponse?.sql_query_executed && (
            <div className="rounded-xl border border-indigo-200 bg-indigo-50/40 p-3.5 shadow-2xs">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-indigo-900 flex items-center gap-1.5 text-xs">
                  <Database className="w-4 h-4 text-indigo-600" />
                  <span>کوئری پایگاه داده مالی اجراشده (Safe Text-to-SQL):</span>
                </span>
                <button
                  type="button"
                  onClick={() => copySqlToClipboard(currentResponse.sql_query_executed!)}
                  className="p-1 rounded text-indigo-600 hover:bg-indigo-100 transition-colors flex items-center gap-1 text-[11px]"
                  title="کپی کوئری SQL"
                >
                  {copiedSql ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-600" />
                      <span className="text-emerald-700 font-medium">کپی شد</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      <span>کپی SQL</span>
                    </>
                  )}
                </button>
              </div>
              <pre className="dir-ltr text-start p-2.5 rounded-lg bg-slate-900 text-teal-300 font-mono text-[11px] overflow-x-auto whitespace-pre-wrap leading-relaxed shadow-inner">
                {currentResponse.sql_query_executed}
              </pre>
            </div>
          )}

          {/* Standard Path Stage Latencies (when not agent) */}
          {currentResponse?.stage_latencies && Object.keys(currentResponse.stage_latencies).length > 0 && (
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5 flex flex-col gap-2">
              <div className="font-bold text-slate-800 text-xs flex items-center gap-1.5">
                <Clock className="w-4 h-4 text-teal-600" />
                <span>زمان‌بندی مراحل پردازش استاندارد:</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-[11px] font-mono-numbers">
                {Object.entries(currentResponse.stage_latencies).map(([stage, ms]) => (
                  <div
                    key={stage}
                    className="p-2 bg-white rounded-lg border border-slate-200 flex items-center justify-between"
                  >
                    <span className="text-slate-600 truncate dir-ltr text-start">{stage}:</span>
                    <span className="font-semibold text-teal-700">{Math.round(ms * 100) / 100} ms</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Trace Timeline Steps */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-800 text-xs">
                خط زمانی نودهای عامل ({currentResponse?.execution_trace?.length || 0} مرحله)
              </span>
              {currentResponse?.route && (
                <span className="px-2 py-0.5 rounded bg-teal-50 text-teal-700 border border-teal-200 text-[10px] font-mono-numbers font-medium">
                  مسیر: {currentResponse.route}
                </span>
              )}
            </div>

            {!currentResponse?.execution_trace || currentResponse.execution_trace.length === 0 ? (
              <div className="text-center py-10 text-slate-400 bg-slate-50 rounded-xl border border-slate-100">
                هیچ مرحله ردیابی فعالی برای این پرسش ثبت نشده است (پایپ‌لاین استاندارد بدون حلقه عامل).
              </div>
            ) : (
              <div className="relative border-s-2 border-teal-200 ms-3 space-y-3 ps-4 my-2">
                {currentResponse.execution_trace.map((step, idx) => {
                  const duration = step.duration_ms ?? step.latency_ms ?? 0;
                  return (
                    <div
                      key={idx}
                      className="relative p-3 rounded-xl border border-slate-200 bg-white hover:border-slate-300 transition-colors shadow-2xs"
                    >
                      {/* Step node dot */}
                      <div className="absolute -start-[23px] top-3.5 w-3 h-3 rounded-full bg-teal-600 ring-4 ring-white" />

                      <div className="flex items-center justify-between gap-2 mb-1">
                        <span className="font-bold text-slate-900 text-xs font-mono-numbers dir-ltr text-start">
                          {step.step || step.step_name || `مرحله ${idx + 1}`}
                        </span>
                        <span className="text-[10px] font-mono-numbers text-teal-700 font-semibold bg-teal-50 px-1.5 py-0.5 rounded">
                          {Math.round(duration * 10) / 10} ms
                        </span>
                      </div>

                      {step.action && (
                        <p className="text-[11px] text-slate-600 mb-1 leading-relaxed dir-ltr text-start">
                          {step.action}
                        </p>
                      )}

                      {/* Detail or metadata */}
                      {step.detail && step.detail !== step.action && (
                        <p className="text-[11px] text-slate-500 bg-slate-50 p-1.5 rounded dir-ltr text-start">
                          {step.detail}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab 2: Evidence & Candidate Facts */}
      {activeTab === 'evidence' && (
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* Candidate Facts Ledger */}
          <div className="flex flex-col gap-2.5">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-800 text-xs">
                دفتر کل ادعاها و گزاره‌های استخراج‌شده ({currentResponse?.candidate_facts?.length || 0})
              </span>
              <span className="text-[10px] text-slate-400">بررسی مستقل اتمیک فکت‌ها</span>
            </div>

            {!currentResponse?.candidate_facts || currentResponse.candidate_facts.length === 0 ? (
              <div className="text-center py-4 text-slate-400 bg-slate-50 rounded-xl border border-slate-100 text-xs">
                گزاره اتمیکی برای نمایش وجود ندارد.
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                {currentResponse.candidate_facts.map((cf, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-xl border border-slate-200 bg-white flex flex-col gap-1.5 shadow-2xs"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <span className="text-xs font-medium text-slate-900 dir-ltr text-start leading-relaxed flex-1">
                        {cf.claim_text || cf.fact}
                      </span>
                      {getStatusBadge(cf.status)}
                    </div>

                    {(cf.supporting_quote || cf.evidence) && (
                      <div className="p-2 bg-slate-50 rounded-lg border border-slate-100 text-[11px] text-slate-600 italic dir-ltr text-start leading-relaxed">
                        &ldquo;{cf.supporting_quote || cf.evidence}&rdquo;
                      </div>
                    )}

                    <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono-numbers mt-0.5">
                      {cf.document_title && (
                        <span className="truncate max-w-[200px] dir-ltr text-start">
                          {cf.document_title}
                        </span>
                      )}
                      <span>ضریب تایید: {Math.round((cf.confidence || 1.0) * 100)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Retrieved Chunks Ledger */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-800 text-xs">
                قطعات متنی بازیافت‌شده از پیکره ({currentResponse?.sources?.length || 0})
              </span>
              <span className="text-[10px] text-slate-400">مرتب‌شده بر اساس رتبه‌بندی نهایی</span>
            </div>

            {!currentResponse?.sources || currentResponse.sources.length === 0 ? (
              <div className="text-center py-6 text-slate-400 bg-slate-50 rounded-xl border border-slate-100 text-xs">
                قطعه‌ای برای نمایش یافت نشد.
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                {currentResponse.sources.map((chunk) => {
                  const isHighlighted = selectedSourceId === chunk.source_id;
                  const isParentExpanded = Boolean(expandedParents[chunk.source_id]);
                  const displayText = isParentExpanded
                    ? chunk.contextualized_text || chunk.text
                    : chunk.text;

                  const isTable = chunk.element_type === 'table';

                  return (
                    <div
                      key={chunk.source_id}
                      id={`chunk-card-${chunk.source_id}`}
                      className={`p-3.5 rounded-xl border transition-all shadow-xs ${
                        isHighlighted
                          ? 'border-teal-500 ring-2 ring-teal-200 bg-teal-50/20'
                          : 'border-slate-200 bg-white hover:border-slate-300'
                      }`}
                    >
                      {/* Chunk Header */}
                      <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-2 mb-2">
                        <div className="flex items-center gap-1.5">
                          <span className="font-mono-numbers font-bold text-teal-700 bg-teal-50 px-2 py-0.5 rounded text-xs border border-teal-200">
                            [{chunk.source_id}]
                          </span>
                          <span
                            className="font-semibold text-slate-800 text-xs dir-ltr text-start line-clamp-1"
                            title={chunk.document_title}
                          >
                            {chunk.document_title}
                          </span>
                        </div>

                        <div className="flex items-center gap-1 text-[10px] text-slate-500 font-mono-numbers">
                          <span>ص {chunk.page_start}</span>
                          {chunk.page_end > chunk.page_start && <span>تا {chunk.page_end}</span>}
                          {isTable && (
                            <span className="ms-1 px-1.5 py-0.2 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
                              جدول
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Section path */}
                      {chunk.section_path && (
                        <div className="text-[10px] text-slate-500 mb-2 truncate dir-ltr text-start">
                          بخش: {chunk.section_path}
                        </div>
                      )}

                      {/* Ranking Badges */}
                      <div className="flex flex-wrap items-center gap-1.5 mb-2.5 text-[10px] font-mono-numbers">
                        {chunk.dense_rank != null && (
                          <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700">
                            Dense #{chunk.dense_rank}
                          </span>
                        )}
                        {chunk.sparse_rank != null && (
                          <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700">
                            BM25 #{chunk.sparse_rank}
                          </span>
                        )}
                        {chunk.rrf_score != null && (
                          <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700">
                            RRF: {Math.round(chunk.rrf_score * 1000) / 1000}
                          </span>
                        )}
                        {chunk.rerank_score != null && (
                          <span className="px-1.5 py-0.5 rounded bg-teal-50 text-teal-800 font-semibold border border-teal-200">
                            Rerank: {Math.round(chunk.rerank_score * 1000) / 1000}
                          </span>
                        )}
                        {chunk.similarity_score != null && (
                          <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-600">
                            Sim: {Math.round(chunk.similarity_score * 100)}%
                          </span>
                        )}
                      </div>

                      {/* Chunk Content Body */}
                      <div className="p-2.5 rounded-lg bg-slate-50/70 border border-slate-100 overflow-hidden">
                        {renderChunkContent(displayText, isTable)}
                      </div>

                      {/* Parent Expansion Toggle */}
                      {chunk.parent_chunk_id && (
                        <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500 pt-1.5 border-t border-slate-100">
                          <span className="text-[10px]">دارای قطعه والد سلسله‌مراتبی</span>
                          <button
                            type="button"
                            onClick={() => toggleParentExpansion(chunk.source_id)}
                            className="text-teal-700 hover:text-teal-900 font-medium flex items-center gap-1 text-[11px]"
                          >
                            {isParentExpanded ? (
                              <>
                                <Minimize2 className="w-3 h-3" />
                                <span>نمایش قطعه فرزند</span>
                              </>
                            ) : (
                              <>
                                <Maximize2 className="w-3 h-3" />
                                <span>گسترش به متن کامل والد</span>
                              </>
                            )}
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab 3: System Performance Metrics */}
      {activeTab === 'metrics' && (
        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          {/* Header & Time Window Selector */}
          <div className="flex items-center justify-between">
            <span className="font-bold text-slate-800 text-xs">
              آمار تجمیعی سیستم (Telemetry Summary)
            </span>
            <div className="flex items-center gap-1">
              <select
                value={timeWindowHours ?? 'all'}
                onChange={(e) => {
                  const val = e.target.value === 'all' ? undefined : parseInt(e.target.value, 10);
                  setTimeWindowHours(val);
                  onRefreshMetrics(val);
                }}
                className="bg-slate-100 border border-slate-200 rounded text-[11px] p-1 text-slate-700 font-medium"
              >
                <option value="1">۱ ساعت اخیر</option>
                <option value="24">۲۴ ساعت اخیر</option>
                <option value="168">۷ روز اخیر</option>
                <option value="all">تمام دوران</option>
              </select>
            </div>
          </div>

          {/* Low Sample Warning Banner (< 20 queries) */}
          {metrics && metrics.total_queries < 20 && (
            <div className="p-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 text-[11px] flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-amber-600 shrink-0" />
              <span>
                تعداد نمونه‌های ثبت‌شده ({metrics.total_queries} پرسش) کمتر از ۲۰ است؛ آماره‌های درصدی ممکن است واریانس بالایی داشته باشند.
              </span>
            </div>
          )}

          {metricsLoading && (
            <div className="text-center py-6 text-slate-400 text-xs">
              در حال دریافت شاخص‌های عملکرد...
            </div>
          )}

          {metrics && (
            <>
              {/* Stat Cards Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
                {/* Total Queries */}
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 flex flex-col">
                  <span className="text-[11px] text-slate-500 font-medium">کل پرسش‌ها</span>
                  <span className="text-lg font-bold font-mono-numbers text-slate-900 mt-1">
                    {metrics.total_queries}
                  </span>
                </div>

                {/* Cache Hit Ratio */}
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 flex flex-col">
                  <span className="text-[11px] text-slate-500 font-medium">نرخ اصابت به کش</span>
                  <span className="text-lg font-bold font-mono-numbers text-emerald-600 mt-1">
                    {Math.round(metrics.cache_hit_ratio * 1000) / 10}%
                  </span>
                </div>

                {/* Abstention Rate */}
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 flex flex-col">
                  <span className="text-[11px] text-slate-500 font-medium">نرخ امتناع (Abstain)</span>
                  <span className="text-lg font-bold font-mono-numbers text-slate-700 mt-1">
                    {Math.round(metrics.abstention_rate * 1000) / 10}%
                  </span>
                </div>

                {/* P50 Latency */}
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 flex flex-col">
                  <span className="text-[11px] text-slate-500 font-medium">تأخیر P50</span>
                  <span className="text-base font-bold font-mono-numbers text-teal-700 mt-1">
                    {Math.round((metrics.latencies?.p50_ms || 0) * 10) / 10} ms
                  </span>
                </div>

                {/* P95 Latency */}
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 flex flex-col">
                  <span className="text-[11px] text-slate-500 font-medium">تأخیر P95</span>
                  <span className="text-base font-bold font-mono-numbers text-teal-700 mt-1">
                    {Math.round((metrics.latencies?.p95_ms || 0) * 10) / 10} ms
                  </span>
                </div>

                {/* P99 Latency */}
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 flex flex-col">
                  <span className="text-[11px] text-slate-500 font-medium">تأخیر P99</span>
                  <span className="text-base font-bold font-mono-numbers text-teal-700 mt-1">
                    {Math.round((metrics.latencies?.p99_ms || 0) * 10) / 10} ms
                  </span>
                </div>

                {/* Cost USD */}
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 flex flex-col">
                  <span className="text-[11px] text-slate-500 font-medium">هزینه کل API</span>
                  <span className="text-base font-bold font-mono-numbers text-slate-900 mt-1">
                    ${metrics.total_cost_usd}
                  </span>
                </div>

                {/* Cost per 1K Queries */}
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 flex flex-col col-span-2 sm:col-span-1">
                  <span className="text-[11px] text-slate-500 font-medium">هزینه هر ۱۰۰۰ کوئری</span>
                  <span className="text-base font-bold font-mono-numbers text-slate-900 mt-1">
                    ${metrics.avg_cost_per_1k_usd}
                  </span>
                </div>
              </div>

              {/* Stage Latency Breakdown Chart */}
              <div className="p-4 rounded-xl border border-slate-200 bg-white flex flex-col gap-3 shadow-2xs">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-slate-800 text-xs">
                    میانگین تأخیر مراحل جستجو (غیرهستی / غیرکش)
                  </span>
                  <span className="text-[10px] text-slate-400">میلی‌ثانیه (ms)</span>
                </div>

                <div className="space-y-2.5 text-[11px] font-mono-numbers">
                  {[
                    { label: 'بازنویسی (Query Rewrite)', val: metrics.stage_latencies_avg?.rewrite_ms || 0, color: 'bg-indigo-500' },
                    { label: 'جستجوی برداری (Dense)', val: metrics.stage_latencies_avg?.dense_search_ms || 0, color: 'bg-teal-500' },
                    { label: 'جستجوی متنی (BM25)', val: metrics.stage_latencies_avg?.sparse_search_ms || 0, color: 'bg-cyan-500' },
                    { label: 'رتبه‌بندی مجدد (Rerank)', val: metrics.stage_latencies_avg?.rerank_ms || 0, color: 'bg-emerald-500' },
                    { label: 'زمان تا اولین توکن (TTFT)', val: metrics.stage_latencies_avg?.llm_ttft_ms || 0, color: 'bg-amber-500' },
                  ].map((stage, idx) => {
                    const maxVal = Math.max(
                      1,
                      metrics.stage_latencies_avg?.rewrite_ms || 0,
                      metrics.stage_latencies_avg?.dense_search_ms || 0,
                      metrics.stage_latencies_avg?.sparse_search_ms || 0,
                      metrics.stage_latencies_avg?.rerank_ms || 0,
                      metrics.stage_latencies_avg?.llm_ttft_ms || 0
                    );
                    const pct = Math.min(100, Math.max(5, (stage.val / maxVal) * 100));

                    return (
                      <div key={idx} className="flex flex-col gap-1">
                        <div className="flex items-center justify-between">
                          <span className="text-slate-600 font-sans">{stage.label}:</span>
                          <span className="font-semibold text-slate-900">{Math.round(stage.val * 10) / 10} ms</span>
                        </div>
                        <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                          <div
                            style={{ width: `${pct}%` }}
                            className={`h-full rounded-full ${stage.color} transition-all duration-500`}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};
