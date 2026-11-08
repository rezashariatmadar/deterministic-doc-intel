import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Square,
  FileText,
  Database,
  Globe,
  ShieldAlert,
  HelpCircle,
  Zap,
  Clock,
  Gauge,
  Copy,
  Check,
} from 'lucide-react';
import { ChatMessage, Citation, CitationSourceType } from '../types';

interface AnswerCanvasProps {
  messages: ChatMessage[];
  inputValue: string;
  onChangeInput: (val: string) => void;
  onSendMessage: (text: string) => void;
  onStopStreaming: () => void;
  isStreaming: boolean;
  onSelectCitation: (sourceId: number) => void;
}

export const AnswerCanvas: React.FC<AnswerCanvasProps> = ({
  messages,
  inputValue,
  onChangeInput,
  onSendMessage,
  onStopStreaming,
  isStreaming,
  onSelectCitation,
}) => {
  const [hoveredCitation, setHoveredCitation] = useState<{
    citation: Citation;
    x: number;
    y: number;
  } | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    if (typeof messagesEndRef.current?.scrollIntoView === 'function') {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (inputValue.trim().length >= 3 && !isStreaming) {
        onSendMessage(inputValue);
      }
    }
  };

  const copyText = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const getSourceIcon = (type?: CitationSourceType) => {
    switch (type) {
      case 'sql_table':
        return <Database className="w-3 h-3 text-indigo-600 inline me-0.5" />;
      case 'external_web':
        return <Globe className="w-3 h-3 text-emerald-600 inline me-0.5" />;
      case 'document':
      default:
        return <FileText className="w-3 h-3 text-teal-600 inline me-0.5" />;
    }
  };

  // Helper to render text with clickable citation chips [1] and [unverified: 1]
  const renderFormattedAnswer = (text: string, citations: Citation[] = []) => {
    // Regex matching [unverified: N] or [N]
    const parts = text.split(/(\[unverified:\s*\d+\]|\[\d+\])/g);

    return parts.map((part, index) => {
      // Unverified citation struck-through
      const unverifiedMatch = part.match(/^\[unverified:\s*(\d+)\]$/);
      if (unverifiedMatch) {
        const id = unverifiedMatch[1];
        return (
          <span
            key={index}
            className="inline-flex items-center text-[11px] font-mono-numbers px-1.5 py-0.5 mx-0.5 rounded bg-rose-50 text-rose-600 line-through border border-rose-200 select-none"
            title={`شاهد ادعاشده [${id}] در اسناد تایید نشد`}
          >
            [unverified: {id}]
          </span>
        );
      }

      // Verified citation chip
      const verifiedMatch = part.match(/^\[(\d+)\]$/);
      if (verifiedMatch) {
        const id = parseInt(verifiedMatch[1], 10);
        const citation = citations.find((c) => c.source_id === id);

        return (
          <button
            key={index}
            type="button"
            onClick={() => onSelectCitation(id)}
            onMouseEnter={(e) => {
              if (citation) {
                const rect = e.currentTarget.getBoundingClientRect();
                setHoveredCitation({
                  citation,
                  x: rect.left + rect.width / 2,
                  y: rect.top,
                });
              }
            }}
            onMouseLeave={() => setHoveredCitation(null)}
            className="inline-flex items-center gap-0.5 px-1.5 py-0.5 mx-0.5 text-[11px] font-mono-numbers font-semibold rounded bg-teal-50 text-teal-700 hover:bg-teal-100 border border-teal-200 hover:border-teal-400 transition-colors shadow-2xs select-none"
          >
            {getSourceIcon(citation?.source_type)}
            <span>[{id}]</span>
          </button>
        );
      }

      return <span key={index}>{part}</span>;
    });
  };

  return (
    <div className="flex flex-col h-full bg-slate-50 relative">
      {/* Messages Thread Container */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-8 text-slate-400 max-w-md mx-auto">
            <div className="w-12 h-12 rounded-2xl bg-slate-200/60 flex items-center justify-center text-slate-500 mb-3">
              <Zap className="w-6 h-6 text-teal-600" />
            </div>
            <h3 className="font-bold text-slate-700 text-sm mb-1">
              پرسش خود را مطرح کنید یا یکی از گزینه‌های نمونه را انتخاب نمایید
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              این سامانه پاسخ‌ها را مستقیماً از صورت‌های مالی و اسناد سازمانی استخراج کرده و هر گزاره را به منبع دقیق ارجاع می‌دهد.
            </p>
          </div>
        ) : (
          messages.map((msg) => {
            if (msg.role === 'user') {
              return (
                <div key={msg.id} className="flex flex-col items-start max-w-2xl">
                  <div className="text-[11px] font-medium text-slate-500 mb-1 ps-1">
                    شما
                  </div>
                  <div className="bg-white border border-slate-200 rounded-2xl p-3.5 shadow-xs text-slate-900 dir-ltr text-start font-sans text-sm leading-relaxed">
                    {msg.content}
                  </div>
                </div>
              );
            }

            // Assistant Card
            const hasRewritten =
              msg.rewrittenQuery &&
              msg.originalQuery &&
              msg.rewrittenQuery.trim() !== msg.originalQuery.trim();

            const isBlocked = Boolean(msg.blockedPattern);
            const isAbstain = msg.isAbstain || (msg.response && msg.response.is_answerable === false);

            return (
              <div key={msg.id} className="flex flex-col items-start w-full max-w-3xl">
                <div className="text-[11px] font-medium text-teal-700 mb-1 ps-1 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-teal-600" />
                  <span>پاسخ سامانه مستند</span>
                </div>

                <div className="w-full bg-white border border-slate-200 rounded-2xl shadow-xs overflow-hidden">
                  {/* Rewritten Query Badge */}
                  {hasRewritten && (
                    <div className="bg-amber-50/70 border-b border-amber-200/60 px-4 py-2 flex items-center gap-2 text-xs text-amber-900">
                      <HelpCircle className="w-3.5 h-3.5 text-amber-700 shrink-0" />
                      <span className="font-semibold shrink-0">پرسش بازنویسی‌شده توسط حافظه:</span>
                      <span className="dir-ltr text-start font-mono-numbers text-[11px] text-amber-800 bg-amber-100/50 px-2 py-0.5 rounded truncate">
                        {msg.rewrittenQuery}
                      </span>
                    </div>
                  )}

                  {/* Body Content */}
                  <div className="p-4 md:p-5">
                    {/* Blocked State (Prompt Injection) */}
                    {isBlocked ? (
                      <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 flex flex-col gap-2">
                        <div className="flex items-center gap-2 text-rose-800 font-bold text-sm">
                          <ShieldAlert className="w-4 h-4 text-rose-600" />
                          <span>تزریق پرامپت شناسایی و توسط گاردریل مسدود شد</span>
                        </div>
                        <p className="text-xs text-rose-700 leading-relaxed">
                          ورودی کاربر با الگوی تزریق دستورات مغایرت داشته و پردازش آن جهت حفاظت از حریم سامانه متوقف گردید. هیچ کوئری به پایگاه داده یا مدل زبانی ارسال نشد.
                        </p>
                        {msg.blockedPattern && (
                          <div className="text-[11px] font-mono-numbers text-rose-800 bg-white/70 p-2 rounded border border-rose-200/60 dir-ltr text-start">
                            الگوی منطبق: <span className="font-semibold">{msg.blockedPattern}</span>
                          </div>
                        )}
                      </div>
                    ) : isAbstain ? (
                      /* Abstain State */
                      <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex flex-col gap-2 text-slate-700">
                        <div className="flex items-center gap-2 font-semibold text-xs text-slate-800">
                          <HelpCircle className="w-4 h-4 text-slate-500" />
                          <span>عدم کفایت شواهد برای پاسخ مستند (Abstain)</span>
                        </div>
                        <p className="text-xs text-slate-600 leading-relaxed">
                          اطلاعات موجود در اسناد نمایه شده برای پاسخ معتبر و اثبات‌شده به این پرسش کافی نیست. سامانه طبق اصل پرهیز از توهم، از تولید پاسخ فرضی خودداری کرد.
                        </p>
                      </div>
                    ) : (
                      /* Normal Answer Content */
                      <div className="dir-ltr text-start text-sm leading-relaxed text-slate-800 font-sans whitespace-pre-wrap">
                        {renderFormattedAnswer(msg.content, msg.response?.citations)}
                        {msg.isLoading && (
                          <span className="inline-block w-2 h-4 ms-1 bg-teal-600 animate-pulse align-middle" />
                        )}
                      </div>
                    )}

                    {/* Copy button */}
                    <div className="mt-3 flex items-center justify-end">
                      <button
                        onClick={() => copyText(msg.content, msg.id)}
                        className="text-slate-400 hover:text-slate-700 p-1 rounded hover:bg-slate-100 transition-colors flex items-center gap-1 text-[11px]"
                        title="کپی متن پاسخ"
                      >
                        {copiedId === msg.id ? (
                          <>
                            <Check className="w-3 h-3 text-emerald-600" />
                            <span className="text-emerald-700 font-medium">کپی شد</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3 h-3" />
                            <span>کپی</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Telemetry Footer */}
                  {msg.response && (
                    <div className="bg-slate-50 border-t border-slate-100 px-4 py-2.5 flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-600 font-mono-numbers">
                      {/* Left: Cache & Route */}
                      <div className="flex items-center gap-2">
                        {/* Cache Hit Badge */}
                        <span
                          className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-sans font-bold border ${
                            msg.cacheHit
                              ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                              : 'bg-slate-100 text-slate-600 border-slate-200'
                          }`}
                        >
                          {msg.cacheHit ? 'CACHE HIT' : 'CACHE MISS'}
                        </span>

                        {/* Route Taken */}
                        {msg.response.route_taken && (
                          <span className="px-2 py-0.5 rounded bg-slate-200/70 text-slate-700 text-[10px]">
                            {msg.response.route_taken}
                          </span>
                        )}
                      </div>

                      {/* Right: Metrics */}
                      <div className="flex items-center gap-3">
                        {/* Latency */}
                        <span title="زمان کل پاسخگویی" className="flex items-center gap-1">
                          <Clock className="w-3 h-3 text-slate-400" />
                          <span>{msg.response.latency_ms} ms</span>
                        </span>

                        {/* TTFT */}
                        {msg.ttftMs != null && (
                          <span title="زمان تا اولین توکن (Time to First Token)" className="flex items-center gap-1">
                            <Zap className="w-3 h-3 text-amber-500" />
                            <span>TTFT: {msg.ttftMs} ms</span>
                          </span>
                        )}

                        {/* Confidence Score */}
                        <span title="ضریب اطمینان استخراج" className="flex items-center gap-1">
                          <Gauge className="w-3 h-3 text-slate-400" />
                          <span>{Math.round((msg.response.confidence_score || 0) * 100)}%</span>
                        </span>

                        {/* CRAG Retries */}
                        {Boolean(msg.response.crag_retries || msg.response.crag_retry_count) && (
                          <span className="text-amber-700 bg-amber-50 px-1.5 py-0.2 rounded border border-amber-200 text-[10px]">
                            {msg.response.crag_retries || msg.response.crag_retry_count} بازنویسی
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Floating Citation Popover */}
      {hoveredCitation && (
        <div
          style={{
            position: 'fixed',
            top: `${hoveredCitation.y - 12}px`,
            left: `${hoveredCitation.x}px`,
            transform: 'translate(-50%, -100%)',
          }}
          className="z-50 max-w-sm w-80 bg-white border border-slate-200 rounded-xl shadow-xl p-3 text-xs flex flex-col gap-2 pointer-events-auto"
        >
          <div className="flex items-center justify-between border-b border-slate-100 pb-1.5">
            <span className="font-bold text-teal-700 font-mono-numbers flex items-center gap-1">
              {getSourceIcon(hoveredCitation.citation.source_type)}
              <span>شاهد [{hoveredCitation.citation.source_id}]</span>
            </span>
            <span className="text-[10px] text-slate-400 font-mono-numbers">
              ص {hoveredCitation.citation.page_start}
            </span>
          </div>
          <div className="font-semibold text-slate-800 text-[11px] truncate dir-ltr text-start">
            {hoveredCitation.citation.document_title}
          </div>
          {hoveredCitation.citation.section_path && (
            <div className="text-[10px] text-slate-500 truncate dir-ltr text-start">
              بخش: {hoveredCitation.citation.section_path}
            </div>
          )}
          {hoveredCitation.citation.supporting_quote && (
            <div className="bg-slate-50 p-2 rounded border border-slate-100 text-[11px] text-slate-700 italic dir-ltr text-start line-clamp-3 leading-relaxed">
              &ldquo;{hoveredCitation.citation.supporting_quote}&rdquo;
            </div>
          )}
          <button
            type="button"
            onClick={() => {
              onSelectCitation(hoveredCitation.citation.source_id);
              setHoveredCitation(null);
            }}
            className="w-full py-1 text-center font-medium text-[11px] bg-teal-50 text-teal-700 hover:bg-teal-100 rounded transition-colors"
          >
            مشاهده کامل قطعه در شواهد
          </button>
        </div>
      )}

      {/* Bottom Input Area */}
      <div className="border-t border-slate-200 bg-white p-3 md:p-4 shrink-0">
        <div className="max-w-3xl mx-auto flex items-end gap-2 bg-slate-50 border border-slate-300 rounded-xl p-2 focus-within:ring-2 focus-within:ring-teal-500 focus-within:border-transparent transition-all shadow-xs">
          <textarea
            ref={textareaRef}
            rows={2}
            value={inputValue}
            onChange={(e) => onChangeInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="پرسش خود را به زبان انگلیسی وارد کنید (ارسال با Enter)..."
            className="flex-1 bg-transparent resize-none border-none outline-none text-slate-800 placeholder-slate-400 text-xs md:text-sm dir-ltr text-start leading-relaxed min-h-[44px]"
          />

          <div className="flex items-center gap-1.5 shrink-0">
            {isStreaming ? (
              <button
                type="button"
                onClick={onStopStreaming}
                className="px-3 py-2 bg-rose-600 hover:bg-rose-700 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-xs"
                title="توقف تولید پاسخ"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
                <span>توقف</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={() => {
                  if (inputValue.trim().length >= 3) {
                    onSendMessage(inputValue);
                  }
                }}
                disabled={inputValue.trim().length < 3}
                className="px-3 py-2 bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-xs"
                title="ارسال پرسش"
              >
                <span>ارسال</span>
                <Send className="w-3.5 h-3.5 transform -scale-x-100" />
              </button>
            )}
          </div>
        </div>

        <div className="max-w-3xl mx-auto flex items-center justify-between text-[11px] text-slate-400 mt-2 px-1">
          <span>برای خط جدید از Shift + Enter استفاده کنید.</span>
          <span className="font-mono-numbers">حداقل ۳ کاراکتر</span>
        </div>
      </div>
    </div>
  );
};
