import React, { useState, useRef } from 'react';
import {
  Sparkles,
  Sliders,
  RotateCcw,
  Upload,
  FileText,
  Trash2,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  AlertCircle,
  Clock,
  Layers,
  ArrowRight,
} from 'lucide-react';
import { DocumentItem, IntentRoute, Preset, QueryRequest } from '../types';
import { PRESETS } from '../presets';

interface ControlsPanelProps {
  requestOptions: QueryRequest;
  onChangeOptions: (updated: Partial<QueryRequest>) => void;
  onSelectPreset: (preset: Preset) => void;
  documents: DocumentItem[];
  docsLoading: boolean;
  onUploadFile: (file: File) => Promise<void>;
  onDeleteDocument: (doc: DocumentItem) => Promise<void>;
  onResetDefaults: () => void;
}

export const ControlsPanel: React.FC<ControlsPanelProps> = ({
  requestOptions,
  onChangeOptions,
  onSelectPreset,
  documents,
  docsLoading,
  onUploadFile,
  onDeleteDocument,
  onResetDefaults,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [docToDelete, setDocToDelete] = useState<DocumentItem | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Handle route override changing: locks enable_agent on if non-Auto
  const handleRouteOverrideChange = (route: string) => {
    if (!route || route === 'AUTO') {
      onChangeOptions({ route_override: null });
    } else {
      onChangeOptions({
        route_override: route as IntentRoute,
        enable_agent: true,
      });
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadError(null);
    setIsUploading(true);
    try {
      await onUploadFile(file);
    } catch (err: unknown) {
      setUploadError((err as Error).message || 'خطا در بارگذاری فایل');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const confirmDelete = async () => {
    if (!docToDelete) return;
    setIsDeleting(true);
    try {
      await onDeleteDocument(docToDelete);
      setDocToDelete(null);
    } catch (err: unknown) {
      alert((err as Error).message || 'خطا در حذف سند');
    } finally {
      setIsDeleting(false);
    }
  };

  const isAgentActive = Boolean(requestOptions.enable_agent || requestOptions.route_override);

  return (
    <div className="flex flex-col h-full overflow-y-auto divide-y divide-slate-200 text-xs bg-white select-none">
      {/* 1. Presets Section */}
      <div className="p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 font-bold text-slate-800 text-sm">
            <Sparkles className="w-4 h-4 text-teal-600" />
            <span>پرسش‌های نمونه (Presets)</span>
          </div>
          <span className="text-[11px] text-slate-400">آزمون یک‌کلیکه سناریوها</span>
        </div>

        <div className="flex flex-col gap-2">
          {PRESETS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => onSelectPreset(preset)}
              className="group flex flex-col text-start p-2.5 rounded-lg border border-slate-200 hover:border-teal-500 hover:bg-teal-50/40 transition-all shadow-xs"
            >
              <div className="flex items-center justify-between w-full">
                <span className="font-semibold text-slate-800 group-hover:text-teal-700">
                  {preset.label}
                </span>
                <ArrowRight className="w-3.5 h-3.5 text-slate-400 group-hover:text-teal-600 group-hover:-translate-x-1 transition-transform" />
              </div>
              <p className="text-[11px] text-slate-500 mt-1 line-clamp-1 leading-relaxed">
                {preset.description}
              </p>
              <div className="dir-ltr text-[11px] text-slate-600 font-mono-numbers mt-1.5 p-1 bg-slate-50 rounded border border-slate-100 truncate">
                {preset.query}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 2. Pipeline Switches */}
      <div className="p-4 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 font-bold text-slate-800 text-sm">
            <Sliders className="w-4 h-4 text-teal-600" />
            <span>تنظیمات پایپ‌لاین (Switches)</span>
          </div>
          <button
            onClick={onResetDefaults}
            title="بازنشانی تمام تنظیمات به مقادیر پیش‌فرض"
            className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-800 hover:bg-slate-100 px-2 py-1 rounded transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
            <span>پیش‌فرض</span>
          </button>
        </div>

        {/* Search Mode */}
        <div className="flex flex-col gap-1.5">
          <label className="text-slate-700 font-medium">حالت جستجو (Search Mode):</label>
          <div className="grid grid-cols-3 gap-1 p-0.5 bg-slate-100 rounded-lg border border-slate-200">
            {(['hybrid', 'dense', 'sparse'] as const).map((mode) => {
              const active = (requestOptions.search_mode || 'hybrid') === mode;
              const labels = {
                hybrid: 'ترکیبی (RRF)',
                dense: 'برداری (Dense)',
                sparse: 'متنی (BM25)',
              };
              return (
                <button
                  key={mode}
                  type="button"
                  onClick={() => onChangeOptions({ search_mode: mode })}
                  className={`py-1.5 text-center font-medium rounded-md transition-all ${
                    active
                      ? 'bg-white text-teal-700 shadow-xs border border-slate-200 font-semibold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {labels[mode]}
                </button>
              );
            })}
          </div>
        </div>

        {/* Reranker */}
        <div className="flex flex-col gap-1.5">
          <label className="text-slate-700 font-medium">رتبه‌بندی مجدد عصبی (Cross-Encoder Reranker):</label>
          <div className="grid grid-cols-3 gap-1 p-0.5 bg-slate-100 rounded-lg border border-slate-200">
            {[
              { val: null, label: 'پیش‌فرض سرور' },
              { val: true, label: 'فعال (On)' },
              { val: false, label: 'غیرفعال (Off)' },
            ].map((item) => {
              const active = requestOptions.enable_reranker === item.val;
              return (
                <button
                  key={String(item.val)}
                  type="button"
                  onClick={() => onChangeOptions({ enable_reranker: item.val })}
                  className={`py-1.5 text-center font-medium rounded-md transition-all ${
                    active
                      ? 'bg-white text-teal-700 shadow-xs border border-slate-200 font-semibold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Enable Agentic State Machine Toggle */}
        <div className="flex items-center justify-between py-1">
          <div className="flex flex-col">
            <span className="font-medium text-slate-800">حالت چندعاملی (Agentic CRAG Loop)</span>
            <span className="text-[11px] text-slate-500">مسیریابی هوشمند، تصحیح خودکار و ابزارها</span>
          </div>
          <button
            type="button"
            onClick={() =>
              onChangeOptions({
                enable_agent: !requestOptions.enable_agent,
                route_override: requestOptions.enable_agent ? null : requestOptions.route_override,
              })
            }
            className={`w-10 h-6 flex items-center rounded-full p-1 transition-colors ${
              requestOptions.enable_agent ? 'bg-teal-600' : 'bg-slate-300'
            }`}
          >
            <div
              className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${
                requestOptions.enable_agent ? '-translate-x-4' : 'translate-x-0'
              }`}
            />
          </button>
        </div>

        {/* Route Override Select (Locks agent on) */}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between">
            <label className="text-slate-700 font-medium">مسیریابی دستی (Route Override):</label>
            {requestOptions.route_override && (
              <span className="text-[10px] text-amber-600 bg-amber-50 px-1.5 py-0.5 rounded border border-amber-200">
                عامل قفل شد
              </span>
            )}
          </div>
          <select
            value={requestOptions.route_override || 'AUTO'}
            onChange={(e) => handleRouteOverrideChange(e.target.value)}
            className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs font-mono-numbers text-slate-800 focus:ring-1 focus:ring-teal-500 focus:outline-none"
          >
            <option value="AUTO">خودکار (Auto - بر اساس دسته‌بندی هوشمند)</option>
            <option value="FAST_TRACK_HYBRID_RAG">FAST_TRACK_HYBRID_RAG (تک‌مرحله‌ای استاندارد)</option>
            <option value="MULTI_HOP_AGENTIC_RAG">MULTI_HOP_AGENTIC_RAG (تجزیه پرسش چندمرحله‌ای)</option>
            <option value="STRUCTURED_SQL_ANALYTICS">STRUCTURED_SQL_ANALYTICS (تحلیل عددی پایگاه‌داده)</option>
            <option value="HYBRID_SQL_AND_RAG">HYBRID_SQL_AND_RAG (ترکیب SQL و مستندات)</option>
            <option value="CHITCHAT_OR_OUT_OF_DOMAIN">CHITCHAT_OR_OUT_OF_DOMAIN (مکالمه عمومی / امتناع)</option>
          </select>
        </div>

        {/* External Web Fallback Toggle */}
        <div className="flex items-center justify-between py-1">
          <div className="flex flex-col">
            <span className="font-medium text-slate-800">جستجوی خارجی (External Fallback)</span>
            <span className="text-[11px] text-slate-500">استفاده از وب عمومی در صورت کمبود شواهد</span>
          </div>
          <button
            type="button"
            onClick={() =>
              onChangeOptions({
                allow_external_fallback: !requestOptions.allow_external_fallback,
              })
            }
            className={`w-10 h-6 flex items-center rounded-full p-1 transition-colors ${
              requestOptions.allow_external_fallback ? 'bg-teal-600' : 'bg-slate-300'
            }`}
          >
            <div
              className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${
                requestOptions.allow_external_fallback ? '-translate-x-4' : 'translate-x-0'
              }`}
            />
          </button>
        </div>

        {/* Bypass Cache Toggle */}
        <div className="flex items-center justify-between py-1">
          <div className="flex flex-col">
            <span className="font-medium text-slate-800">نادیده‌گیری حافظه موقت (Bypass Cache)</span>
            <span className="text-[11px] text-slate-500">اجرای اجباری پردازش بدون استفاده از کش</span>
          </div>
          <button
            type="button"
            onClick={() =>
              onChangeOptions({
                bypass_cache: !requestOptions.bypass_cache,
              })
            }
            className={`w-10 h-6 flex items-center rounded-full p-1 transition-colors ${
              requestOptions.bypass_cache ? 'bg-amber-600' : 'bg-slate-300'
            }`}
          >
            <div
              className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${
                requestOptions.bypass_cache ? '-translate-x-4' : 'translate-x-0'
              }`}
            />
          </button>
        </div>

        {/* Collapsible Advanced Settings */}
        <div className="border-t border-slate-100 pt-2">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center justify-between w-full text-slate-600 hover:text-slate-900 py-1 font-medium"
          >
            <span className="flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-slate-400" />
              <span>تنظیمات پیشرفته (Advanced Tuning)</span>
            </span>
            {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>

          {showAdvanced && (
            <div className="flex flex-col gap-3 mt-3 ps-2 pe-1 border-s-2 border-slate-200">
              {/* top_k */}
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-slate-600">تعداد قطعات بازیافتی (top_k):</span>
                  <span className="font-mono-numbers font-semibold text-teal-700">
                    {requestOptions.top_k || 5}
                  </span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="20"
                  value={requestOptions.top_k || 5}
                  onChange={(e) => onChangeOptions({ top_k: parseInt(e.target.value, 10) })}
                  className="w-full accent-teal-600 cursor-pointer"
                />
              </div>

              {/* RRF Dense / Sparse Weights */}
              <div className="grid grid-cols-2 gap-2">
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between text-[11px]">
                    <span className="text-slate-600">وزن برداری:</span>
                    <span className="font-mono-numbers text-slate-800">
                      {requestOptions.weight_dense ?? 1.0}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={requestOptions.weight_dense ?? 1.0}
                    onChange={(e) =>
                      onChangeOptions({ weight_dense: parseFloat(e.target.value) })
                    }
                    className="w-full accent-teal-600 cursor-pointer"
                  />
                </div>

                <div className="flex flex-col gap-1">
                  <div className="flex justify-between text-[11px]">
                    <span className="text-slate-600">وزن متنی:</span>
                    <span className="font-mono-numbers text-slate-800">
                      {requestOptions.weight_sparse ?? 1.0}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={requestOptions.weight_sparse ?? 1.0}
                    onChange={(e) =>
                      onChangeOptions({ weight_sparse: parseFloat(e.target.value) })
                    }
                    className="w-full accent-teal-600 cursor-pointer"
                  />
                </div>
              </div>

              {/* Max CRAG Retries */}
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-slate-600">سقف تلاش بازنویسی CRAG:</span>
                  <span className="font-mono-numbers font-semibold text-slate-800">
                    {requestOptions.max_crag_retries ?? 2}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="5"
                  value={requestOptions.max_crag_retries ?? 2}
                  onChange={(e) =>
                    onChangeOptions({ max_crag_retries: parseInt(e.target.value, 10) })
                  }
                  className="w-full accent-teal-600 cursor-pointer"
                />
              </div>

              {/* Resolve Parents Hierarchical Toggle */}
              <div className="flex items-center justify-between py-1">
                <span className="text-slate-600">ارتقا به قطعات والد (Resolve Parents):</span>
                <input
                  type="checkbox"
                  checked={requestOptions.resolve_parents !== false}
                  onChange={(e) => onChangeOptions({ resolve_parents: e.target.checked })}
                  className="accent-teal-600 rounded"
                />
              </div>

              {/* HyDE & Query Rewrite (Disabled if Agent is active) */}
              <div
                className={`flex flex-col gap-2 pt-1 border-t border-slate-100 ${
                  isAgentActive ? 'opacity-50 pointer-events-none' : ''
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-slate-600">بازنویسی چندپرسشی (Query Rewrite):</span>
                  <input
                    type="checkbox"
                    checked={Boolean(requestOptions.enable_query_rewrite)}
                    onChange={(e) => onChangeOptions({ enable_query_rewrite: e.target.checked })}
                    className="accent-teal-600 rounded"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-600">سند فرضی (HyDE Expansion):</span>
                  <input
                    type="checkbox"
                    checked={Boolean(requestOptions.enable_hyde)}
                    onChange={(e) => onChangeOptions({ enable_hyde: e.target.checked })}
                    className="accent-teal-600 rounded"
                  />
                </div>
                {isAgentActive && (
                  <span className="text-[10px] text-amber-700">
                    در حالت عامل، فرمول‌بندی و بازنویسی پرسش توسط نودهای اختصاصی مدیریت می‌شود.
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 3. Documents List & Upload */}
      <div className="p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 font-bold text-slate-800 text-sm">
            <FileText className="w-4 h-4 text-teal-600" />
            <span>پیکره دانش اسناد ({documents.length})</span>
          </div>
          {docsLoading && <Clock className="w-3.5 h-3.5 text-slate-400 animate-spin" />}
        </div>

        {/* Upload Button */}
        <div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md"
            onChange={handleFileChange}
            className="hidden"
            id="doc-upload-input"
          />
          <label
            htmlFor="doc-upload-input"
            className={`flex items-center justify-center gap-2 w-full p-2.5 rounded-lg border-2 border-dashed border-slate-300 hover:border-teal-500 hover:bg-teal-50/30 text-slate-700 font-medium cursor-pointer transition-all ${
              isUploading ? 'opacity-50 pointer-events-none' : ''
            }`}
          >
            <Upload className="w-4 h-4 text-teal-600" />
            <span>{isUploading ? 'در حال بارگذاری سند...' : 'بارگذاری سند جدید (PDF / Markdown)'}</span>
          </label>
          {uploadError && (
            <p className="text-[11px] text-rose-600 mt-1.5 flex items-center gap-1">
              <AlertCircle className="w-3 h-3 shrink-0" />
              <span>{uploadError}</span>
            </p>
          )}
        </div>

        {/* Document items list */}
        <div className="flex flex-col gap-2 max-h-72 overflow-y-auto pe-1">
          {documents.length === 0 ? (
            <div className="text-center py-6 text-slate-400 bg-slate-50 rounded-lg border border-slate-100">
              هیچ سندی بارگذاری نشده است.
            </div>
          ) : (
            documents.map((doc) => {
              const isProcessing = doc.status === 'PROCESSING' || doc.status === 'PENDING';
              const isFailed = doc.status === 'FAILED';
              const isCompleted = doc.status === 'COMPLETED';

              return (
                <div
                  key={doc.id}
                  className="p-2.5 rounded-lg border border-slate-200 bg-white hover:border-slate-300 transition-colors flex flex-col gap-1.5 shadow-2xs"
                >
                  <div className="flex items-start justify-between gap-2">
                    <span
                      className="font-medium text-slate-800 dir-ltr text-start line-clamp-1 flex-1 text-[11px]"
                      title={doc.title || doc.filename}
                    >
                      {doc.title || doc.filename}
                    </span>
                    <button
                      onClick={() => setDocToDelete(doc)}
                      className="text-slate-400 hover:text-rose-600 p-0.5 rounded transition-colors"
                      title="حذف سند"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-slate-500 font-mono-numbers">
                    <span>
                      {doc.total_pages} ص | {doc.total_chunks} قطعه
                    </span>

                    {/* Status Badge */}
                    <span
                      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-sans ${
                        isCompleted
                          ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                          : isProcessing
                          ? 'bg-amber-50 text-amber-700 border border-amber-200 animate-pulse'
                          : 'bg-rose-50 text-rose-700 border border-rose-200'
                      }`}
                    >
                      {isCompleted && <CheckCircle2 className="w-3 h-3" />}
                      {isProcessing && <Clock className="w-3 h-3 animate-spin" />}
                      {isFailed && <AlertCircle className="w-3 h-3" />}
                      <span>
                        {doc.processing_stage || (isCompleted ? 'آماده' : isProcessing ? 'پردازش' : 'خطا')}
                      </span>
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Delete Document Confirmation Modal */}
      {docToDelete && (
        <div className="fixed inset-0 z-50 bg-slate-900/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-xl shadow-xl max-w-sm w-full p-4 flex flex-col gap-3 border border-slate-200">
            <div className="flex items-center gap-2 text-rose-600 font-bold text-sm">
              <AlertCircle className="w-5 h-5 shrink-0" />
              <span>تایید حذف سند</span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              آیا از حذف کامل سند{' '}
              <strong className="text-slate-900 dir-ltr inline-block font-mono-numbers">
                «{docToDelete.title || docToDelete.filename}»
              </strong>{' '}
              و تمام قطعات برداری آن از پایگاه داده اطمینان دارید؟
            </p>
            <div className="flex items-center justify-end gap-2 mt-2">
              <button
                type="button"
                onClick={() => setDocToDelete(null)}
                disabled={isDeleting}
                className="px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-100 rounded-lg transition-colors font-medium"
              >
                انصراف
              </button>
              <button
                type="button"
                onClick={confirmDelete}
                disabled={isDeleting}
                className="px-3 py-1.5 text-xs bg-rose-600 text-white hover:bg-rose-700 rounded-lg font-medium transition-colors flex items-center gap-1.5"
              >
                {isDeleting ? <Clock className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
                <span>حذف دائمی سند</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
