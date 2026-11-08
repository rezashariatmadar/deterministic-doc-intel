import React from 'react';
import { Database, RefreshCw, Server, Plus, SlidersHorizontal, Eye } from 'lucide-react';
import { HealthStatus } from '../types';

interface HeaderProps {
  health: HealthStatus | null;
  healthLoading: boolean;
  onNewConversation: () => void;
  onRefreshHealth: () => void;
  onToggleControls?: () => void;
  onToggleInspector?: () => void;
  showInspectorDrawer?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  healthLoading,
  onNewConversation,
  onRefreshHealth,
  onToggleControls,
  onToggleInspector,
}) => {
  const isDbConnected = health?.database === 'connected';
  const isRedisConnected = health?.redis === 'connected';

  return (
    <header className="h-14 border-b border-slate-200 bg-white px-4 flex items-center justify-between select-none shrink-0 sticky top-0 z-30">
      {/* Title & Branding */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-teal-600 text-white flex items-center justify-center font-bold text-sm tracking-tighter">
            RAG
          </div>
          <div className="flex flex-col">
            <span className="font-bold text-slate-800 text-sm leading-tight">
              داشبورد مهندسی رگ
            </span>
            <span className="text-[11px] text-slate-500 leading-tight">
              سامانه جستجوی ترکیبی و عامل‌محور
            </span>
          </div>
        </div>
        <span className="hidden sm:inline-block text-[11px] font-mono-numbers px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
          v1.0.0
        </span>
      </div>

      {/* Health Indicator & Actions */}
      <div className="flex items-center gap-3">
        {/* System Health Status Badges */}
        <div className="hidden md:flex items-center gap-2 text-xs text-slate-600 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200">
          <div className="flex items-center gap-1.5" title="وضعیت ارتباط با پایگاه داده PostgreSQL / pgvector">
            <Database className="w-3.5 h-3.5 text-slate-500" />
            <span className="font-medium">PostgreSQL:</span>
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                isDbConnected ? 'bg-emerald-500' : 'bg-rose-500'
              }`}
            />
            <span className="text-[11px] text-slate-500">
              {isDbConnected ? 'متصل' : 'قطع'}
            </span>
          </div>

          <span className="text-slate-300">|</span>

          <div className="flex items-center gap-1.5" title="وضعیت حافظه موقت معنایی Redis">
            <Server className="w-3.5 h-3.5 text-slate-500" />
            <span className="font-medium">Redis:</span>
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                isRedisConnected ? 'bg-emerald-500' : 'bg-rose-500'
              }`}
            />
            <span className="text-[11px] text-slate-500">
              {isRedisConnected ? 'متصل' : 'قطع'}
            </span>
          </div>

          <button
            onClick={onRefreshHealth}
            disabled={healthLoading}
            title="بروزرسانی وضعیت سرویس‌ها"
            className="ms-1 p-1 text-slate-400 hover:text-slate-700 hover:bg-slate-200/50 rounded transition-colors"
          >
            <RefreshCw className={`w-3 h-3 ${healthLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {/* Mobile controls toggle */}
        {onToggleControls && (
          <button
            onClick={onToggleControls}
            className="lg:hidden p-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg border border-slate-200"
            title="تنظیمات و اسناد"
          >
            <SlidersHorizontal className="w-4 h-4" />
          </button>
        )}

        {/* Mobile / Tablet inspector toggle */}
        {onToggleInspector && (
          <button
            onClick={onToggleInspector}
            className="xl:hidden p-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg border border-slate-200 flex items-center gap-1.5 text-xs font-medium"
            title="پنل بازرسی شواهد و ردیابی"
          >
            <Eye className="w-4 h-4" />
            <span className="hidden sm:inline">بازرسی</span>
          </button>
        )}

        {/* New conversation button */}
        <button
          onClick={onNewConversation}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-slate-900 hover:bg-slate-800 rounded-lg shadow-sm transition-colors focus:ring-2 focus:ring-slate-400 focus:outline-none"
        >
          <Plus className="w-4 h-4" />
          <span>مکالمه جدید</span>
        </button>
      </div>
    </header>
  );
};
