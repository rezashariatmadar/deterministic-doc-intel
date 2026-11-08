import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ControlsPanel } from '../components/ControlsPanel';
import { DocumentItem, QueryRequest } from '../types';

describe('ControlsPanel Component', () => {
  const defaultOptions: QueryRequest = {
    query: '',
    search_mode: 'hybrid',
    enable_reranker: null,
    enable_agent: false,
    route_override: null,
    allow_external_fallback: false,
    bypass_cache: false,
  };

  const mockDocs: DocumentItem[] = [
    {
      id: 'doc-1',
      filename: 'nvidia_10k.pdf',
      file_hash: 'hash123',
      title: 'NVIDIA 2024 Form 10-K',
      total_pages: 96,
      total_chunks: 342,
      status: 'COMPLETED',
      processing_stage: 'COMPLETED',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    },
  ];

  it('renders presets and handles preset selection', () => {
    const onSelectPreset = vi.fn();
    const onChangeOptions = vi.fn();

    render(
      <ControlsPanel
        requestOptions={defaultOptions}
        onChangeOptions={onChangeOptions}
        onSelectPreset={onSelectPreset}
        documents={mockDocs}
        docsLoading={false}
        onUploadFile={vi.fn()}
        onDeleteDocument={vi.fn()}
        onResetDefaults={vi.fn()}
      />
    );

    expect(screen.getByText('پرسش‌های نمونه (Presets)')).toBeInTheDocument();
    expect(screen.getByText('ارقام جدول مالی')).toBeInTheDocument();

    fireEvent.click(screen.getByText('ارقام جدول مالی'));
    expect(onSelectPreset).toHaveBeenCalledTimes(1);
  });

  it('handles search mode switches', () => {
    const onChangeOptions = vi.fn();

    render(
      <ControlsPanel
        requestOptions={defaultOptions}
        onChangeOptions={onChangeOptions}
        onSelectPreset={vi.fn()}
        documents={mockDocs}
        docsLoading={false}
        onUploadFile={vi.fn()}
        onDeleteDocument={vi.fn()}
        onResetDefaults={vi.fn()}
      />
    );

    fireEvent.click(screen.getByText('برداری (Dense)'));
    expect(onChangeOptions).toHaveBeenCalledWith({ search_mode: 'dense' });
  });

  it('locks agent on when route override is selected', () => {
    const onChangeOptions = vi.fn();

    render(
      <ControlsPanel
        requestOptions={defaultOptions}
        onChangeOptions={onChangeOptions}
        onSelectPreset={vi.fn()}
        documents={mockDocs}
        docsLoading={false}
        onUploadFile={vi.fn()}
        onDeleteDocument={vi.fn()}
        onResetDefaults={vi.fn()}
      />
    );

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'MULTI_HOP_AGENTIC_RAG' } });

    expect(onChangeOptions).toHaveBeenCalledWith({
      route_override: 'MULTI_HOP_AGENTIC_RAG',
      enable_agent: true,
    });
  });

  it('renders document items and delete confirmation modal', () => {
    const onDeleteDoc = vi.fn();

    render(
      <ControlsPanel
        requestOptions={defaultOptions}
        onChangeOptions={vi.fn()}
        onSelectPreset={vi.fn()}
        documents={mockDocs}
        docsLoading={false}
        onUploadFile={vi.fn()}
        onDeleteDocument={onDeleteDoc}
        onResetDefaults={vi.fn()}
      />
    );

    expect(screen.getByText('NVIDIA 2024 Form 10-K')).toBeInTheDocument();
    expect(screen.getByText(/96 ص/)).toBeInTheDocument();

    const deleteBtn = screen.getByTitle('حذف سند');
    fireEvent.click(deleteBtn);

    expect(screen.getByText('تایید حذف سند')).toBeInTheDocument();
  });
});
