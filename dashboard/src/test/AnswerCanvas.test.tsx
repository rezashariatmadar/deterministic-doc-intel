import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AnswerCanvas } from '../components/AnswerCanvas';
import { ChatMessage } from '../types';

describe('AnswerCanvas Component', () => {
  it('renders citation chips and unverified markers correctly', () => {
    const onSelectCitation = vi.fn();
    const messages: ChatMessage[] = [
      {
        id: 'msg-1',
        role: 'user',
        content: 'What was NVIDIA revenue in 2024?',
      },
      {
        id: 'msg-2',
        role: 'assistant',
        content: 'NVIDIA reported $60,922 million [1] but unverified claim [unverified: 2].',
        response: {
          answer: 'NVIDIA reported $60,922 million [1] but unverified claim [unverified: 2].',
          is_answerable: true,
          confidence_score: 0.95,
          latency_ms: 150.2,
          route_taken: 'FAST_TRACK_HYBRID_RAG',
          citations: [
            {
              source_id: 1,
              document_title: 'NVIDIA 10-K',
              page_start: 12,
              page_end: 12,
              section_path: 'Financials',
              supporting_quote: 'Total revenue was $60,922M.',
              source_type: 'document',
            },
          ],
          sources: [],
          candidate_facts: [],
          execution_trace: [],
        },
        cacheHit: true,
      },
    ];

    render(
      <AnswerCanvas
        messages={messages}
        inputValue=""
        onChangeInput={vi.fn()}
        onSendMessage={vi.fn()}
        onStopStreaming={vi.fn()}
        isStreaming={false}
        onSelectCitation={onSelectCitation}
      />
    );

    // Verify user message
    expect(screen.getByText('What was NVIDIA revenue in 2024?')).toBeInTheDocument();

    // Verify citation chip [1]
    const chip = screen.getByText('[1]');
    expect(chip).toBeInTheDocument();
    fireEvent.click(chip);
    expect(onSelectCitation).toHaveBeenCalledWith(1);

    // Verify unverified marker
    expect(screen.getByText('[unverified: 2]')).toBeInTheDocument();

    // Verify telemetry footer badge CACHE HIT
    expect(screen.getByText('CACHE HIT')).toBeInTheDocument();
    expect(screen.getByText('150.2 ms')).toBeInTheDocument();
  });

  it('renders blocked state when prompt injection detected', () => {
    const messages: ChatMessage[] = [
      {
        id: 'msg-inj',
        role: 'assistant',
        content: '',
        blockedPattern: 'ignore previous instructions',
      },
    ];

    render(
      <AnswerCanvas
        messages={messages}
        inputValue=""
        onChangeInput={vi.fn()}
        onSendMessage={vi.fn()}
        onStopStreaming={vi.fn()}
        isStreaming={false}
        onSelectCitation={vi.fn()}
      />
    );

    expect(screen.getByText(/تزریق پرامپت شناسایی و توسط گاردریل مسدود شد/)).toBeInTheDocument();
    expect(screen.getByText(/ignore previous instructions/)).toBeInTheDocument();
  });

  it('renders abstain state when is_answerable is false', () => {
    const messages: ChatMessage[] = [
      {
        id: 'msg-abstain',
        role: 'assistant',
        content: 'Insufficient information.',
        isAbstain: true,
      },
    ];

    render(
      <AnswerCanvas
        messages={messages}
        inputValue=""
        onChangeInput={vi.fn()}
        onSendMessage={vi.fn()}
        onStopStreaming={vi.fn()}
        isStreaming={false}
        onSelectCitation={vi.fn()}
      />
    );

    expect(screen.getByText(/عدم کفایت شواهد برای پاسخ مستند/)).toBeInTheDocument();
  });
});
