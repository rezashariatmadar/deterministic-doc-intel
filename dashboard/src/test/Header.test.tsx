import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Header } from '../components/Header';
import { HealthStatus } from '../types';

describe('Header Component', () => {
  const mockHealth: HealthStatus = {
    status: 'ok',
    database: 'connected',
    redis: 'connected',
    timestamp: new Date().toISOString(),
  };

  it('renders branding and health indicators when connected', () => {
    const onNew = vi.fn();
    const onRefresh = vi.fn();

    render(
      <Header
        health={mockHealth}
        healthLoading={false}
        onNewConversation={onNew}
        onRefreshHealth={onRefresh}
      />
    );

    expect(screen.getByText('داشبورد مهندسی رگ')).toBeInTheDocument();
    expect(screen.getByText('مکالمه جدید')).toBeInTheDocument();
    const connectedBadges = screen.getAllByText('متصل');
    expect(connectedBadges.length).toBeGreaterThanOrEqual(2);
  });

  it('triggers onNewConversation callback when clicked', () => {
    const onNew = vi.fn();
    const onRefresh = vi.fn();

    render(
      <Header
        health={mockHealth}
        healthLoading={false}
        onNewConversation={onNew}
        onRefreshHealth={onRefresh}
      />
    );

    fireEvent.click(screen.getByText('مکالمه جدید'));
    expect(onNew).toHaveBeenCalledTimes(1);
  });
});
